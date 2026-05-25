package com.one5.personremoval.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * On-device LaMa inpainting via ONNX Runtime Mobile.
 *
 * The Carve LaMa export expects:
 *   - input "image": float32 [1, 3, H, W], range [0,1], channel-first
 *   - input "mask":  float32 [1, 1, H, W], 1.0 where the model should hallucinate
 *   - output "output": float32 [1, 3, H, W]. Carve's lama_fp32.onnx emits values
 *     in roughly [0, 255]; other LaMa exports emit [0, 1]. We auto-detect the
 *     range on the first inference (see [outputScale]) and rescale before
 *     truncating to bytes — otherwise a [0,1]-output model would render as
 *     solid black (every pixel rounding to 0).
 *
 * The model is fully convolutional, so H,W can be any multiple of 8. We use 512×512
 * which trades a bit of resize blur for inference speed. The caller's image is
 * upsampled in and downsampled back out.
 *
 * Construction extracts the .onnx asset to internal storage on first launch and
 * then hands the path to ORT, which mmaps it natively. This avoids loading the
 * full 100+ MB model through the JVM heap (readBytes() peaks at ~2x the file size
 * due to the underlying ByteArrayOutputStream doubling pattern, and we'd OOM at
 * capture time with the rest of the live state on the heap). The extracted copy
 * persists in filesDir until app data is cleared.
 *
 * Failure modes (init or inference) throw — the caller is expected to catch and fall
 * back to the OpenCV inpaint path.
 */
class LamaInpainter(
    context: Context,
    private val modelAssetFp16: String = "lama_fp16.onnx",
    private val modelAssetFp32: String = "lama.onnx",
    private val workingSize: Int = 512
) : AutoCloseable {

    companion object {
        private const val TAG = "LamaInpainter"
        private const val INPUT_IMG  = "image"
        private const val INPUT_MASK = "mask"
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private var outputScale: Float? = null

    init {
        val modelCandidates = resolveModelCandidates(context)
        session = createSession(modelCandidates)

        val ins = session.inputInfo
        val outs = session.outputInfo
        Log.i(TAG, "LaMa loaded; inputs=${ins.keys} outputs=${outs.keys} workingSize=$workingSize")
    }

    private fun buildSessionOptions(): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        try {
            opts.addNnapi()
            Log.i(TAG, "NNAPI execution provider enabled")
        } catch (t: Throwable) {
            Log.w(TAG, "NNAPI unavailable: ${t.message}")
        }
        try {
            opts.addXnnpack(emptyMap())
            Log.i(TAG, "XNNPACK execution provider enabled")
        } catch (t: Throwable) {
            Log.w(TAG, "XNNPACK unavailable: ${t.message}")
        }
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        return opts
    }

    private fun createSession(candidates: List<String>): OrtSession {
        for ((i, path) in candidates.withIndex()) {
            try {
                val s = env.createSession(path, buildSessionOptions())
                Log.i(TAG, "Session created from: $path")
                return s
            } catch (t: Throwable) {
                if (i < candidates.lastIndex) {
                    Log.w(TAG, "Failed to load $path, trying next: ${t.message}")
                } else {
                    throw t
                }
            }
        }
        throw IllegalStateException("No model candidates")
    }

    private fun resolveModelCandidates(context: Context): List<String> {
        val candidates = mutableListOf<String>()
        try {
            context.assets.openFd(modelAssetFp16).close()
            Log.i(TAG, "FP16 model found: $modelAssetFp16")
            candidates.add(ensureModelOnDisk(context, modelAssetFp16))
        } catch (_: IOException) {
            Log.i(TAG, "FP16 model not found, skipping")
        }
        candidates.add(ensureModelOnDisk(context, modelAssetFp32))
        return candidates
    }

    /**
     * Stream the .onnx asset to internal storage if it's not already there, and
     * return the absolute path for ORT.
     *
     * - If the destination file exists with the same length as the asset, reuse it.
     * - Otherwise stream-copy via an 8 KB buffer to a `.tmp` sibling, then rename
     *   atomically. On any failure, delete the temp so we don't leave a partial.
     *
     * Streaming keeps the peak JVM allocation at ~8 KB regardless of model size,
     * vs. `readBytes()` which peaks at ~2x file size (≈200 MB → OOM).
     */
    private fun ensureModelOnDisk(context: Context, asset: String): String {
        val outFile = File(context.filesDir, asset)
        val expectedSize: Long = try {
            context.assets.openFd(asset).use { it.declaredLength }
        } catch (_: IOException) {
            // .onnx is stored uncompressed (noCompress in build.gradle.kts), so
            // openFd should always succeed. If it doesn't, we'll fall through
            // and just re-extract every time.
            -1L
        }
        if (outFile.exists() && (expectedSize < 0 || outFile.length() == expectedSize)) {
            Log.i(TAG, "reusing cached model at ${outFile.absolutePath}")
            return outFile.absolutePath
        }

        Log.i(TAG, "extracting $asset → ${outFile.absolutePath} (one-time)")
        val tempFile = File(context.filesDir, "$asset.tmp")
        try {
            context.assets.open(asset).use { input ->
                tempFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            if (outFile.exists()) outFile.delete()
            if (!tempFile.renameTo(outFile)) {
                tempFile.delete()
                throw IOException("rename $tempFile → $outFile failed")
            }
        } catch (t: Throwable) {
            tempFile.delete()
            throw t
        }
        return outFile.absolutePath
    }

    /**
     * Inpaints [rgb] inside the non-zero regions of [mask] and returns a new RGB buffer.
     * The output covers the entire image — pixels outside the mask are alpha-blended
     * from the input so seams from the resize round-trip don't show.
     *
     * @param rgb    raw bytes, length = width*height*3 (channels-last)
     * @param mask   raw bytes, length = width*height (non-zero = inpaint)
     */
    fun inpaint(rgb: ByteArray, mask: ByteArray, width: Int, height: Int): ByteArray {
        require(rgb.size >= width * height * 3) { "rgb too small" }
        require(mask.size >= width * height) { "mask too small" }

        val w = workingSize
        val h = workingSize

        // 1. Resize image + mask to working size via Bitmap.scale (fastest available on Android).
        val srcBmp = rgbBytesToBitmap(rgb, width, height)
        val scaledBmp = srcBmp.scale(w, h)
        srcBmp.recycle()
        val resizedRgb = bitmapToRgbBytes(scaledBmp)
        scaledBmp.recycle()

        val resizedMask = nearestResizeMask(mask, width, height, w, h)

        // 2. Build input tensors. ONNX wants channel-first float32.
        //    IMPORTANT: zero out the masked pixels in the image before sending.
        //    LaMa exports (especially Carve's) expect the inpaint region to be black,
        //    not the original pixels. Sending the original RGB through the hole was
        //    producing all-white output.
        val imgBuf = FloatBuffer.allocate(1 * 3 * w * h)
        val planeR = 0
        val planeG = w * h
        val planeB = 2 * w * h
        val imgArr = imgBuf.array()
        val maskBuf = FloatBuffer.allocate(1 * 1 * w * h)
        val maskArr = maskBuf.array()
        for (i in 0 until w * h) {
            val isHole = resizedMask[i].toInt() != 0
            maskArr[i] = if (isHole) 1f else 0f
            if (isHole) {
                imgArr[planeR + i] = 0f
                imgArr[planeG + i] = 0f
                imgArr[planeB + i] = 0f
            } else {
                imgArr[planeR + i] = (resizedRgb[i * 3].toInt() and 0xFF) / 255f
                imgArr[planeG + i] = (resizedRgb[i * 3 + 1].toInt() and 0xFF) / 255f
                imgArr[planeB + i] = (resizedRgb[i * 3 + 2].toInt() and 0xFF) / 255f
            }
        }

        val imgTensor = OnnxTensor.createTensor(env, imgBuf, longArrayOf(1, 3, h.toLong(), w.toLong()))
        val maskTensor = OnnxTensor.createTensor(env, maskBuf, longArrayOf(1, 1, h.toLong(), w.toLong()))

        // 3. Run inference. Use the first input/output names from the session in case
        //    Carve's export uses different naming than our defaults.
        val inputNames = session.inputNames.toList()
        val outputNames = session.outputNames.toList()
        val inputs = mutableMapOf<String, OnnxTensor>()
        // Heuristic: 4-channel single input means image+mask concatenated; most LaMa
        // exports use two separate inputs. We pick by name if our defaults match,
        // otherwise by position.
        if (INPUT_IMG in inputNames && INPUT_MASK in inputNames) {
            inputs[INPUT_IMG] = imgTensor
            inputs[INPUT_MASK] = maskTensor
        } else if (inputNames.size >= 2) {
            inputs[inputNames[0]] = imgTensor
            inputs[inputNames[1]] = maskTensor
        } else {
            imgTensor.close(); maskTensor.close()
            throw IllegalStateException("Unexpected LaMa input layout: $inputNames")
        }

        val t0 = System.currentTimeMillis()
        val result = session.run(inputs, setOf(outputNames[0]))
        val ms = System.currentTimeMillis() - t0

        val outTensor = result.get(0) as OnnxTensor
        // Output shape is [1, 3, h, w].
        val outArr = outTensor.floatBuffer.array()
        result.close()
        imgTensor.close(); maskTensor.close()

        // Auto-detect output range on first run. LaMa exports differ: some emit
        // [0,255], others [0,1]. Without rescaling, a [0,1]-output model rounds
        // every byte to 0 → solid black inpaint.
        if (outputScale == null) {
            var mn = Float.POSITIVE_INFINITY
            var mx = Float.NEGATIVE_INFINITY
            var nanCount = 0
            for (v in outArr) {
                if (v.isNaN()) { nanCount++; continue }
                if (v < mn) mn = v
                if (v > mx) mx = v
            }
            val scale = if (mx <= 2.0f) 255f else 1f
            outputScale = scale
            Log.i(TAG, "output stats: min=$mn max=$mx nan=$nanCount → scale=$scale " +
                    "samples=[${outArr[0]}, ${outArr[w * h / 2]}, ${outArr[w * h - 1]}]")
        }
        val scale = outputScale ?: 1f

        // 4. Convert output back to channels-last bytes at working size.
        val outRgbWorking = ByteArray(w * h * 3)
        for (i in 0 until w * h) {
            val r = outArr[planeR + i] * scale
            val g = outArr[planeG + i] * scale
            val b = outArr[planeB + i] * scale
            outRgbWorking[i * 3]     = floatPixelToByte(r)
            outRgbWorking[i * 3 + 1] = floatPixelToByte(g)
            outRgbWorking[i * 3 + 2] = floatPixelToByte(b)
        }

        // 5. Resize back to source size.
        val outBmpWorking = rgbBytesToBitmap(outRgbWorking, w, h)
        val outBmpFinal = outBmpWorking.scale(width, height)
        outBmpWorking.recycle()
        val outRgbFinal = bitmapToRgbBytes(outBmpFinal)
        outBmpFinal.recycle()

        // 6. Composite: only replace mask pixels in the original. Keeps stitched
        //    pixels pixel-perfect and only LaMa-inpaints the holes.
        val composed = rgb.copyOf()
        for (i in 0 until width * height) {
            if (mask[i].toInt() != 0) {
                composed[i * 3]     = outRgbFinal[i * 3]
                composed[i * 3 + 1] = outRgbFinal[i * 3 + 1]
                composed[i * 3 + 2] = outRgbFinal[i * 3 + 2]
            }
        }

        Log.i(TAG, "inpaint: src=${width}x${height} work=${w}x${h} infer=${ms}ms")
        return composed
    }

    /**
     * Strategy-A inpainter. Finds each connected gap region in [mask],
     * expands its bounding box by [paddingPx] for surrounding context, crops
     * image + mask to that bbox, runs LaMa on the crop, and composites the
     * result back into the source at the gap location.
     *
     * Versus [inpaint] (Strategy B — downsamples the whole frame to 512×512):
     *  - Same per-call inference cost (LaMa always runs at 512×512).
     *  - **Higher quality** for small/medium gaps: the gap fills more of the
     *    model's 512×512 input, so more output pixels are devoted to it and
     *    less is wasted on the wide untouched surroundings.
     *  - **Slower** for multi-gap scenes (one inference call per gap).
     *
     * For the typical 1-person-1-gap case this is roughly equal in speed to
     * [inpaint] but visibly sharper inside the patch.
     *
     * Components smaller than [minPixels] are filtered out — they're too tiny
     * for LaMa to fill usefully and the surrounding pixels are visually
     * indistinguishable.
     */
    fun inpaintGaps(
        rgb: ByteArray,
        mask: ByteArray,
        width: Int,
        height: Int,
        paddingPx: Int = 60,
        minPixels: Int = 30
    ): ByteArray {
        require(rgb.size >= width * height * 3) { "rgb too small" }
        require(mask.size >= width * height) { "mask too small" }

        val components = findConnectedComponents(mask, width, height, minPixels)
        if (components.isEmpty()) {
            Log.i(TAG, "inpaintGaps: no qualifying components, returning rgb unchanged")
            return rgb
        }

        Log.i(TAG, "inpaintGaps: ${components.size} component(s) in ${width}x${height}")
        val result = rgb.copyOf()

        for ((idx, comp) in components.withIndex()) {
            // Expand bbox with padding, clamp to image bounds.
            val x0 = (comp.left - paddingPx).coerceAtLeast(0)
            val y0 = (comp.top - paddingPx).coerceAtLeast(0)
            val x1 = (comp.right + paddingPx).coerceAtMost(width)
            val y1 = (comp.bottom + paddingPx).coerceAtMost(height)
            val cw = x1 - x0
            val ch = y1 - y0
            if (cw <= 0 || ch <= 0) continue

            // Crop image + mask from the running result.
            val cropRgb = ByteArray(cw * ch * 3)
            val cropMask = ByteArray(cw * ch)
            for (y in 0 until ch) {
                val srcRow = (y0 + y) * width
                val dstRow = y * cw
                for (x in 0 until cw) {
                    val srcIdx = srcRow + (x0 + x)
                    val dstIdx = dstRow + x
                    cropMask[dstIdx] = mask[srcIdx]
                    cropRgb[dstIdx * 3]     = result[srcIdx * 3]
                    cropRgb[dstIdx * 3 + 1] = result[srcIdx * 3 + 1]
                    cropRgb[dstIdx * 3 + 2] = result[srcIdx * 3 + 2]
                }
            }

            Log.i(TAG, "inpaintGaps: component $idx bbox=(${comp.left},${comp.top})-" +
                    "(${comp.right},${comp.bottom}) crop=${cw}x${ch}")

            // Recurse into the single-pass inpainter on this crop.
            val inpainted = inpaint(cropRgb, cropMask, cw, ch)

            // Composite ONLY where the original (uncropped) mask was non-zero.
            // Pixels in the crop padding region keep the original — they were
            // never gaps, they were just context for LaMa.
            for (y in 0 until ch) {
                val srcRow = y * cw
                val dstRow = (y0 + y) * width
                for (x in 0 until cw) {
                    val srcIdx = srcRow + x
                    val dstIdx = dstRow + (x0 + x)
                    if (mask[dstIdx].toInt() != 0) {
                        result[dstIdx * 3]     = inpainted[srcIdx * 3]
                        result[dstIdx * 3 + 1] = inpainted[srcIdx * 3 + 1]
                        result[dstIdx * 3 + 2] = inpainted[srcIdx * 3 + 2]
                    }
                }
            }
        }
        return result
    }

    override fun close() { session.close() }

    // -------------------------- helpers --------------------------- //

    private fun rgbBytesToBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        for (i in 0 until w * h) {
            val r = rgb[i * 3].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun bitmapToRgbBytes(bmp: Bitmap): ByteArray {
        val w = bmp.width; val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = ByteArray(w * h * 3)
        for (i in 0 until w * h) {
            val p = pixels[i]
            out[i * 3]     = ((p shr 16) and 0xFF).toByte()
            out[i * 3 + 1] = ((p shr 8) and 0xFF).toByte()
            out[i * 3 + 2] = (p and 0xFF).toByte()
        }
        return out
    }

    private fun nearestResizeMask(src: ByteArray, sw: Int, sh: Int, dw: Int, dh: Int): ByteArray {
        val out = ByteArray(dw * dh)
        for (y in 0 until dh) {
            val sy = (y.toFloat() * sh / dh).toInt().coerceIn(0, sh - 1)
            val rowOff = y * dw
            val srcRow = sy * sw
            for (x in 0 until dw) {
                val sx = (x.toFloat() * sw / dw).toInt().coerceIn(0, sw - 1)
                out[rowOff + x] = src[srcRow + sx]
            }
        }
        return out
    }

    private fun floatPixelToByte(v: Float): Byte {
        val clamped = max(0f, min(255f, v))
        return clamped.toInt().toByte()
    }

    private fun Bitmap.scale(w: Int, h: Int): Bitmap {
        return Bitmap.createScaledBitmap(this, w, h, /* filter = */ true)
    }

    /**
     * Find connected components (4-connected) in a binary mask. Returns each
     * component's bounding box [left, top, right, bottom) with right/bottom
     * exclusive (standard half-open-rect convention). Components smaller than
     * [minPixels] are filtered out.
     *
     * Iterative stack-based flood-fill (not recursion — Kotlin's stack would
     * blow up on a 240×320 mask with a single large component, ~75 000 deep).
     */
    private fun findConnectedComponents(
        mask: ByteArray, w: Int, h: Int, minPixels: Int
    ): List<BBox> {
        val visited = BooleanArray(w * h)
        val components = mutableListOf<BBox>()
        val stack = ArrayDeque<Int>()

        for (startIdx in 0 until w * h) {
            if (visited[startIdx] || mask[startIdx].toInt() == 0) continue

            stack.clear()
            stack.addLast(startIdx)
            var minX = w; var maxX = -1
            var minY = h; var maxY = -1
            var count = 0

            while (stack.isNotEmpty()) {
                val idx = stack.removeLast()
                if (visited[idx]) continue
                if (mask[idx].toInt() == 0) continue
                visited[idx] = true
                count++
                val x = idx % w
                val y = idx / w
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y

                if (x > 0)      stack.addLast(idx - 1)
                if (x < w - 1)  stack.addLast(idx + 1)
                if (y > 0)      stack.addLast(idx - w)
                if (y < h - 1)  stack.addLast(idx + w)
            }

            if (count >= minPixels) {
                components.add(BBox(minX, minY, maxX + 1, maxY + 1))
            }
        }
        return components
    }

    /** Half-open bounding rect: [left, right) × [top, bottom). */
    private data class BBox(val left: Int, val top: Int, val right: Int, val bottom: Int)
}
