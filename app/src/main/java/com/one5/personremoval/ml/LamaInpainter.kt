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
    private val modelAsset: String = "lama.onnx",
    private val workingSize: Int = 512
) : AutoCloseable {

    companion object {
        private const val TAG = "LamaInpainter"
        private const val INPUT_IMG  = "image"
        private const val INPUT_MASK = "mask"
        // Crops larger than this in either dimension trigger the 2×2 tiled
        // path inside inpaintGaps. Below this threshold the single-pass
        // downscale path is sharp enough and 4× cheaper.
        private const val LARGE_CROP_THRESHOLD = 768
        // Fraction of each tile that overlaps with its neighbor. The
        // overlap region carries the feather-blended seam.
        private const val TILE_OVERLAP_FRAC = 0.40f
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private var outputScale: Float? = null

    private val cachedImgInputName: String
    private val cachedMaskInputName: String
    private val cachedOutputName: String

    init {
        val modelPath = ensureModelOnDisk(context, modelAsset)
        session = createSession(listOf(modelPath))

        val inputNames = session.inputNames.toList()
        val outputNames = session.outputNames.toList()
        if (INPUT_IMG in inputNames && INPUT_MASK in inputNames) {
            cachedImgInputName = INPUT_IMG
            cachedMaskInputName = INPUT_MASK
        } else if (inputNames.size >= 2) {
            cachedImgInputName = inputNames[0]
            cachedMaskInputName = inputNames[1]
        } else {
            throw IllegalStateException("Unexpected LaMa input layout: $inputNames")
        }
        cachedOutputName = outputNames[0]

        Log.i(TAG, "LaMa loaded; inputs=$inputNames outputs=$outputNames " +
                "workingSize=$workingSize")
    }

    private fun buildSessionOptions(useNnapi: Boolean, useXnnpack: Boolean):
            OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        if (useNnapi) {
            try {
                opts.addNnapi()
                Log.i(TAG, "NNAPI execution provider enabled")
            } catch (t: Throwable) {
                Log.w(TAG, "NNAPI unavailable: ${t.message}")
            }
        }
        if (useXnnpack) {
            try {
                opts.addXnnpack(emptyMap())
                Log.i(TAG, "XNNPACK execution provider enabled")
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK unavailable: ${t.message}")
            }
        }
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        return opts
    }

    /**
     * Resolve a usable session by walking the candidate list (typically
     * [fp16, fp32]) and within each candidate trying delegate combinations
     * in order of preference. Falls back from NNAPI+XNNPACK → XNNPACK →
     * CPU-only so a delegate that rejects FP16 ops on a given device
     * doesn't take the whole model down. Last attempt's exception is
     * rethrown so callers can surface a meaningful failure.
     */
    private fun createSession(candidates: List<String>): OrtSession {
        // (label, useNnapi, useXnnpack) — applied in order.
        val delegateTiers = listOf(
            Triple("NNAPI+XNNPACK", true,  true),
            Triple("XNNPACK-only",  false, true),
            Triple("CPU-only",      false, false),
        )

        var lastError: Throwable? = null
        for (path in candidates) {
            for ((label, useNn, useXn) in delegateTiers) {
                try {
                    val s = env.createSession(path, buildSessionOptions(useNn, useXn))
                    Log.i(TAG, "Session created from: $path ($label)")
                    return s
                } catch (t: Throwable) {
                    lastError = t
                    Log.w(TAG, "Load failed: $path with $label: ${t.message}")
                }
            }
        }
        throw lastError ?: IllegalStateException("No model candidates")
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
    fun inpaint(rgb: ByteArray, mask: ByteArray, width: Int, height: Int): ByteArray? {
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

        val imgTensor = createInputTensor(imgBuf, longArrayOf(1, 3, h.toLong(), w.toLong()))
        val maskTensor = createInputTensor(maskBuf, longArrayOf(1, 1, h.toLong(), w.toLong()))

        val inputs = mapOf(
            cachedImgInputName to imgTensor,
            cachedMaskInputName to maskTensor
        )

        val t0 = System.currentTimeMillis()
        val result = session.run(inputs, setOf(cachedOutputName))
        val ms = System.currentTimeMillis() - t0

        val outTensor = result.get(0) as OnnxTensor
        // Output shape is [1, 3, h, w]. Read into a FloatArray via the dtype-
        // aware helper so FP16 outputs are unpacked to floats.
        val outArr = readOutputFloats(outTensor, 3 * w * h)
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
            Log.i(TAG, "output stats: min=$mn max=$mx nan=$nanCount → " +
                    "samples=[${outArr[0]}, ${outArr[w * h / 2]}, ${outArr[w * h - 1]}]")
            if (mx == Float.NEGATIVE_INFINITY || nanCount > outArr.size / 2) {
                Log.e(TAG, "LaMa output degenerate (all-NaN or >50% NaN), bailing out")
                return null
            }
            val scale = if (mx <= 2.0f) 255f else 1f
            outputScale = scale
            Log.i(TAG, "output scale=$scale")
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
        minPixels: Int = 30,
        maxLamaCalls: Int = 3
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
        var lamaCalls = 0

        for ((idx, comp) in components.withIndex()) {
            val compW = comp.right - comp.left
            val compH = comp.bottom - comp.top
            val pad = maxOf(paddingPx, compW / 3, compH / 3)
            val x0 = (comp.left - pad).coerceAtLeast(0)
            val y0 = (comp.top - pad).coerceAtLeast(0)
            val x1 = (comp.right + pad).coerceAtMost(width)
            val y1 = (comp.bottom + pad).coerceAtMost(height)
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

            if (lamaCalls >= maxLamaCalls) {
                Log.i(TAG, "inpaintGaps: LaMa call cap ($maxLamaCalls) reached, skipping remaining")
                break
            }

            val inpainted = inpaint(cropRgb, cropMask, cw, ch) ?: continue
            lamaCalls++

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

    /**
     * Inpaint a large crop via a 2×2 grid of overlapping tiles. Each tile
     * is fed to [inpaint] (which resizes to LaMa's 512×512 internally), so
     * the per-tile downscale ratio stays low — sharper output than the
     * one-shot path on big crops.
     *
     * Tile coordinates within the crop (with overlap on the inner edges):
     *   - TL: (0,    0,    tw, th)
     *   - TR: (cw-tw,0,    cw, th)
     *   - BL: (0,    ch-th,tw, ch)
     *   - BR: (cw-tw,ch-th,cw, ch)
     * where tw, th are chosen so each tile covers (cw + overlap*cw) / 2.
     *
     * Composition: each tile contributes to the output weighted by a
     * cosine-tapered window centered on the tile. Overlap pixels see the
     * weighted average of the two (or four) covering tiles. Result =
     * Σ(tileOut × w) / Σ(w).
     */
    private fun inpaintLargeCrop(rgb: ByteArray, mask: ByteArray, cw: Int, ch: Int): ByteArray {
        // Each tile dimension covers slightly more than half the crop to
        // create overlap. ceil ensures union covers the full crop.
        val tw = ((cw * (1f + TILE_OVERLAP_FRAC) / 2f).toInt()).coerceAtMost(cw)
        val th = ((ch * (1f + TILE_OVERLAP_FRAC) / 2f).toInt()).coerceAtMost(ch)

        // Tile origins.
        val tilesXY = listOf(
            0 to 0,
            cw - tw to 0,
            0 to ch - th,
            cw - tw to ch - th
        )

        // Accumulators.
        val accum = FloatArray(cw * ch * 3)
        val weight = FloatArray(cw * ch)

        Log.i(TAG, "inpaintLargeCrop: ${cw}x${ch} via 4 tiles each ${tw}x${th}")
        val tileWeightLut = buildCosineWindow(tw, th)

        for ((tIdx, origin) in tilesXY.withIndex()) {
            val (tx, ty) = origin
            // Crop tile from the running rgb + mask.
            val tileRgb = ByteArray(tw * th * 3)
            val tileMask = ByteArray(tw * th)
            for (y in 0 until th) {
                val srcRow = (ty + y) * cw
                val dstRow = y * tw
                for (x in 0 until tw) {
                    val srcIdx = srcRow + (tx + x)
                    val dstIdx = dstRow + x
                    tileMask[dstIdx] = mask[srcIdx]
                    tileRgb[dstIdx * 3]     = rgb[srcIdx * 3]
                    tileRgb[dstIdx * 3 + 1] = rgb[srcIdx * 3 + 1]
                    tileRgb[dstIdx * 3 + 2] = rgb[srcIdx * 3 + 2]
                }
            }

            val tInpainted = inpaint(tileRgb, tileMask, tw, th)
                ?: continue
            // Accumulate weighted into crop-space.
            for (y in 0 until th) {
                val tileRowBase = y * tw
                val accRowBase = (ty + y) * cw
                for (x in 0 until tw) {
                    val w = tileWeightLut[tileRowBase + x]
                    if (w <= 0f) continue
                    val accIdx = accRowBase + (tx + x)
                    val r = (tInpainted[tileRowBase * 3 + x * 3].toInt() and 0xFF).toFloat()
                    val g = (tInpainted[tileRowBase * 3 + x * 3 + 1].toInt() and 0xFF).toFloat()
                    val b = (tInpainted[tileRowBase * 3 + x * 3 + 2].toInt() and 0xFF).toFloat()
                    accum[accIdx * 3]     += r * w
                    accum[accIdx * 3 + 1] += g * w
                    accum[accIdx * 3 + 2] += b * w
                    weight[accIdx] += w
                }
            }
            Log.i(TAG, "inpaintLargeCrop: tile $tIdx ($tx,$ty)-(${tx+tw},${ty+th}) done")
        }

        // Normalize: result = accum / weight. Where weight is zero (should
        // not happen with cosine window > 0 everywhere except corners),
        // copy original rgb so we don't divide by zero.
        val out = ByteArray(cw * ch * 3)
        for (i in 0 until cw * ch) {
            val w = weight[i]
            if (w > 1e-3f) {
                val inv = 1f / w
                out[i * 3]     = (accum[i * 3] * inv).toInt().coerceIn(0, 255).toByte()
                out[i * 3 + 1] = (accum[i * 3 + 1] * inv).toInt().coerceIn(0, 255).toByte()
                out[i * 3 + 2] = (accum[i * 3 + 2] * inv).toInt().coerceIn(0, 255).toByte()
            } else {
                out[i * 3]     = rgb[i * 3]
                out[i * 3 + 1] = rgb[i * 3 + 1]
                out[i * 3 + 2] = rgb[i * 3 + 2]
            }
        }
        return out
    }

    /**
     * Build a 2D cosine-tapered weight window of size [w]×[h]. Weight is
     * 1 at the center, smoothly decreasing to ~0 at the edges. Separable:
     * w(x,y) = wx(x) * wy(y) where each axis uses 0.5 * (1 - cos(2πi/N)).
     * Sum of overlapping windows along an axis stays ~1 (Hann window
     * property), so cross-tile blending is amplitude-correct.
     */
    private fun buildCosineWindow(w: Int, h: Int): FloatArray {
        val wx = FloatArray(w)
        for (i in 0 until w) {
            wx[i] = (0.5 * (1.0 - kotlin.math.cos(2.0 * Math.PI * i / (w - 1)))).toFloat()
        }
        val wy = FloatArray(h)
        for (i in 0 until h) {
            wy[i] = (0.5 * (1.0 - kotlin.math.cos(2.0 * Math.PI * i / (h - 1)))).toFloat()
        }
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) out[base + x] = wx[x] * wy[y]
        }
        return out
    }

    override fun close() { session.close() }

    // -------------------------- helpers --------------------------- //

    private fun createInputTensor(srcFloats: FloatBuffer, shape: LongArray): OnnxTensor {
        return OnnxTensor.createTensor(env, srcFloats, shape)
    }

    private fun readOutputFloats(tensor: OnnxTensor, expectedSize: Int): FloatArray {
        val buf = tensor.floatBuffer
        val out = FloatArray(expectedSize)
        buf.get(out, 0, minOf(expectedSize, buf.remaining()))
        return out
    }

    private fun rgbBytesToBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        if (com.one5.personremoval.core.NativeSession.fillBitmapFromRgb(rgb, w, h, bmp)) {
            return bmp
        }
        // Native fast path failed (wrong bitmap format / lock failure). Fall
        // back to the original setPixels(IntArray) packer so the inpaint
        // still completes — slower, but never silently produces a black bmp.
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
        com.one5.personremoval.core.NativeSession.readRgbFromBitmap(bmp)?.let { return it }
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
