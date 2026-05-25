package com.one5.personremoval.ml

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.graphics.RectF
import android.util.Half
import android.util.Log
import com.one5.personremoval.core.Person
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer

/**
 * Alpha-matting refiner backed by a MODNet ONNX model (via ONNX Runtime).
 * Used at capture time on the reference frame to convert coarse binary YOLO
 * masks into sub-pixel alpha mattes (hair / jaw / fingertip precision).
 * Per-frame analyzer masks stay binary YOLO + dilation — refinement only
 * happens once per capture.
 *
 * Expected model: MODNet ONNX export
 *   - Input  shape `[1, 3, H, W]` (NCHW) or `[1, H, W, 3]` (NHWC), FLOAT32.
 *   - Output shape `[1, 1, H, W]` (NCHW) or `[1, H, W, 1]` (NHWC), FLOAT32
 *     alpha matte in [0, 1].
 *
 * Both layouts are detected from the loaded model's input tensor shape.
 * FP16 model variants are also supported transparently (ShortBuffer +
 * `android.util.Half` casts; API 26+, which matches this app's minSdk).
 *
 * Two preprocessing schemes are toggled via [normalizeMode]:
 *   - UNIT: `rgb / 255 → [0, 1]`. Used by some MODNet TFLite exports.
 *   - SYMMETRIC: `(rgb / 255 - 0.5) / 0.5 → [-1, 1]`. Default for the
 *     official MODNet ONNX export.
 *
 * If the model asset is missing or fails to load, [isAvailable] stays false
 * and [refineRemoveMask] returns the combined binary YOLO mask unchanged —
 * the rest of the pipeline keeps working at YOLO precision instead of
 * crashing.
 */
class MattingRefiner(
    context: Context,
    private val modelAsset: String = "modnet.onnx",
    private val workingSize: Int = 512,
    private val paddingFrac: Float = 0.15f,
    private val normalizeMode: NormalizeMode = NormalizeMode.SYMMETRIC
) : AutoCloseable {

    enum class NormalizeMode { UNIT, SYMMETRIC }

    companion object {
        private const val TAG = "MattingRefiner"
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession?
    private val isNhwc: Boolean
    private val inputDtype: OnnxJavaType
    private val outputDtype: OnnxJavaType
    private val inputName: String?
    private val outputName: String?

    val isAvailable: Boolean get() = session != null

    init {
        val load = tryLoad(context)
        session = load.session
        isNhwc = load.nhwc
        inputDtype = load.inputDtype
        outputDtype = load.outputDtype
        inputName = load.inputName
        outputName = load.outputName

        if (session != null) {
            Log.i(TAG, "MattingRefiner loaded ($modelAsset). " +
                    "layout=${if (isNhwc) "NHWC" else "NCHW"} working=$workingSize " +
                    "in=$inputName($inputDtype) out=$outputName($outputDtype) " +
                    "normalize=$normalizeMode")
        } else {
            Log.i(TAG, "MattingRefiner unavailable (asset $modelAsset missing or load failed); " +
                    "refinement disabled, binary YOLO masks passed through")
        }
    }

    /**
     * For each person in [persons] with state REMOVE, refine the binary YOLO
     * mask in [referenceRgb] using MODNet, then merge all refined alphas
     * into a single combined REMOVE mask (max-blended at overlaps).
     *
     * @return a w×h byte mask, 0 = keep, 255 = remove. Falls back to the
     *         combined binary YOLO mask if the model is not loaded.
     */
    fun refineRemoveMask(
        referenceRgb: ByteArray,
        width: Int,
        height: Int,
        persons: List<Person>
    ): ByteArray {
        val combined = ByteArray(width * height)

        // Fallback: model not loaded → OR every person's binary mask.
        if (session == null) {
            for (p in persons) {
                if (p.maskWidth != width || p.maskHeight != height) continue
                for (i in 0 until width * height) {
                    if (p.mask[i].toInt() != 0) combined[i] = 0xFF.toByte()
                }
            }
            return combined
        }

        val srcBmp = rgbBytesToBitmap(referenceRgb, width, height)
        try {
            for (p in persons) {
                if (p.maskWidth != width || p.maskHeight != height) continue
                val bbox = expandBbox(p.bBox, width, height, paddingFrac)
                if (bbox.width() <= 0 || bbox.height() <= 0) continue

                val alphaCrop = inferOnBbox(srcBmp, bbox) ?: continue
                pasteAlphaIntoMask(combined, width, height, alphaCrop, bbox)
            }
        } finally {
            srcBmp.recycle()
        }
        return combined
    }

    private fun inferOnBbox(srcBmp: Bitmap, bbox: Rect): FloatArray? {
        val cropBmp = Bitmap.createBitmap(bbox.width(), bbox.height(), Bitmap.Config.ARGB_8888)
        Canvas(cropBmp).drawBitmap(
            srcBmp,
            Rect(bbox.left, bbox.top, bbox.right, bbox.bottom),
            Rect(0, 0, bbox.width(), bbox.height()),
            null
        )
        val resized = Bitmap.createScaledBitmap(cropBmp, workingSize, workingSize, true)
        cropBmp.recycle()

        return try {
            val tensor = buildInputTensor(resized)
            val t0 = System.currentTimeMillis()
            val raw = runInference(tensor)
            val ms = System.currentTimeMillis() - t0
            tensor.close()
            Log.i(TAG, "matting infer ${workingSize}x$workingSize ${ms}ms (bbox=${bbox.width()}x${bbox.height()})")
            resizeAlphaBilinear(raw, workingSize, workingSize, bbox.width(), bbox.height())
        } catch (t: Throwable) {
            Log.w(TAG, "matting infer failed: ${t.message}")
            null
        } finally {
            resized.recycle()
        }
    }

    private fun buildInputTensor(bmp: Bitmap): OnnxTensor {
        val w = workingSize
        val pixels = IntArray(w * w)
        bmp.getPixels(pixels, 0, w, 0, 0, w, w)

        val scale: Float
        val bias: Float
        when (normalizeMode) {
            NormalizeMode.UNIT       -> { scale = 1f / 255f;   bias = 0f }
            NormalizeMode.SYMMETRIC  -> { scale = 1f / 127.5f; bias = -1f }
        }

        val floats = FloatArray(3 * w * w)
        if (isNhwc) {
            // [1, H, W, 3] interleaved.
            for (i in 0 until w * w) {
                val p = pixels[i]
                floats[i * 3]     = ((p shr 16) and 0xFF) * scale + bias
                floats[i * 3 + 1] = ((p shr 8) and 0xFF) * scale + bias
                floats[i * 3 + 2] = (p and 0xFF) * scale + bias
            }
        } else {
            // [1, 3, H, W] channel-first: R plane, G plane, B plane.
            val planeR = 0
            val planeG = w * w
            val planeB = 2 * w * w
            for (i in 0 until w * w) {
                val p = pixels[i]
                floats[planeR + i] = ((p shr 16) and 0xFF) * scale + bias
                floats[planeG + i] = ((p shr 8) and 0xFF) * scale + bias
                floats[planeB + i] = (p and 0xFF) * scale + bias
            }
        }

        val shape = if (isNhwc) longArrayOf(1, w.toLong(), w.toLong(), 3)
                    else longArrayOf(1, 3, w.toLong(), w.toLong())

        return when (inputDtype) {
            OnnxJavaType.FLOAT -> OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)
            OnnxJavaType.FLOAT16 -> {
                // ORT 1.18 Java rejects ShortBuffer for FP16 via the typed
                // overload; pack halves into a direct native-order ByteBuffer
                // and pass through the (Buffer, shape, type) form.
                val n = floats.size
                val bb = ByteBuffer.allocateDirect(n * 2).order(ByteOrder.nativeOrder())
                val sb = bb.asShortBuffer()
                for (i in 0 until n) sb.put(i, Half.toHalf(floats[i]))
                OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.FLOAT16)
            }
            else -> throw IllegalStateException("Unsupported MODNet input dtype: $inputDtype")
        }
    }

    private fun runInference(input: OnnxTensor): FloatArray {
        val w = workingSize
        val sess = session ?: throw IllegalStateException("session not loaded")
        val inputMap = mapOf((inputName ?: sess.inputNames.first()) to input)
        val result = sess.run(inputMap, setOf(outputName ?: sess.outputNames.first()))
        try {
            val outTensor = result.get(0) as OnnxTensor
            val expectedSize = w * w
            val flat = FloatArray(expectedSize)
            when (outputDtype) {
                OnnxJavaType.FLOAT -> {
                    val buf = outTensor.floatBuffer
                    buf.get(flat, 0, minOf(expectedSize, buf.remaining()))
                }
                OnnxJavaType.FLOAT16 -> {
                    // FP16 output exposed via raw ByteBuffer. Each pair of
                    // native-order bytes is a half-precision float bit
                    // pattern. ByteBuffer path avoids ORT-version
                    // differences in short-buffer dtype gating.
                    val bb = outTensor.byteBuffer.order(ByteOrder.nativeOrder())
                    val sb = bb.asShortBuffer()
                    val n = minOf(expectedSize, sb.remaining())
                    for (i in 0 until n) flat[i] = Half.toFloat(sb.get(i))
                }
                else -> throw IllegalStateException("Unsupported MODNet output dtype: $outputDtype")
            }
            return flat
        } finally {
            result.close()
        }
    }

    private fun pasteAlphaIntoMask(
        dst: ByteArray, dw: Int, dh: Int,
        alpha: FloatArray, bbox: Rect
    ) {
        val cw = bbox.width()
        val ch = bbox.height()
        if (alpha.size < cw * ch) return
        for (y in 0 until ch) {
            val dstY = bbox.top + y
            if (dstY !in 0 until dh) continue
            val srcRow = y * cw
            val dstRow = dstY * dw
            for (x in 0 until cw) {
                val dstX = bbox.left + x
                if (dstX !in 0 until dw) continue
                val a = (alpha[srcRow + x].coerceIn(0f, 1f) * 255f).toInt()
                val cur = dst[dstRow + dstX].toInt() and 0xFF
                if (a > cur) dst[dstRow + dstX] = a.toByte()
            }
        }
    }

    private fun expandBbox(box: RectF, w: Int, h: Int, frac: Float): Rect {
        val cx = (box.left + box.right) * 0.5f
        val cy = (box.top + box.bottom) * 0.5f
        val halfW = (box.width() * 0.5f) * (1f + frac)
        val halfH = (box.height() * 0.5f) * (1f + frac)
        val l = (cx - halfW).toInt().coerceIn(0, w - 1)
        val t = (cy - halfH).toInt().coerceIn(0, h - 1)
        val r = (cx + halfW).toInt().coerceIn(l + 1, w)
        val b = (cy + halfH).toInt().coerceIn(t + 1, h)
        return Rect(l, t, r, b)
    }

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

    private fun resizeAlphaBilinear(
        src: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int
    ): FloatArray {
        val out = FloatArray(dw * dh)
        val scaleX = sw.toFloat() / dw
        val scaleY = sh.toFloat() / dh
        for (y in 0 until dh) {
            val sy = y * scaleY
            val y0 = sy.toInt().coerceIn(0, sh - 1)
            val y1 = (y0 + 1).coerceAtMost(sh - 1)
            val fy = sy - y0
            val srcRow0 = y0 * sw
            val srcRow1 = y1 * sw
            val dstRow = y * dw
            for (x in 0 until dw) {
                val sx = x * scaleX
                val x0 = sx.toInt().coerceIn(0, sw - 1)
                val x1 = (x0 + 1).coerceAtMost(sw - 1)
                val fx = sx - x0
                val v00 = src[srcRow0 + x0]
                val v01 = src[srcRow0 + x1]
                val v10 = src[srcRow1 + x0]
                val v11 = src[srcRow1 + x1]
                val v0 = v00 * (1 - fx) + v01 * fx
                val v1 = v10 * (1 - fx) + v11 * fx
                out[dstRow + x] = v0 * (1 - fy) + v1 * fy
            }
        }
        return out
    }

    override fun close() { session?.close() }

    // -------------------------- model loading --------------------------- //

    private data class LoadResult(
        val session: OrtSession?,
        val nhwc: Boolean,
        val inputDtype: OnnxJavaType,
        val outputDtype: OnnxJavaType,
        val inputName: String?,
        val outputName: String?
    )

    private fun tryLoad(context: Context): LoadResult {
        val onDiskPath = try {
            ensureModelOnDisk(context, modelAsset)
        } catch (e: IOException) {
            Log.i(TAG, "$modelAsset not found in assets; refiner disabled")
            return LoadResult(null, true, OnnxJavaType.FLOAT, OnnxJavaType.FLOAT, null, null)
        } catch (t: Throwable) {
            Log.w(TAG, "$modelAsset extract failed: ${t.message}; refiner disabled")
            return LoadResult(null, true, OnnxJavaType.FLOAT, OnnxJavaType.FLOAT, null, null)
        }

        val sess = try {
            env.createSession(onDiskPath, buildSessionOptions())
        } catch (t: Throwable) {
            Log.w(TAG, "MODNet session create failed: ${t.message}; refiner disabled")
            return LoadResult(null, true, OnnxJavaType.FLOAT, OnnxJavaType.FLOAT, null, null)
        }

        val ins = sess.inputInfo
        val outs = sess.outputInfo
        val inName = ins.keys.first()
        val outName = outs.keys.first()
        val inInfo = ins[inName]?.info as? TensorInfo
        val outInfo = outs[outName]?.info as? TensorInfo
        val inShape = inInfo?.shape ?: longArrayOf(1, 3, workingSize.toLong(), workingSize.toLong())
        val outShape = outInfo?.shape ?: longArrayOf()

        // NHWC if the last dim of the input is 3 (channels). Otherwise NCHW.
        val nhwc = inShape.size == 4 && inShape[3] == 3L
        val inDt = inInfo?.type ?: OnnxJavaType.FLOAT
        val outDt = outInfo?.type ?: OnnxJavaType.FLOAT

        Log.i(TAG, "MODNet shape: in=${inShape.toList()} out=${outShape.toList()}")
        return LoadResult(sess, nhwc, inDt, outDt, inName, outName)
    }

    private fun buildSessionOptions(): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        try {
            opts.addNnapi()
            Log.i(TAG, "matting NNAPI execution provider enabled")
        } catch (t: Throwable) {
            Log.w(TAG, "matting NNAPI unavailable: ${t.message}")
        }
        try {
            opts.addXnnpack(emptyMap())
            Log.i(TAG, "matting XNNPACK execution provider enabled")
        } catch (t: Throwable) {
            Log.w(TAG, "matting XNNPACK unavailable: ${t.message}")
        }
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        return opts
    }

    @Throws(IOException::class)
    private fun ensureModelOnDisk(context: Context, asset: String): String {
        val outFile = File(context.filesDir, asset)
        val expectedSize: Long = try {
            context.assets.openFd(asset).use { it.declaredLength }
        } catch (_: IOException) {
            // openFd fails when the asset is missing — propagate as IOException.
            throw IOException("asset $asset not found")
        }
        if (outFile.exists() && outFile.length() == expectedSize) {
            return outFile.absolutePath
        }
        Log.i(TAG, "extracting $asset → ${outFile.absolutePath} (one-time, $expectedSize bytes)")
        val tempFile = File(context.filesDir, "$asset.tmp")
        try {
            context.assets.open(asset).use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
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
}
