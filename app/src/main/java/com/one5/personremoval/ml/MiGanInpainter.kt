package com.one5.personremoval.ml

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer

/**
 * On-device MI-GAN inpainting via ONNX Runtime (Picsart MI-GAN, ICCV 2023).
 *
 * Uses the self-contained `migan_pipeline_v2.onnx` pipeline export, which does
 * ALL pre/post-processing inside the graph: it crops to the masked region,
 * resizes to the model's 512 working size, runs the GAN (single forward pass —
 * no FFC, no denoising steps), pastes the result back, and blends the seam.
 * So the caller just hands it the full frame + mask and gets the full frame
 * back with the hole filled. Far simpler than LaMa's manual crop/letterbox/
 * composite, and ~100× faster on-device (a single conv GAN vs LaMa's FFC,
 * which has no mobile acceleration).
 *
 * I/O contract (verified against the export):
 *   - input  "image" : uint8  [1, 3, H, W]  (CHW, RGB, arbitrary H/W)
 *   - input  "mask"  : uint8  [1, 1, H, W]  — 255 = KEEP, 0 = HOLE (inpaint).
 *                      NOTE: inverted vs the rest of this app, where non-zero
 *                      marks the hole. We invert on the way in.
 *   - output "result": uint8  [1, 3, H, W]  (CHW, RGB); known region preserved.
 *
 * Failure (init or inference) throws — the caller catches and falls back to the
 * OpenCV path, same as LaMa.
 */
class MiGanInpainter(
    context: Context,
    private val modelAsset: String = "migan_pipeline_v2.onnx"
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val imgInputName: String
    private val maskInputName: String
    private val outputName: String

    init {
        val modelPath = ensureModelOnDisk(context, modelAsset)
        session = createSession(modelPath)
        val inputs = session.inputNames.toList()
        // The export names them "image"/"mask"; fall back to positional if a
        // future export renames them.
        imgInputName = if (INPUT_IMG in inputs) INPUT_IMG else inputs[0]
        maskInputName = if (INPUT_MASK in inputs) INPUT_MASK else inputs[1]
        outputName = session.outputNames.first()
        Log.i(TAG, "MI-GAN loaded; inputs=$inputs outputs=${session.outputNames.toList()}")
    }

    /**
     * Inpaint [rgb] inside the non-zero regions of [mask] and return a new full
     * RGB buffer (channels-last, w*h*3). The pipeline preserves the known
     * region, so pixels outside the mask come back unchanged.
     *
     * @param rgb  raw bytes, length = width*height*3 (channels-last RGB)
     * @param mask raw bytes, length = width*height (non-zero = inpaint here)
     */
    fun inpaint(rgb: ByteArray, mask: ByteArray, width: Int, height: Int): ByteArray? {
        require(rgb.size >= width * height * 3) { "rgb too small" }
        require(mask.size >= width * height) { "mask too small" }
        val hw = width * height

        // image: HWC (interleaved) -> CHW planar uint8.
        val imgBuf = ByteBuffer.allocateDirect(3 * hw)
        run {
            var r = 0; var g = hw; var b = 2 * hw
            val arr = ByteArray(3 * hw)
            for (i in 0 until hw) {
                arr[r++] = rgb[i * 3]
                arr[g++] = rgb[i * 3 + 1]
                arr[b++] = rgb[i * 3 + 2]
            }
            imgBuf.put(arr); imgBuf.rewind()
        }

        // mask: our convention (non-zero = hole) -> MI-GAN (0 = hole, 255 = keep).
        val maskBuf = ByteBuffer.allocateDirect(hw)
        run {
            val arr = ByteArray(hw)
            for (i in 0 until hw) arr[i] = if (mask[i].toInt() != 0) 0 else 255.toByte()
            maskBuf.put(arr); maskBuf.rewind()
        }

        val imgTensor = OnnxTensor.createTensor(
            env, imgBuf, longArrayOf(1, 3, height.toLong(), width.toLong()), OnnxJavaType.UINT8)
        val maskTensor = OnnxTensor.createTensor(
            env, maskBuf, longArrayOf(1, 1, height.toLong(), width.toLong()), OnnxJavaType.UINT8)

        val t0 = System.currentTimeMillis()
        val result = session.run(mapOf(imgInputName to imgTensor, maskInputName to maskTensor),
                                 setOf(outputName))
        val ms = System.currentTimeMillis() - t0

        val outT = result.get(0) as OnnxTensor
        val outBuf = outT.byteBuffer  // CHW uint8, 3*hw
        // CHW planar -> HWC interleaved.
        val out = ByteArray(hw * 3)
        run {
            var r = 0; var g = hw; var b = 2 * hw
            for (i in 0 until hw) {
                out[i * 3]     = outBuf.get(r++)
                out[i * 3 + 1] = outBuf.get(g++)
                out[i * 3 + 2] = outBuf.get(b++)
            }
        }
        result.close(); imgTensor.close(); maskTensor.close()

        Log.i(TAG, "migan inpaint: ${width}x${height} infer=${ms}ms")
        return out
    }

    /** One tiny dummy pass at construction so ORT/delegate init is off the
     *  capture path. MI-GAN is cheap, so this is fast (no 18s warmup like LaMa). */
    fun warmUp() {
        val w = 256; val h = 256
        val rgb = ByteArray(w * h * 3)
        val mask = ByteArray(w * h).apply { for (i in (h / 3 * w) until (2 * h / 3 * w)) this[i] = 1 }
        runCatching {
            val t0 = System.currentTimeMillis()
            inpaint(rgb, mask, w, h)
            Log.i(TAG, "MI-GAN warmUp done in ${System.currentTimeMillis() - t0}ms")
        }.onFailure { Log.w(TAG, "MI-GAN warmUp failed (non-fatal): ${it.message}") }
    }

    override fun close() = session.close()

    private fun createSession(path: String): OrtSession {
        val tiers = listOf(
            Triple("NNAPI+XNNPACK", true, true),
            Triple("XNNPACK-only", false, true),
            Triple("CPU-only", false, false),
        )
        var lastError: Throwable? = null
        for ((label, useNn, useXn) in tiers) {
            try {
                val opts = OrtSession.SessionOptions()
                if (useNn) runCatching { opts.addNnapi() }
                if (useXn) runCatching { opts.addXnnpack(emptyMap()) }
                opts.setIntraOpNumThreads(4)
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                val s = env.createSession(path, opts)
                Log.i(TAG, "MI-GAN session created ($label)")
                return s
            } catch (t: Throwable) {
                lastError = t
                Log.w(TAG, "MI-GAN load failed with $label: ${t.message}")
            }
        }
        throw lastError ?: IllegalStateException("MI-GAN: no session")
    }

    /** Stream the .onnx asset to internal storage if not already present
     *  (streams the asset in 8 KB chunks — keeps peak JVM alloc tiny). */
    private fun ensureModelOnDisk(context: Context, asset: String): String {
        val outFile = File(context.filesDir, asset)
        val expectedSize: Long = try {
            context.assets.openFd(asset).use { it.declaredLength }
        } catch (_: IOException) { -1L }
        if (outFile.exists() && (expectedSize < 0 || outFile.length() == expectedSize)) {
            return outFile.absolutePath
        }
        val tempFile = File(context.filesDir, "$asset.tmp")
        try {
            context.assets.open(asset).use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            if (outFile.exists()) outFile.delete()
            if (!tempFile.renameTo(outFile)) {
                tempFile.delete(); throw IOException("rename $tempFile -> $outFile failed")
            }
        } catch (t: Throwable) {
            tempFile.delete(); throw t
        }
        return outFile.absolutePath
    }

    private companion object {
        const val TAG = "MiGanInpainter"
        const val INPUT_IMG = "image"
        const val INPUT_MASK = "mask"
    }
}