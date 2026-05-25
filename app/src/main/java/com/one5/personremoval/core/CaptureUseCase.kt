package com.one5.personremoval.core

import android.util.Log
import com.one5.personremoval.ml.LamaInpainter
import com.one5.personremoval.ml.MattingRefiner

data class PipelineResult(
    val rgb: ByteArray,
    val width: Int,
    val height: Int,
    val holeMask: ByteArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as PipelineResult
        if (width != other.width) return false
        if (height != other.height) return false
        if (!rgb.contentEquals(other.rgb)) return false
        if (!holeMask.contentEquals(other.holeMask)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = width
        result = 31 * result + height
        result = 31 * result + rgb.contentHashCode()
        result = 31 * result + holeMask.contentHashCode()
        return result
    }
}

class CaptureUseCase(
    private val nativeSession: NativeSession,
    @Volatile var lamaInpainter: LamaInpainter? = null,
    @Volatile var mattingRefiner: MattingRefiner? = null
) {

    /**
     * Run the capture pipeline.
     *
     * @param referenceRgba  Optional RGBA snapshot of the latest analyzer
     *                       frame. When provided alongside an available
     *                       [mattingRefiner], MODNet refines the binary YOLO
     *                       masks into sub-pixel alpha before building the
     *                       REMOVE hole — much tighter boundaries (hair,
     *                       jaw, fingertips) so the stitcher samples cleaner
     *                       background. When omitted or the refiner is
     *                       unavailable, falls back to the binary YOLO mask
     *                       + bbox-floor approximation.
     */
    fun execute(
        persons: List<Person>,
        states: Map<Int, PersonState>,
        sourceWidth: Int,
        sourceHeight: Int,
        referenceRgba: ByteArray? = null
    ): Pair<PipelineResult?, String> {
        val removePersons = persons.filter { (states[it.trackId] ?: PersonState.KEEP) == PersonState.REMOVE }
        if (removePersons.isEmpty()) {
            return null to "tap people to mark them for removal"
        }

        val w = sourceWidth
        val h = sourceHeight

        // Build REMOVE mask. Two paths:
        //   1. MattingRefiner available + reference RGBA present → refine
        //      each person's binary YOLO mask via MODNet. The refined alpha
        //      is precise enough that the bbox-floor backup (which fills
        //      the full bbox rect for large persons to compensate for YOLO
        //      mask leak at hair/feet) is no longer needed.
        //   2. Fallback path: binary YOLO mask OR'd across REMOVE persons,
        //      plus bbox-floor for bbox area > 25% of frame.
        val refiner = mattingRefiner
        val canMatte = refiner != null && refiner.isAvailable &&
                referenceRgba != null && referenceRgba.size >= w * h * 4
        val removeMask = if (canMatte) {
            val rgb = rgbaToRgb(referenceRgba!!, w, h)
            val alpha = refiner!!.refineRemoveMask(rgb, w, h, removePersons)
            // Threshold at 0.5 alpha so only confidently-foreground pixels
            // become the REMOVE hole. Sub-threshold alphas live in YOLO's
            // dilation slack on the JNI side. Downstream native code
            // binarizes anything non-zero, so we map [128..255] → 1.
            val out = ByteArray(w * h)
            for (i in 0 until w * h) {
                if ((alpha[i].toInt() and 0xFF) >= 128) out[i] = 1
            }
            out
        } else {
            buildBinaryRemoveMaskWithBboxFloor(removePersons, w, h)
        }

        val t0 = System.currentTimeMillis()
        val stitch = nativeSession.stitchForInpaint(removeMask, w, h)
            ?: return null to "no stitch (empty buffer or stale detection)"

        // Inpaint routing:
        //   - fill ≥ 0.95          → no inpaint needed.
        //   - fill ≥ 0.85          → OpenCV (Navier-Stokes). Small gaps; LaMa
        //                            adds no quality at this hole size.
        //   - noSampleRatio > 0.60 → LaMa would be pure hallucination (no
        //                            real context band for the model to
        //                            extrapolate from). Use Navier-Stokes
        //                            instead — smoother, less surprising.
        //   - otherwise            → LaMa. Big-enough holes with enough
        //                            context band that LaMa beats classical
        //                            inpaint on plausibility.
        //
        // The 0.85 threshold + LaMa FP16 brings worst-case inpaint into the
        // ~10-15s range (FP32 was 22-32s). Re-enabled now that FP16 path
        // exists.
        val noInpaintThreshold = 0.95f
        val classicalThreshold = 0.85f
        val needsInpaint = stitch.fillRatio < noInpaintThreshold
        val lamaHasNoContext = stitch.noSampleRatio > 0.60f
        val useLama = stitch.fillRatio < classicalThreshold && !lamaHasNoContext

        val filledRgb: ByteArray
        val inpaintMethod: String
        if (!needsInpaint) {
            filledRgb = stitch.rgb
            inpaintMethod = "none"
        } else if (!useLama) {
            val cv = nativeSession.opencvInpaint(stitch.rgb, stitch.unfilledMask, stitch.width, stitch.height)
            filledRgb = cv ?: stitch.rgb
            inpaintMethod = when {
                cv == null -> "none-failed"
                lamaHasNoContext -> "opencv-ns-noctx"
                else -> "opencv-ns-fast"
            }
        } else {
            var ok: ByteArray? = null
            var method = "opencv"
            lamaInpainter?.let { lama ->
                try {
                    ok = lama.inpaintGaps(stitch.rgb, stitch.unfilledMask,
                                          stitch.width, stitch.height)
                    method = "lama"
                } catch (t: Throwable) {
                    Log.w(TAG, "LaMa inpaint threw, falling back", t)
                }
            }
            if (ok == null) {
                ok = nativeSession.opencvInpaint(stitch.rgb, stitch.unfilledMask,
                                                 stitch.width, stitch.height)
            }
            filledRgb = ok ?: stitch.rgb
            inpaintMethod = if (ok == null) "none-failed" else method
        }

        val polished = nativeSession.finalize(
            stitch.rgb, filledRgb, stitch.unfilledMask,
            stitch.width, stitch.height
        ) ?: filledRgb

        val totalMs = System.currentTimeMillis() - t0
        Log.i(TAG,
            "pipeline done method=$inpaintMethod fill=%.2f noSample=%.2f total=${totalMs}ms"
                .format(stitch.fillRatio, stitch.noSampleRatio))

        val baseStatus = "fill=%.2f via %s  %dms".format(
            stitch.fillRatio, inpaintMethod, totalMs)
        // noSampleRatio > 0.3 → at least 30% of the hole has zero clean-bg
        // samples across the entire buffer window. No amount of waiting will
        // recover those pixels because the subject occluded them every frame;
        // the user must move the subject (or shift the camera) before retake.
        // Fall back to the older fill-based hint for borderline captures.
        val status = when {
            stitch.noSampleRatio > 0.30f ->
                "$baseStatus  — large area never visible; ask subject to step aside briefly and retake"
            stitch.fillRatio < 0.50f ->
                "$baseStatus  — tricky scene, try stepping out of frame for a few seconds and retake"
            else -> baseStatus
        }
        return PipelineResult(
            rgb = polished,
            width = stitch.width,
            height = stitch.height,
            holeMask = stitch.fullHoleMask
        ) to status
    }

    private fun buildBinaryRemoveMaskWithBboxFloor(
        removePersons: List<Person>,
        w: Int,
        h: Int
    ): ByteArray {
        val bboxFloorAreaThreshold = (w.toLong() * h * 0.25).toInt()
        val removeMask = ByteArray(w * h)
        for (p in removePersons) {
            if (p.maskWidth != w || p.maskHeight != h || p.mask.size < w * h) continue

            for (i in 0 until w * h) {
                if (p.mask[i].toInt() != 0) removeMask[i] = 1
            }

            val xMin = p.bBox.left.toInt().coerceAtLeast(0)
            val yMin = p.bBox.top.toInt().coerceAtLeast(0)
            val xMax = p.bBox.right.toInt().coerceAtMost(w - 1)
            val yMax = p.bBox.bottom.toInt().coerceAtMost(h - 1)
            val bboxArea = (xMax - xMin + 1) * (yMax - yMin + 1)
            if (bboxArea > bboxFloorAreaThreshold) {
                for (y in yMin..yMax) {
                    val rowBase = y * w
                    for (x in xMin..xMax) {
                        removeMask[rowBase + x] = 1
                    }
                }
            }
        }
        return removeMask
    }

    private fun rgbaToRgb(rgba: ByteArray, w: Int, h: Int): ByteArray {
        val n = w * h
        val rgb = ByteArray(n * 3)
        for (i in 0 until n) {
            rgb[i * 3]     = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return rgb
    }

    private companion object {
        const val TAG = "PRPipeline"
    }
}