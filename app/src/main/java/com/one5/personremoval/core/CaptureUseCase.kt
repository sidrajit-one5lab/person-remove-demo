package com.one5.personremoval.core

import android.util.Log
import com.one5.personremoval.ml.LamaInpainter

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
) {

    fun execute(
        persons: List<Person>,
        states: Map<Int, PersonState>,
        sourceWidth: Int,
        sourceHeight: Int
    ): Pair<PipelineResult?, String> {
        val removePersons = persons.filter { (states[it.trackId] ?: PersonState.KEEP) == PersonState.REMOVE }
        if (removePersons.isEmpty()) {
            return null to "tap people to mark them for removal"
        }

        val w = sourceWidth
        val h = sourceHeight

        // Binary YOLO mask OR'd across REMOVE persons, plus bbox-floor for
        // large persons (>40% of frame) to compensate for YOLO mask leak at
        // hair/feet boundaries.
        val removeMask = buildBinaryRemoveMaskWithBboxFloor(removePersons, w, h)

        val removeTrackIds = removePersons.map { it.trackId }.toIntArray()

        val t0 = System.currentTimeMillis()
        val stitch = nativeSession.stitchForInpaint(removeMask, w, h, removeTrackIds)
            ?: return null to "keep phone steady — hold still and ask the subject to step aside"

        // Recompute fill ratio using actual unfilled mask (ghost detector may
        // have flagged additional pixels after native fillRatio was computed).
        val actualUnfilled = stitch.unfilledMask.count { it.toInt() != 0 }
        val holePx = stitch.fullHoleMask.count { it.toInt() != 0 }.coerceAtLeast(1)
        val actualFill = 1f - actualUnfilled.toFloat() / holePx

        val classicalThreshold = 0.90f
        val needsInpaint = actualUnfilled > 0
        val useLama = actualFill < classicalThreshold

        val filledRgb: ByteArray
        val inpaintMethod: String
        if (!needsInpaint) {
            filledRgb = stitch.rgb
            inpaintMethod = "none"
        } else if (!useLama) {
            val cv = nativeSession.opencvInpaint(stitch.rgb, stitch.unfilledMask, stitch.width, stitch.height)
            filledRgb = cv ?: stitch.rgb
            inpaintMethod = if (cv == null) "none-failed" else "opencv-ns-fast"
        } else {
            var ok: ByteArray? = null
            var method = "lama"
            lamaInpainter?.let { lama ->
                try {
                    ok = lama.inpaintGaps(stitch.rgb, stitch.unfilledMask,
                                          stitch.width, stitch.height)
                } catch (t: Throwable) {
                    Log.w(TAG, "LaMa inpaint threw, falling back", t)
                    method = "opencv"
                }
            } ?: run { method = "opencv" }
            if (ok == null) {
                ok = nativeSession.opencvInpaint(stitch.rgb, stitch.unfilledMask,
                                                 stitch.width, stitch.height)
                method = "opencv"
            }
            filledRgb = ok ?: stitch.rgb
            inpaintMethod = if (ok == null) "none-failed" else method
        }

        if (useLama) {
            nativeSession.harmonizeLamaRegion(
                filledRgb, stitch.unfilledMask, stitch.width, stitch.height)
        }

        val polished = nativeSession.finalize(
            stitch.rgb, filledRgb, stitch.unfilledMask,
            stitch.width, stitch.height
        ) ?: filledRgb

        // Grain match the inpainted region as the final step. The LaMa/Telea
        // fill is low-frequency and reads as a too-smooth patch against the
        // grainy surround; injecting matched sensor noise lets it blend.
        // Done AFTER finalize so the Poisson clone doesn't smooth it back out,
        // and only on the inpaint path. Self-skips on clean/thin surrounds, so
        // it never degrades a good capture.
        if (useLama) {
            nativeSession.textureLamaRegion(
                polished, stitch.unfilledMask, stitch.width, stitch.height)
        }

        val totalMs = System.currentTimeMillis() - t0
        // Report `actualFill` (recomputed above) instead of the raw native
        // `stitch.fillRatio`. The native ratio is measured before the ghost
        // detector flags additional unfilled pixels, so it overstates how
        // much of the hole was filled by real samples. Routing decisions
        // already use `actualFill`; logs, toast, and hint buckets must use
        // the same number or the user sees one value while the pipeline
        // routed off a different one.
        Log.i(TAG,
            "pipeline done method=$inpaintMethod fill=%.2f (native=%.2f) noSample=%.2f total=${totalMs}ms"
                .format(actualFill, stitch.fillRatio, stitch.noSampleRatio))

        val baseStatus = "fill=%.2f via %s  %dms".format(
            actualFill, inpaintMethod, totalMs)
        // noSampleRatio > 0.3 → at least 30% of the hole has zero clean-bg
        // samples across the entire buffer window. No amount of waiting will
        // recover those pixels because the subject occluded them every frame;
        // the user must move the subject (or shift the camera) before retake.
        // Fall back to the older fill-based hint for borderline captures.
        val status = when {
            actualFill < 0.25f ->
                "$baseStatus  — keep phone steady and ask subject to step aside"
            stitch.noSampleRatio >= 0.30f ->
                "$baseStatus  — large area never visible; ask subject to step aside briefly and retake"
            actualFill < 0.50f ->
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
        val bboxFloorAreaThreshold = (w.toLong() * h * 0.40).toInt()
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

    private companion object {
        const val TAG = "PRPipeline"
    }
}