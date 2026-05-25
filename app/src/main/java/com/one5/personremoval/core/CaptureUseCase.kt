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
    @Volatile var lamaInpainter: LamaInpainter? = null
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

        val t0 = System.currentTimeMillis()
        val stitch = nativeSession.stitchForInpaint(removeMask, w, h)
            ?: return null to "no stitch (empty buffer or stale detection)"

        // LaMa for fill < 0.85 (larger unfilled gaps where Telea produces visible blur).
        // With FP16 model (~10-15s) this is tolerable; with FP32 (~22-32s) the user
        // waits longer but gets better output than Telea on these hard cases.
        val lamaThreshold = 0.85f
        val needsInpaint = stitch.fillRatio < 1.0f
        val useLama = stitch.fillRatio < lamaThreshold

        val filledRgb: ByteArray
        val inpaintMethod: String
        if (!needsInpaint) {
            filledRgb = stitch.rgb
            inpaintMethod = "none"
        } else if (!useLama) {
            val cv = nativeSession.opencvInpaint(stitch.rgb, stitch.unfilledMask, stitch.width, stitch.height)
            filledRgb = cv ?: stitch.rgb
            inpaintMethod = if (cv == null) "none-failed" else "opencv-fast"
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
            "pipeline done method=$inpaintMethod fill=%.2f total=${totalMs}ms"
                .format(stitch.fillRatio))

        val baseStatus = "fill=%.2f via %s  %dms".format(
            stitch.fillRatio, inpaintMethod, totalMs)
        val status = if (stitch.fillRatio < 0.50f) {
            "$baseStatus  — tricky scene, try stepping out of frame for a few seconds and retake"
        } else {
            baseStatus
        }
        return PipelineResult(
            rgb = polished,
            width = stitch.width,
            height = stitch.height,
            holeMask = stitch.fullHoleMask
        ) to status
    }

    private companion object {
        const val TAG = "PRPipeline"
    }
}