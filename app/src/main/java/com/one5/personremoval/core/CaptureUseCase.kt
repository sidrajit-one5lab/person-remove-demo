package com.one5.personremoval.core

import android.util.Log
import com.one5.personremoval.ml.MiGanInpainter

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
    @Volatile var miganInpainter: MiGanInpainter? = null,
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
        // Current KEEP track ids. The stitcher admits ONLY these persons as
        // background; everyone else — the REMOVE target and any stale-track
        // instance of a person who left and returned with a new id — is
        // excluded from sampling, so the departed subject can't leak back in
        // as a ghost. Persons with no explicit state default to KEEP.
        val keepTrackIds = persons
            .filter { (states[it.trackId] ?: PersonState.KEEP) == PersonState.KEEP }
            .map { it.trackId }
            .toIntArray()

        val t0 = System.currentTimeMillis()
        val stitch = nativeSession.stitchForInpaint(removeMask, w, h, removeTrackIds, keepTrackIds)
            ?: return null to "keep phone steady — hold still and ask the subject to step aside"

        // Recompute fill ratio using actual unfilled mask (ghost detector may
        // have flagged additional pixels after native fillRatio was computed).
        val actualUnfilled = stitch.unfilledMask.count { it.toInt() != 0 }
        val holePx = stitch.fullHoleMask.count { it.toInt() != 0 }.coerceAtLeast(1)
        val actualFill = 1f - actualUnfilled.toFloat() / holePx

        // Ghost detector — LOW-PRIORITY ADVISORY backstop. The real defense is
        // upstream (keepIds exclusion, mask dilation, alignment); this only
        // mops up a ghost those miss. It is deliberately NON-BLOCKING: we never
        // refuse to save on its verdict, because a false positive would cost a
        // good capture. It just flags a retake hint (see status below). Cheap
        // to run; always logged so we can calibrate without it interfering.
        val ghost = GhostDetector.detect(
            stitch.rgb, stitch.fullHoleMask, stitch.width, stitch.height)
        Log.i(TAG, "ghost check holeEdge=%.3f surroundEdge=%.3f isGhost=%b fill=%.2f"
            .format(ghost.holeEdgeDensity, ghost.surroundEdgeDensity, ghost.isGhost, actualFill))

        // Fill engine and user-messaging are decoupled:
        //  - ENGINE: MI-GAN fills every hole in one fast forward pass
        //    (purpose-built remover); classical OpenCV (Telea/NS) is the
        //    fallback only if MI-GAN is unavailable or throws.
        //  - MESSAGING: the fill ratio still shapes the retake hint.
        //    actualFill < LOW_FILL_THRESHOLD = the subject occluded almost the
        //    whole hole (little real background); combined with surround
        //    complexity it decides "fill quietly" vs "ask for a retake".
        val needsInpaint = actualUnfilled > 0
        val lowFill = actualFill < LOW_FILL_THRESHOLD

        // On a low-fill capture the hole is mostly invention regardless of
        // engine, so the question stops being "which inpainter" and becomes
        // "can this background honestly be faked?". A flat/uniform surround
        // (low edge density) extends convincingly — fill it quietly. A busy
        // surround (text, screens, furniture) cannot be reconstructed — prompt
        // a retake instead of shipping a convincing-but-wrong patch. Measured
        // on the real pixels *around* the hole, so the partial fill inside it
        // is irrelevant. Only computed on the low-fill path (cheap to skip).
        val lowFillComplexity = if (lowFill) {
            BackgroundComplexity.analyze(
                stitch.rgb, stitch.fullHoleMask, stitch.width, stitch.height)
        } else null
        val extendableLowFill = lowFillComplexity?.isLowInformation == true

        // Dilate the fill mask by a small margin before inpainting so the
        // inpainter repaints a thin band at the hole boundary instead of
        // trusting the exact stitch seam, where sub-pixel softening can linger.
        // Kept small (INPAINT_MASK_DILATE_PX) so we don't discard good real bg —
        // the keepIds exclusion already removed the main edge-contamination
        // source, so this is a light seam guard, not a heavy repaint. The SAME
        // dilated mask drives inpaint + finalize + grain so all three cover one
        // region; the undilated unfilledMask still drives the fill-ratio, ghost
        // and complexity measurements above.
        val inpaintMask = if (needsInpaint) {
            NativeSession.nativeDilateMask(
                stitch.unfilledMask, stitch.width, stitch.height, INPAINT_MASK_DILATE_PX)
        } else {
            stitch.unfilledMask
        }

        val filledRgb: ByteArray
        val inpaintMethod: String
        if (!needsInpaint) {
            filledRgb = stitch.rgb
            inpaintMethod = "none"
        } else {
            // MI-GAN fills every hole: its pipeline takes the FULL frame +
            // (dilated) hole mask and crops/resizes/inpaints/pastes/blends in
            // one fast forward pass. Falls back to classical OpenCV (Telea/NS)
            // only if MI-GAN is unavailable or throws.
            var ok: ByteArray? = null
            var method = "opencv"
            miganInpainter?.let { mig ->
                try {
                    ok = mig.inpaint(stitch.rgb, inpaintMask, stitch.width, stitch.height)
                    method = "migan"
                } catch (t: Throwable) {
                    Log.w(TAG, "MI-GAN threw, falling back to OpenCV", t)
                }
            }
            if (ok == null) {
                ok = nativeSession.opencvInpaint(stitch.rgb, inpaintMask,
                                                 stitch.width, stitch.height)
                method = when {
                    ok == null -> "none-failed"
                    lowFill -> "opencv-ns-lowfill"
                    else -> "opencv-ns-fast"
                }
            }
            filledRgb = ok ?: stitch.rgb
            inpaintMethod = method
        }

        // Color reconciliation between the fill and the surrounding photo is
        // handled solely by the Poisson seamless clone in finalize(). The old
        // harmonizeLamaRegion mean-shift pass is intentionally disabled: it
        // sampled a thin ring just outside the fill that can still carry the
        // subject's tint, then dragged the whole patch toward it — feeding the
        // person-shaped ghost and fighting the very clone that runs next. One
        // color pass, not two stacked passes that partially undo each other.

        val polished = nativeSession.finalize(
            stitch.rgb, filledRgb, inpaintMask,
            stitch.width, stitch.height
        ) ?: filledRgb

        // Grain match the inpainted region as the final step. Any AI/classical
        // fill is low-frequency and reads as a too-smooth patch against the
        // grainy surround; injecting matched sensor noise lets it blend.
        // Done AFTER finalize so the Poisson clone doesn't smooth it back out,
        // and only when something was actually filled. Self-skips on clean/thin
        // surrounds, so it never degrades a good capture.
        if (needsInpaint) {
            nativeSession.textureLamaRegion(
                polished, inpaintMask, stitch.width, stitch.height)
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
            ("pipeline done method=$inpaintMethod fill=%.2f (native=%.2f) noSample=%.2f " +
                "lowFillEdgeDensity=%.3f extendable=%b total=${totalMs}ms")
                .format(actualFill, stitch.fillRatio, stitch.noSampleRatio,
                    lowFillComplexity?.edgeDensity ?: -1f, extendableLowFill))

        val baseStatus = "fill=%.2f via %s  %dms".format(
            actualFill, inpaintMethod, totalMs)
        // noSampleRatio > 0.3 → at least 30% of the hole has zero clean-bg
        // samples across the entire buffer window. No amount of waiting will
        // recover those pixels because the subject occluded them every frame;
        // the user must move the subject (or shift the camera) before retake.
        // Fall back to the older fill-based hint for borderline captures.
        val status = when {
            // Advisory only — image is still saved; just suggest a retake.
            ghost.isGhost ->
                "$baseStatus  — subject may still be faintly visible; pan a few cm across them and retake for a cleaner result"
            // Low fill + busy surround: an honest fill is impossible. Retake.
            lowFill && !extendableLowFill ->
                "$baseStatus  — little background behind the subject and the area is detailed; ask them to step aside (or pan slightly) and retake"
            // Low fill + flat surround: the extended fill reads as real. Ship
            // it quietly, just nudge toward a slight pan for an even better one.
            lowFill ->
                "$baseStatus  — filled a plain background; pan slightly next time for a sharper result"
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

        // Below this real-pixel fill ratio the hole is mostly invention; the
        // surround-complexity check then decides "fill quietly" vs "retake".
        // Tunable: raise to gate more aggressively toward retake.
        const val LOW_FILL_THRESHOLD = 0.35f

        // Margin (px) the inpaint fill mask is dilated by before the inpainter,
        // so it repaints a thin band at the stitch seam rather than
        // conditioning on it. Small on purpose — larger discards good real
        // background. Raise only if a seam halo persists.
        const val INPAINT_MASK_DILATE_PX = 6
    }
}