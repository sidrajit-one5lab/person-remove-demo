package com.one5.personremoval.ui

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Live capture coach that turns the app's core trick into a user instruction.
 *
 * The buffered multi-frame stitch can only recover the background BEHIND a
 * person from frames where that background was actually visible. That happens
 * when there is parallax: the person shifts, or — more reliably — the camera
 * moves a few centimetres, so a near subject slides against a far background
 * and reveals what it was hiding. A perfectly still subject in front of a
 * still camera reveals nothing, which is the one case the stitch can't win.
 *
 * So instead of the old "keep phone steady" instruction (which actively
 * defeats the stitch), this coach reads the gyro and the tracked subject and
 * nudges the user to sweep a small arc until enough background has been
 * exposed, then tells them they're ready. For a subject so large that panning
 * can't clear it, it asks them to step aside — the honest fallback.
 *
 * The coach is advisory only: it never blocks the shutter. [onFrame] is fed
 * from the analyzer loop; [state] drives the overlay.
 *
 * Threading: [onFrame] and [reset] mutate the sweep history under [lock].
 * onFrame runs on the analyzer dispatcher; reset may be called from a camera
 * callback thread (exposure relock) or the main thread (flip).
 */
class ParallaxCoach {

    enum class Phase { HIDDEN, PAN, ALMOST, READY, STEP_ASIDE }

    data class State(
        val phase: Phase,
        /** Sweep progress 0..1 toward the target arc; for a progress affordance. */
        val progress: Float,
        val message: String
    ) {
        val visible: Boolean get() = phase != Phase.HIDDEN
    }

    private val _state = MutableStateFlow(HIDDEN_STATE)
    val state: StateFlow<State> = _state.asStateFlow()

    private val lock = Any()
    // Sweep history: (timeMs, cumulativeDegrees). cumulative is monotonic; the
    // sweep over the recent window is cumulativeNow - cumulativeAtWindowStart.
    private val history = ArrayDeque<Pair<Long, Float>>()
    private var cumulativeDeg = 0f
    private var prevRotation: FloatArray? = null

    /**
     * @param rotation        gyro rotation snapshot (row-major 3x3), or null if
     *                        the device has no gyro / no sample yet.
     * @param hasRemoveTarget at least one person is marked for removal.
     * @param subjectStatic   a removal target is holding still (no parallax
     *                        coming from subject motion).
     * @param largestAreaFrac largest removal target's bbox area as a fraction
     *                        of the frame (0..1).
     */
    fun onFrame(
        rotation: FloatArray?,
        hasRemoveTarget: Boolean,
        subjectStatic: Boolean,
        largestAreaFrac: Float,
        nowMs: Long
    ) {
        val sweepDeg = synchronized(lock) {
            accumulateSweep(rotation, nowMs)
            sweepInWindow(nowMs)
        }
        _state.value = decide(hasRemoveTarget, subjectStatic, largestAreaFrac, sweepDeg)
    }

    /** Clear accumulated sweep. Call whenever the ring buffer is flushed
     *  (exposure relock, camera flip) so coaching restarts with the buffer. */
    fun reset() {
        synchronized(lock) {
            history.clear()
            cumulativeDeg = 0f
            prevRotation = null
        }
        _state.value = HIDDEN_STATE
    }

    // --- internals (all sweep state touched under lock) ---

    private fun accumulateSweep(rotation: FloatArray?, nowMs: Long) {
        if (rotation != null) {
            val prev = prevRotation
            if (prev != null) {
                val step = angularDeg(prev, rotation)
                // Deadband rejects gyro noise; ceiling rejects glitch jumps.
                if (step in NOISE_DEG..MAX_STEP_DEG) cumulativeDeg += step
            }
            prevRotation = rotation.copyOf()
        }
        history.addLast(nowMs to cumulativeDeg)
        while (history.size > 1 && nowMs - history.first().first > WINDOW_MS) {
            history.removeFirst()
        }
    }

    private fun sweepInWindow(nowMs: Long): Float {
        val first = history.firstOrNull() ?: return 0f
        // Only credit sweep observed within the window.
        if (nowMs - first.first > WINDOW_MS) return 0f
        return cumulativeDeg - first.second
    }

    private fun decide(
        hasRemoveTarget: Boolean,
        subjectStatic: Boolean,
        largestAreaFrac: Float,
        sweepDeg: Float
    ): State {
        if (!hasRemoveTarget) return HIDDEN_STATE

        val progress = (sweepDeg / TARGET_SWEEP_DEG).coerceIn(0f, 1f)

        // A moving subject reveals its own background — already winning.
        if (!subjectStatic) {
            return State(Phase.READY, 1f, MSG_READY)
        }
        // Enough camera sweep banked → ready regardless of subject stillness.
        if (sweepDeg >= TARGET_SWEEP_DEG) {
            return State(Phase.READY, 1f, MSG_READY)
        }
        // Static subject filling the frame: no small pan clears it. Be honest.
        if (largestAreaFrac >= BIG_SUBJECT_FRAC) {
            return State(Phase.STEP_ASIDE, progress, MSG_STEP_ASIDE)
        }
        if (sweepDeg >= TARGET_SWEEP_DEG * ALMOST_FRAC) {
            return State(Phase.ALMOST, progress, MSG_ALMOST)
        }
        return State(Phase.PAN, progress, MSG_PAN)
    }

    /** Geodesic angle (degrees) between two row-major 3x3 rotation matrices. */
    private fun angularDeg(r1: FloatArray, r2: FloatArray): Float {
        // trace(R2 * R1^T) = Frobenius inner product of R1 and R2.
        var trace = 0f
        for (i in 0 until 9) trace += r1[i] * r2[i]
        val cos = ((trace - 1f) / 2f).coerceIn(-1f, 1f)
        return Math.toDegrees(kotlin.math.acos(cos.toDouble())).toFloat()
    }

    private companion object {
        val HIDDEN_STATE = State(Phase.HIDDEN, 0f, "")

        // Cumulative camera arc (deg) over WINDOW_MS that exposes enough
        // background behind a typical standing subject. Rotation is a proxy for
        // the hand translation that actually creates parallax — fine for
        // handheld pans, which always translate. Tune against real captures.
        const val TARGET_SWEEP_DEG = 8f
        const val ALMOST_FRAC = 0.6f
        const val WINDOW_MS = 2500L

        // Per-frame gyro deadband / glitch ceiling (deg).
        const val NOISE_DEG = 0.12f
        const val MAX_STEP_DEG = 25f

        // Subject bbox fraction above which panning can't clear it → step aside.
        const val BIG_SUBJECT_FRAC = 0.45f

        const val MSG_PAN = "Pan slowly across the subject to reveal the background"
        const val MSG_ALMOST = "Almost there — keep moving slowly"
        const val MSG_READY = "Ready — hold steady and capture"
        const val MSG_STEP_ASIDE = "Ask the subject to step aside briefly, then retake"
    }
}