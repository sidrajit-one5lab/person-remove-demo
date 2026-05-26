package com.one5.personremoval.core

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/**
 * Assigns stable IDs to detected persons across frames using greedy IoU matching.
 * Persists KEEP/REMOVE state per track ID so the user's tap survives momentary mask jitter.
 */
class Tracker(
    private val iouThreshold: Float = 0.2f,
    private val maxGapMs: Long = 1000L
) {

    private data class Track(
        val id: Int,
        var bbox: RectF,
        var lastSeenMs: Long,
        var state: PersonState
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    /**
     * Match detections to existing tracks, assign IDs.
     * Returns tracked persons and a snapshot of KEEP/REMOVE states for all active tracks.
     *
     * Synchronized: [update] runs on the analyzer coroutine (Dispatchers.Default)
     * while [toggle] runs on Main from tap input. Without the monitor the shared
     * `tracks` list would CME or lose a tap that races with the update loop.
     */
    @Synchronized
    fun update(detections: List<Person>, nowMs: Long): Pair<List<Person>, Map<Int, PersonState>> {
        val originalSize = tracks.size
        val used = BooleanArray(originalSize)
        val out = ArrayList<Person>(detections.size)

        for (det in detections) {
            var bestIdx = -1
            var bestIoU = iouThreshold
            for (i in 0 until originalSize) {
                if (used[i]) continue
                val v = iou(det.bBox, tracks[i].bbox)
                if (v > bestIoU) { bestIoU = v; bestIdx = i }
            }
            // Fallback: center-distance match for walking persons whose
            // IoU dropped below threshold due to large frame-to-frame shift.
            if (bestIdx < 0) {
                val cx = (det.bBox.left + det.bBox.right) / 2f
                val cy = (det.bBox.top + det.bBox.bottom) / 2f
                val detW = det.bBox.right - det.bBox.left
                val detH = det.bBox.bottom - det.bBox.top
                val maxDist = max(detW, detH) * 1.5f
                var bestDist = maxDist
                for (i in 0 until originalSize) {
                    if (used[i]) continue
                    val t = tracks[i]
                    val tcx = (t.bbox.left + t.bbox.right) / 2f
                    val tcy = (t.bbox.top + t.bbox.bottom) / 2f
                    val dist = kotlin.math.sqrt((cx - tcx) * (cx - tcx) + (cy - tcy) * (cy - tcy))
                    if (dist < bestDist) { bestDist = dist; bestIdx = i }
                }
            }
            if (bestIdx >= 0) {
                val t = tracks[bestIdx]
                used[bestIdx] = true
                t.bbox = det.bBox
                t.lastSeenMs = nowMs
                out.add(det.copy(trackId = t.id))
            } else {
                val t = Track(nextId++, det.bBox, nowMs, PersonState.KEEP)
                tracks.add(t)
                out.add(det.copy(trackId = t.id))
            }
        }

        tracks.removeAll { nowMs - it.lastSeenMs > maxGapMs }

        val states = tracks.associate { it.id to it.state }
        return out to states
    }

    @Synchronized
    fun getStates(): Map<Int, PersonState> = tracks.associate { it.id to it.state }

    /** Toggle KEEP↔REMOVE for a given track id (no-op if not found). */
    @Synchronized
    fun toggle(trackId: Int) {
        val t = tracks.find { it.id == trackId } ?: return
        t.state = if (t.state == PersonState.KEEP) PersonState.REMOVE else PersonState.KEEP
    }

    /** Reset all tracks (e.g. on screen restart). */
    @Synchronized
    fun reset() {
        tracks.clear()
        nextId = 1
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val iy = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val inter = ix * iy
        val ua = (a.right - a.left) * (a.bottom - a.top)
        val ub = (b.right - b.left) * (b.bottom - b.top)
        val union = ua + ub - inter
        return if (union <= 0f) 0f else inter / union
    }
}
