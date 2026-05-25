package com.one5.personremoval.core

import android.graphics.RectF

enum class PersonState { KEEP, REMOVE }

/**
 * A detected person in a single frame.
 *
 * @param trackId stable id across frames (assigned by Tracker; -1 = unassigned)
 * @param bBox    pixel-space bounding box in the SOURCE image coordinates
 * @param mask    binary mask, width*height bytes (0 or 1), aligned with the source image.
 *                Held as a CPU ByteArray for now; will move to native heap in Phase 6.
 * @param maskWidth / maskHeight match the source frame size
 * @param confidence YOLO detection confidence (0..1)
 */
data class Person(
    val trackId: Int,
    val bBox: RectF,
    val mask: ByteArray,
    val maskWidth: Int,
    val maskHeight: Int,
    val confidence: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Person) return false
        return trackId == other.trackId &&
                bBox == other.bBox &&
                confidence == other.confidence
    }

    override fun hashCode(): Int {
        var result = trackId
        result = 31 * result + bBox.hashCode()
        result = 31 * result + confidence.hashCode()
        return result
    }
}

/**
 * Output of a single YOLO inference, before tracking.
 */
data class DetectionResult(
    val persons: List<Person>,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inferenceMs: Long
)
