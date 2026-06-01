package com.one5.personremoval.core

/**
 * Measures how "busy" the background immediately surrounding a hole is, so the
 * capture pipeline can decide between two honest outcomes when little real
 * background was recovered (low fill):
 *
 *   - LOW information surround (flat wall, sky, uniform floor, repeating
 *     pattern): a generated/extended fill is indistinguishable from the truth,
 *     because there is no unique detail to get wrong. Fill it and ship it.
 *
 *   - HIGH information surround (text, screens, edges, furniture, faces): any
 *     fill must invent specific content that is guaranteed to be wrong. No
 *     method recovers pixels the camera never captured, so the honest product
 *     behaviour is to ask the user to step aside / pan and retake rather than
 *     ship a convincing-but-false patch.
 *
 * The signal is edge density in a thin band just OUTSIDE the hole. Edges are
 * cheap, robust, and exactly what distinguishes "plain surface we can extend"
 * from "unique structure we can't". The band excludes pixels adjacent to the
 * hole so the hole's own boundary doesn't masquerade as scene detail.
 *
 * Pure CPU on the stitched RGB buffer — no JNI, no model, fully unit-testable.
 * Runs once per capture over a bounded region, so cost is a few ms even on a
 * frame-filling hole.
 */
object BackgroundComplexity {

    /**
     * @param edgeDensity  fraction of measured surround pixels that sit on an
     *                     edge (0 = perfectly flat, 1 = every pixel an edge).
     * @param sampleCount  number of surround pixels actually measured. Low
     *                     counts mean the hole leaves too little surround to
     *                     judge (it fills nearly the whole frame) — treated as
     *                     NOT low-information so we fall back to the honest
     *                     retake path rather than trusting a guess.
     * @param isLowInformation  true when the surround is flat/uniform enough
     *                          that an extended fill will read as real.
     */
    data class Complexity(
        val edgeDensity: Float,
        val sampleCount: Int,
        val isLowInformation: Boolean
    )

    /**
     * @param rgb       stitched RGB, [width*height*3] bytes, row-major.
     * @param holeMask  [width*height] bytes; non-zero marks the hole.
     */
    fun analyze(rgb: ByteArray, holeMask: ByteArray, width: Int, height: Int): Complexity {
        if (width <= 2 || height <= 2 ||
            holeMask.size < width * height ||
            rgb.size < width * height * 3
        ) {
            return Complexity(0f, 0, isLowInformation = false)
        }

        // Hole bounding box.
        var minX = width; var minY = height; var maxX = -1; var maxY = -1
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                if (holeMask[row + x].toInt() != 0) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        if (maxX < 0) return Complexity(0f, 0, isLowInformation = false)

        // Band width scales with frame size; clamped so tiny/huge frames behave.
        val band = (minOf(width, height) / 20).coerceIn(MIN_BAND_PX, MAX_BAND_PX)

        // Region of interest = hole bbox grown by the band, clamped one pixel
        // in from the frame edge so the 3-tap gradient stays in bounds.
        val x0 = (minX - band).coerceAtLeast(1)
        val y0 = (minY - band).coerceAtLeast(1)
        val x1 = (maxX + band).coerceAtMost(width - 2)
        val y1 = (maxY + band).coerceAtMost(height - 2)

        var samples = 0
        var edges = 0
        for (y in y0..y1) {
            val row = y * width
            val rowUp = row - width
            val rowDown = row + width
            for (x in x0..x1) {
                val c = row + x
                // Only the surround: skip hole pixels...
                if (holeMask[c].toInt() != 0) continue
                // ...and any pixel touching the hole, so the hole edge itself
                // never counts as scene detail.
                if (holeMask[c - 1].toInt() != 0 || holeMask[c + 1].toInt() != 0 ||
                    holeMask[rowUp + x].toInt() != 0 || holeMask[rowDown + x].toInt() != 0
                ) continue

                val lLeft = lum(rgb, c - 1)
                val lRight = lum(rgb, c + 1)
                val lUp = lum(rgb, rowUp + x)
                val lDown = lum(rgb, rowDown + x)
                // L1 gradient magnitude (|gx| + |gy|) — cheaper than hypot and
                // adequate for a density statistic.
                val mag = kotlin.math.abs(lRight - lLeft) + kotlin.math.abs(lDown - lUp)
                samples++
                if (mag > EDGE_MAG_THRESHOLD) edges++
            }
        }

        val density = if (samples > 0) edges.toFloat() / samples else 0f
        val lowInfo = samples >= MIN_SAMPLES && density < EDGE_DENSITY_LOW_THRESHOLD
        return Complexity(density, samples, lowInfo)
    }

    /** Fast integer luminance (Rec.601 approx) at byte index pixel `idx`. */
    private fun lum(rgb: ByteArray, idx: Int): Int {
        val o = idx * 3
        val r = rgb[o].toInt() and 0xFF
        val g = rgb[o + 1].toInt() and 0xFF
        val b = rgb[o + 2].toInt() and 0xFF
        return (r * 77 + g * 150 + b * 29) shr 8
    }

    // --- Tunables. Starting points; calibrate by logging Complexity on real
    // captures and setting the cutoff where plain-wall scenes sit versus
    // whiteboard/screen/furniture scenes (see CaptureUseCase low-fill log). ---

    // Neighbour-to-neighbour luminance step (summed over x and y) above which a
    // surround pixel is "on an edge". ~40/510 ≈ a modest local contrast jump.
    private const val EDGE_MAG_THRESHOLD = 40

    // Surround edge fraction below which we treat the background as extendable.
    private const val EDGE_DENSITY_LOW_THRESHOLD = 0.10f

    // Minimum surround pixels before the density is trustworthy.
    private const val MIN_SAMPLES = 500

    private const val MIN_BAND_PX = 16
    private const val MAX_BAND_PX = 64
}