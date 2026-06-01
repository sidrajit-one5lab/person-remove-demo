package com.one5.personremoval.core

/**
 * Catches the "false success" ghost: the stitcher reports a full hole
 * (fill≈1.0, plenty of samples) yet the filled region still shows the
 * subject — a translucent face / smeared body — because the temporal-median
 * samples were contaminated by mask leak or misalignment rather than clean
 * background.
 *
 * The fill RATIO can't see this: it counts pixels that received samples, not
 * whether those samples were correct. This gate is independent of count. It
 * compares the structure INSIDE the hole against the structure of the flat
 * background just OUTSIDE it:
 *
 *   - A correct fill on a plain surround (wall, ceiling) is itself plain —
 *     the real background behind the subject continues the surround, so the
 *     hole is as smooth as its surroundings.
 *   - A ghost on a plain surround is BUSY — a face, hair, shirt seams, and
 *     misalignment shards light up with edges the flat surround doesn't have.
 *
 * So: flat surround + busy hole = the subject survived → retake. The gate
 * deliberately fires ONLY when the surround is genuinely flat, which is also
 * the case where a real fill would have to be flat too — that keeps it from
 * flagging legitimately busy recovered backgrounds (a hallway, foliage) as
 * ghosts. It trades recall (won't catch ghosts on busy backdrops, which are
 * far less visually offensive) for precision (won't block a good capture).
 *
 * Pure CPU, one bounded pass, no model — runs in a few ms per capture.
 */
object GhostDetector {

    /**
     * @param holeEdgeDensity      edge fraction inside the hole interior.
     * @param surroundEdgeDensity  edge fraction in the flat band outside it.
     * @param isGhost              true when the subject is judged to have
     *                             survived the stitch and the capture should
     *                             be retaken rather than shipped.
     */
    data class Verdict(
        val holeEdgeDensity: Float,
        val surroundEdgeDensity: Float,
        val isGhost: Boolean
    )

    /**
     * @param rgb       stitched RGB (the candidate, ghost and all),
     *                  [width*height*3] bytes, row-major.
     * @param holeMask  [width*height] bytes; non-zero marks the hole.
     */
    fun detect(rgb: ByteArray, holeMask: ByteArray, width: Int, height: Int): Verdict {
        if (width <= 2 || height <= 2 ||
            holeMask.size < width * height ||
            rgb.size < width * height * 3
        ) {
            return Verdict(0f, 0f, isGhost = false)
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
        if (maxX < 0) return Verdict(0f, 0f, isGhost = false)

        val band = (minOf(width, height) / 20).coerceIn(MIN_BAND_PX, MAX_BAND_PX)
        val x0 = (minX - band).coerceAtLeast(1)
        val y0 = (minY - band).coerceAtLeast(1)
        val x1 = (maxX + band).coerceAtMost(width - 2)
        val y1 = (maxY + band).coerceAtMost(height - 2)

        var holeSamples = 0; var holeEdges = 0
        var surrSamples = 0; var surrEdges = 0
        for (y in y0..y1) {
            val row = y * width
            val rowUp = row - width
            val rowDown = row + width
            for (x in x0..x1) {
                val c = row + x
                val hereHole = holeMask[c].toInt() != 0
                val leftHole = holeMask[c - 1].toInt() != 0
                val rightHole = holeMask[c + 1].toInt() != 0
                val upHole = holeMask[rowUp + x].toInt() != 0
                val downHole = holeMask[rowDown + x].toInt() != 0

                // Skip the boundary ring entirely: a pixel whose neighbours
                // straddle the hole edge would register the silhouette itself
                // as structure in both buckets. We want internal hole texture
                // vs. external surround texture, never the seam between them.
                val allHole = hereHole && leftHole && rightHole && upHole && downHole
                val allSurround = !hereHole && !leftHole && !rightHole && !upHole && !downHole
                if (!allHole && !allSurround) continue

                val mag = kotlin.math.abs(lum(rgb, c + 1) - lum(rgb, c - 1)) +
                          kotlin.math.abs(lum(rgb, rowDown + x) - lum(rgb, rowUp + x))
                val isEdge = mag > EDGE_MAG_THRESHOLD
                if (allHole) {
                    holeSamples++; if (isEdge) holeEdges++
                } else {
                    surrSamples++; if (isEdge) surrEdges++
                }
            }
        }

        val holeDensity = if (holeSamples > 0) holeEdges.toFloat() / holeSamples else 0f
        val surrDensity = if (surrSamples > 0) surrEdges.toFloat() / surrSamples else 0f

        val isGhost =
            holeSamples >= MIN_SAMPLES &&
            surrSamples >= MIN_SAMPLES &&
            surrDensity < FLAT_SURROUND_MAX &&          // surround genuinely flat
            holeDensity > HOLE_BUSY_MIN &&              // hole has real structure
            holeDensity > surrDensity * CONTRAST_FACTOR // and far busier than surround

        return Verdict(holeDensity, surrDensity, isGhost)
    }

    private fun lum(rgb: ByteArray, idx: Int): Int {
        val o = idx * 3
        val r = rgb[o].toInt() and 0xFF
        val g = rgb[o + 1].toInt() and 0xFF
        val b = rgb[o + 2].toInt() and 0xFF
        return (r * 77 + g * 150 + b * 29) shr 8
    }

    // --- Tunables. Conservative by design (precision over recall): only fire
    // on a clearly flat surround with a clearly busy hole. Calibrate against
    // the logged densities — loosen FLAT_SURROUND_MAX / lower CONTRAST_FACTOR
    // if real ghosts slip through, tighten them if good captures get blocked. ---

    private const val EDGE_MAG_THRESHOLD = 40

    // Surround must be at least this flat for the verdict to be trusted.
    private const val FLAT_SURROUND_MAX = 0.06f

    // Hole must carry at least this much structure to count as a ghost (not
    // just sensor noise on an otherwise-clean fill).
    private const val HOLE_BUSY_MIN = 0.07f

    // Hole must be this many times busier than the surround.
    private const val CONTRAST_FACTOR = 3.5f

    private const val MIN_SAMPLES = 500
    private const val MIN_BAND_PX = 16
    private const val MAX_BAND_PX = 64
}