#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "buffer.h"
#include "aligner.h"

/**
 * Result of one stitch pass.
 *
 *   image           — RGB CV_8UC3. Reference frame with the hole replaced by real
 *                     background pixels wherever the buffer had clean samples.
 *   stillUnfilled   — CV_8UC1 binary. Pixels inside the hole that NO buffered frame
 *                     ever saw clean. These go to Phase 9 AI inpainting.
 *   sampleCount     — CV_8UC1, per-pixel count of clean buffer samples that
 *                     contributed to the median (saturating at 255). 0 means
 *                     unfilled (also flagged in stillUnfilled). Used today for
 *                     the diagnostic histogram log; will drive the green/yellow/
 *                     red confidence overlay once that UI lands.
 *   realFillRatio   — filled / total_hole_pixels, in [0, 1]. Confidence signal.
 */
struct StitchResult {
    cv::Mat image;
    cv::Mat stillUnfilled;
    cv::Mat sampleCount;
    float   realFillRatio = 0.f;
    // Fraction of hole pixels with ZERO clean-bg samples across the buffer
    // (the hist0 bucket). Distinct from (1 - realFillRatio), which also
    // counts pixels with 1-2 marginal samples. A high noSampleRatio means
    // no amount of buffer waiting will recover those pixels — the subject
    // never moved off them. UI uses this to prompt "step aside briefly".
    float   noSampleRatio  = 0.f;
};

/**
 * Multi-band (Laplacian pyramid) blend: fuses the patched hole region with the
 * reference outside the hole using cv::detail::MultiBandBlender. The pyramid
 * decomposes both images into spatial frequency bands and blends each band
 * separately, so low-frequency mean/tint differences match the surroundings
 * naturally while high-frequency texture inside the hole is preserved.
 *
 * Handles multi-region holes correctly, where a constant tint offset would
 * produce a visible bright/dark patch at the hole shape.
 *
 * Operates in place on [stitched]. No-op on OpenCV exception.
 */
void multiBandBlendStitch(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask);

/**
 * Unsharp mask, applied only inside the hole. Restores high-frequency detail
 * that the temporal median necessarily smooths out: every sample in the median
 * is from a slightly different sub-pixel position (small alignment errors), so
 * the median averages a fraction of a pixel of position jitter across the
 * sample stack and edges/textures end up softer than the reference outside
 * the hole.
 *
 * Procedure: blurred = GaussianBlur(image, sigma); sharpened = image +
 * amount * (image - blurred). Only writes back into the hole region.
 *
 * Defaults are conservative — sharper inputs (good alignment, lots of samples)
 * tolerate higher [amount]; noisy stitches show ringing if [amount] is pushed
 * too high. ~1.0 px sigma targets pixel-level detail without amplifying
 * mid-frequency artifacts.
 */
void unsharpMaskInHole(
        cv::Mat& image,
        const cv::Mat& holeMask,
        float sigma = 1.2f,
        float amount = 0.7f);

/**
 * Inject per-pixel gaussian noise inside the hole to match the noise floor of
 * the reference frame just outside the hole. The temporal median necessarily
 * averages out sensor noise (median of N noisy samples ≈ noise/√N), so the
 * patched region reads as artificially smooth compared to the surrounding
 * reference. On featureless surfaces (white walls, sky), this gives the patch
 * an obvious "too smooth" look that the eye picks up.
 *
 * Procedure:
 *   - Estimate per-channel high-frequency std-dev (reference − blur(reference))
 *     in a band just outside the hole.
 *   - Generate channel-independent gaussian noise of matching std-dev.
 *   - Add to [stitched] inside the hole only.
 *
 * Skips if the surrounding band is too thin to measure reliably or if the
 * measured noise floor is so low that injecting noise would make things look
 * worse (e.g. synthetic test inputs).
 */
void matchNoiseToReferenceSurround(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask,
        int bandPx = 10);

/**
 * Temporal-median stitcher.
 *
 * For each pixel inside [holeMask]:
 *   collect samples from every valid AlignedFrame where that pixel was NOT covered
 *   by a person mask (i.e. real background).
 *   - 0–4 samples → mark stillUnfilled (too few for reliable median)
 *   - 5+          → channel-wise median (prefers hi-quality frames when ≥5 available)
 *
 * Pixels outside the hole are copied verbatim from [reference].
 *
 * @param reference  the capture-moment frame
 * @param holeMask   CV_8UC1, non-zero where we want to remove (union of REMOVE persons)
 * @param aligned    frames already warped into reference's viewpoint (from Phase 7)
 */
StitchResult stitch(
        const BufferedFrame& reference,
        const cv::Mat& holeMask,
        const std::vector<AlignedFrame>& aligned);
