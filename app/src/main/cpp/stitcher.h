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
};

/**
 * Stage 5 (lighting / exposure drift): shift the stitched region's mean color
 * to match the reference's mean color in a band just outside the hole.
 * Corrects uniform tint (e.g. AWB drift across the buffer window) without
 * touching gradients — sharp detail in the patch is preserved.
 *
 * NOTE on [bandPx]: a too-narrow band hugs the very edge of the dilation
 * halo, where local variations (shadow gradient at the wall-meets-ceiling
 * line, paint subtleties, mirror reflection edges) get over-represented in
 * the mean and can produce a residual chromatic shift inside the patch. A
 * wider band averages over more of the surrounding region — more stable
 * mean, less likely to bias one channel asymmetrically.
 *
 * Operates in place on [stitched]. Outside [holeMask] the image is untouched.
 */
void matchStitchToReferenceTint(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask,
        int bandPx = 16);

/**
 * Stage 6 (seam blending): gaussian-feathered alpha-blend across the hole
 * boundary. Stitched pixels deep inside the hole and reference pixels far
 * outside remain unchanged; only a ~bandPx-wide annulus around the boundary
 * gets blended. With the removal mask dilated outward, the reference values
 * sampled inside the feather band are already background (not person), so no
 * person silhouette leaks back in.
 *
 * Operates in place on [stitched].
 *
 * NOTE on [bandPx]: must stay strictly under the removal-mask dilation radius
 * used in the JNI bridge. The feather pulls reference values from up to
 * bandPx pixels inside the hole; if that distance exceeds the dilation
 * radius, the feather would start blending in actual person pixels. With
 * the current 30-px dilation, 15 is the safe maximum.
 *
 * Wider bands hide boundary-line artifacts much better on smooth surfaces
 * (uniform walls / sky) where the eye can pin a discontinuity to a single
 * row of pixels. On textured surfaces the band width barely matters.
 */
void featherStitchBoundary(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask,
        int bandPx = 15);

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
 *   - 0 samples → mark stillUnfilled
 *   - 1 sample  → use it directly
 *   - 2+        → channel-wise median
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
