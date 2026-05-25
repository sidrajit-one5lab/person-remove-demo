#pragma once

#include <opencv2/core.hpp>

/**
 * Final polish pass — applied after stitching + (optional) inpainting.
 *
 * The job is to hide the seam between the "real" pixels we already had (everything
 * outside the original hole) and the "new" pixels we just produced (everything
 * inside the hole, whether stitched from older frames or AI-invented). Even if both
 * sources are individually correct, a hard 1-pixel boundary between them looks
 * unnatural to the eye.
 *
 *   - Poisson seamless cloning blends colors across the seam without disturbing
 *     internal texture/gradients.
 *   - CLAHE on the filled region softens any leftover local-contrast mismatch.
 *   - A Gaussian-feathered boundary smooths the remaining 1-pixel ring.
 *
 * Inputs are RGB CV_8UC3. holeMask is CV_8UC1 (non-zero where pixels were filled).
 * Returns a new image with the same size/type as the inputs.
 */
cv::Mat finalize(
        const cv::Mat& reference,        // original capture-moment frame
        const cv::Mat& filled,           // reference with the hole replaced by stitch+inpaint
        const cv::Mat& holeMask);        // where the fill happened
