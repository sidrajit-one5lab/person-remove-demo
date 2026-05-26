#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "buffer.h"

/**
 * One buffer frame after being warped into the reference frame's viewpoint.
 *
 * If alignment failed (too few features, RANSAC couldn't agree), [valid] is false
 * and the other fields are empty. The stitcher must skip these.
 */
struct AlignedFrame {
    cv::Mat image;                          // warped RGB, same size as reference
    std::vector<cv::Mat> personMasks;       // each warped to reference space
    std::vector<int> personTrackIds;        // parallel to personMasks; carried from BufferedFrame
    cv::Mat anyPersonMask;                  // union, warped (already dilated)
    cv::Mat validMask;                      // 255 where pixel is real, 0 where warp put black
    bool   valid   = false;
    // Quality ∈ (0, 1]. 1 / (1 + meanReprojError). Higher = sharper H fit.
    // Stitcher uses this to prefer high-quality samples when many candidates
    // are available, avoiding contamination of the median by marginal fits.
    float  quality = 0.f;
};

/**
 * Aligns every entry in [frames] to the viewpoint of [reference].
 *
 * Uses ORB features, masking out person regions before detection so static-background
 * features dominate. Returns one AlignedFrame per input; check .valid for success.
 *
 * The reference itself is NOT returned in the result — callers know it already.
 */
std::vector<AlignedFrame> alignToReference(
        const BufferedFrame& reference,
        const std::vector<BufferedFrame>& frames);
