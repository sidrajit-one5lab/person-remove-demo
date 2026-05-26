#include "blender.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <android/log.h>

#define LOG_TAG "PRNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace {

/**
 * cv::seamlessClone requires the mask to NOT touch the image border. If it does,
 * it throws. We erode the mask by a few pixels and clip it to a safe inset rect
 * before handing it to OpenCV.
 */
cv::Mat prepareSafeMask(const cv::Mat& holeMask, int border) {
    cv::Mat safe;
    holeMask.copyTo(safe);
    if (border > 0) {
        cv::rectangle(safe,
                      cv::Point(0, 0),
                      cv::Point(safe.cols - 1, border - 1),
                      cv::Scalar(0), cv::FILLED);
        cv::rectangle(safe,
                      cv::Point(0, safe.rows - border),
                      cv::Point(safe.cols - 1, safe.rows - 1),
                      cv::Scalar(0), cv::FILLED);
        cv::rectangle(safe,
                      cv::Point(0, 0),
                      cv::Point(border - 1, safe.rows - 1),
                      cv::Scalar(0), cv::FILLED);
        cv::rectangle(safe,
                      cv::Point(safe.cols - border, 0),
                      cv::Point(safe.cols - 1, safe.rows - 1),
                      cv::Scalar(0), cv::FILLED);
    }
    return safe;
}

/**
 * Compute the bounding-box center of the non-zero region of mask.
 *
 * seamlessClone's "point" argument is treated as the destination location of
 * the mask's bbox center, NOT its mass centroid — internally it computes
 *   destOrigin = point - bbox.size() / 2
 * and constructs `dst(Rect(destOrigin, bbox.size()))`. Passing a mass-weighted
 * centroid on an asymmetric mask makes destOrigin go negative and trips the
 * Mat(roi) bounds assertion at matrix.cpp.
 */
cv::Point maskBboxCenter(const cv::Mat& mask) {
    cv::Rect bbox = cv::boundingRect(mask);
    if (bbox.width == 0 || bbox.height == 0) {
        return {mask.cols / 2, mask.rows / 2};
    }
    return {bbox.x + bbox.width / 2, bbox.y + bbox.height / 2};
}

/**
 * Pass 1: Poisson seamless cloning of `filled` into `reference` inside the hole.
 * If OpenCV throws (rare, e.g. mask topology edge case), fall back to a feathered
 * alpha-blend so we always produce *something*.
 */
cv::Mat seamlessPass(const cv::Mat& reference, const cv::Mat& filled,
                     const cv::Mat& holeMask) {
    // Mask must not touch image border for seamlessClone.
    cv::Mat safeMask = prepareSafeMask(holeMask, /*border=*/2);
    if (cv::countNonZero(safeMask) == 0) {
        // Hole is entirely on the border — skip seamless, do feathered alpha blend.
        LOGW("blender: hole touches border, using alpha blend");
        cv::Mat feather;
        cv::GaussianBlur(holeMask, feather, cv::Size(7, 7), 0);
        cv::Mat fmask;
        feather.convertTo(fmask, CV_32F, 1.0 / 255.0);
        cv::Mat result;
        cv::Mat ref32, fill32;
        reference.convertTo(ref32, CV_32FC3);
        filled.convertTo(fill32, CV_32FC3);
        std::vector<cv::Mat> fmask3{fmask, fmask, fmask};
        cv::Mat fmaskRgb;
        cv::merge(fmask3, fmaskRgb);
        cv::Mat blended = fill32.mul(fmaskRgb) + ref32.mul(cv::Scalar::all(1) - fmaskRgb);
        blended.convertTo(result, CV_8UC3);
        return result;
    }

    // Guard: cv::seamlessClone's internal matrix construction blows up
    // (std::length_error from a vector resize) when the mask is extremely
    // sparse — e.g. 84 isolated pixels left over from a near-perfect stitch,
    // or a hole whose bbox is narrower than the Poisson stencil. Below this
    // threshold the blending wouldn't be visually meaningful anyway (the
    // upstream Telea/LaMa diffusion already produced a smooth result), so we
    // just hand back `filled` and skip the Poisson pass.
    const int safeArea = cv::countNonZero(safeMask);
    const cv::Rect bbox = cv::boundingRect(safeMask);
    if (safeArea < 100 || bbox.width < 5 || bbox.height < 5) {
        LOGW("blender: mask too sparse for seamlessClone "
             "(area=%d bbox=%dx%d), returning filled as-is",
             safeArea, bbox.width, bbox.height);
        return filled.clone();
    }

    cv::Point center = maskBboxCenter(safeMask);
    cv::Mat out;
    try {
        // NORMAL_CLONE preserves filled's gradients while matching reference's color tone
        // at the seam. This is what hides the "patchy" look between zones.
        // seamlessClone is hardcoded to work with BGR ordering internally for the
        // gradient computation, but the math is channel-agnostic so RGB works too —
        // any tone shift would be self-consistent because both inputs use the same order.
        cv::seamlessClone(filled, reference, safeMask, center, out, cv::NORMAL_CLONE);
    } catch (const cv::Exception& e) {
        LOGW("seamlessClone (cv) threw: %s, falling back to filled image", e.what());
        out = filled.clone();
    } catch (const std::exception& e) {
        // seamlessClone has been observed to throw std::length_error from
        // inside its internal vectors on edge-case masks (very sparse, tiny
        // bbox). cv::Exception doesn't catch std types, and an uncaught throw
        // out of JNI terminates the process. Keep this catch.
        LOGW("seamlessClone (std) threw: %s, falling back to filled image", e.what());
        out = filled.clone();
    } catch (...) {
        LOGW("seamlessClone threw unknown exception, falling back to filled image");
        out = filled.clone();
    }
    return out;
}

/**
 * Pass 2: CLAHE on the L channel of LAB color space, applied to the previously-hole
 * region only. This smooths leftover local contrast mismatches without affecting the
 * untouched part of the photo.
 */
cv::Mat clahePass(const cv::Mat& blended, const cv::Mat& holeMask) {
    cv::Mat lab;
    cv::cvtColor(blended, lab, cv::COLOR_RGB2Lab);

    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);

    cv::Mat lEnhanced;
    auto clahe = cv::createCLAHE(/*clipLimit=*/2.0, cv::Size(8, 8));
    clahe->apply(labChannels[0], lEnhanced);

    // Composite: lEnhanced inside the hole, original elsewhere.
    cv::Mat lFinal = labChannels[0].clone();
    lEnhanced.copyTo(lFinal, holeMask);
    labChannels[0] = lFinal;

    cv::Mat labOut;
    cv::merge(labChannels, labOut);

    cv::Mat rgbOut;
    cv::cvtColor(labOut, rgbOut, cv::COLOR_Lab2RGB);
    return rgbOut;
}

/**
 * Pass 3: feather a 3-pixel band at the mask boundary so the very-edge pixels blend
 * gradually rather than hard-cutting. Operates on the previously-hole region only.
 */
cv::Mat featherPass(const cv::Mat& original, const cv::Mat& processed,
                    const cv::Mat& holeMask) {
    cv::Mat feather;
    cv::GaussianBlur(holeMask, feather, cv::Size(7, 7), 0);
    cv::Mat fmask;
    feather.convertTo(fmask, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> fmask3{fmask, fmask, fmask};
    cv::Mat fmaskRgb;
    cv::merge(fmask3, fmaskRgb);

    cv::Mat orig32, proc32;
    original.convertTo(orig32, CV_32FC3);
    processed.convertTo(proc32, CV_32FC3);

    cv::Mat blended = proc32.mul(fmaskRgb) +
                      orig32.mul(cv::Scalar::all(1) - fmaskRgb);
    cv::Mat out;
    blended.convertTo(out, CV_8UC3);
    return out;
}

}  // namespace

cv::Mat finalize(
        const cv::Mat& reference,
        const cv::Mat& filled,
        const cv::Mat& holeMask) {

    if (reference.empty() || filled.empty() || holeMask.empty()) {
        LOGW("finalize: empty input");
        return filled.empty() ? reference.clone() : filled.clone();
    }
    if (cv::countNonZero(holeMask) == 0) {
        // Nothing was filled (no people to remove) → just return reference.
        return reference.clone();
    }

    cv::Mat blended  = seamlessPass(reference, filled, holeMask);
    cv::Mat enhanced = clahePass(blended, holeMask);
    cv::Mat final    = featherPass(reference, enhanced, holeMask);
    return final;
}
