#include "stitcher.h"

#include <algorithm>
#include <cmath>
#include <android/log.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#define LOG_TAG "PRNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

// Returns the median of `values` in-place (rearranges the vector).
// Channel-wise: caller invokes once per R/G/B.
inline uchar medianInPlace(std::vector<uchar>& values) {
    const size_t n = values.size();
    if (n == 0) return 0;
    if (n == 1) return values[0];
    auto mid = values.begin() + n / 2;
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

}  // namespace

StitchResult stitch(
        const BufferedFrame& reference,
        const cv::Mat& holeMask,
        const std::vector<AlignedFrame>& aligned) {

    StitchResult result;
    if (reference.image.empty() || holeMask.empty()) {
        LOGI("stitch: empty reference or hole");
        return result;
    }

    const int H = reference.image.rows;
    const int W = reference.image.cols;

    // Start from a clean copy of the reference: pixels outside the hole stay as-is.
    result.image = reference.image.clone();
    result.stillUnfilled = cv::Mat::zeros(H, W, CV_8UC1);
    result.sampleCount = cv::Mat::zeros(H, W, CV_8UC1);

    // Two scratch buckets reused across pixels to avoid alloc churn. We
    // collect samples into BOTH on each pixel: one bucket holds samples
    // from "high-quality" aligned frames (low reprojection error), the
    // other holds ALL qualifying samples. After collection we pick the
    // high-quality bucket when it has enough samples, otherwise fall back
    // to all samples — this prevents the median from being contaminated by
    // marginal-fit frames when we have plenty of sharper ones, while still
    // recovering coverage on sparse captures.
    std::vector<uchar> rsAll, gsAll, bsAll;
    std::vector<uchar> rsHi, gsHi, bsHi;
    const size_t numFrames = aligned.size();
    rsAll.reserve(numFrames); gsAll.reserve(numFrames); bsAll.reserve(numFrames);
    rsHi.reserve(numFrames);  gsHi.reserve(numFrames);  bsHi.reserve(numFrames);

    // Per-frame "is high quality" flag, computed once outside the per-pixel
    // loop. Threshold 0.6 corresponds to ~0.67 px mean reprojection error;
    // matches the noise floor at the current detect scale.
    constexpr float kHiQualityCutoff = 0.6f;
    std::vector<uint8_t> isHiQuality(numFrames, 0);
    for (size_t fi = 0; fi < numFrames; ++fi) {
        isHiQuality[fi] = (aligned[fi].quality >= kHiQualityCutoff) ? 1 : 0;
    }

    // Row-pointer caches: one entry per aligned frame, refreshed per y.
    // Eliminates ~24M bounds-checked .at<uchar>(y,x) calls per stitch on a
    // 800K-pixel hole × 30 frames. nullptr in imgRows[fi] means that frame
    // is invalid / skipped; person/valid rows can be nullptr if the
    // corresponding mask was empty.
    std::vector<const uchar*> imgRows(numFrames, nullptr);
    std::vector<const uchar*> personRows(numFrames, nullptr);
    std::vector<const uchar*> validRows(numFrames, nullptr);

    int totalHole = 0;
    int filled = 0;
    // Confidence buckets matching the spec's green/yellow/red visualization:
    //   0       → red    (no real samples — goes to AI inpaint)
    //   1–2     → yellow (marginal, alignment artifacts likely)
    //   3–9     → green  (solid)
    //   10+     → green  (high confidence)
    int hist0 = 0, hist12 = 0, hist39 = 0, hist10p = 0;

    for (int y = 0; y < H; ++y) {
        const auto* holeRow = holeMask.ptr<uchar>(y);
        auto*    outRow = result.image.ptr<cv::Vec3b>(y);
        auto*  unfilledRow  = result.stillUnfilled.ptr<uchar>(y);
        auto*  countRow     = result.sampleCount.ptr<uchar>(y);

        // Refresh per-frame row pointers for this y.
        for (size_t fi = 0; fi < numFrames; ++fi) {
            const auto& af = aligned[fi];
            if (!af.valid || af.image.empty()) {
                imgRows[fi] = nullptr;
                continue;
            }
            imgRows[fi] = af.image.ptr<uchar>(y);
            personRows[fi] = af.anyPersonMask.empty()
                             ? nullptr : af.anyPersonMask.ptr<uchar>(y);
            validRows[fi]  = af.validMask.empty()
                             ? nullptr : af.validMask.ptr<uchar>(y);
        }

        for (int x = 0; x < W; ++x) {
            if (holeRow[x] == 0) continue;          // outside the hole
            ++totalHole;

            rsAll.clear(); gsAll.clear(); bsAll.clear();
            rsHi.clear();  gsHi.clear();  bsHi.clear();

            for (size_t fi = 0; fi < numFrames; ++fi) {
                const uchar* imgRow = imgRows[fi];
                if (imgRow == nullptr) continue;    // frame invalid
                // Skip frames where this same pixel was on a person (no usable bg).
                const uchar* pRow = personRows[fi];
                if (pRow && pRow[x] != 0) continue;
                // Skip pixels that the warp filled with black (out-of-frame).
                const uchar* vRow = validRows[fi];
                if (vRow && vRow[x] == 0) continue;

                // CV_8UC3 row layout: 3 bytes/pixel, interleaved.
                const uchar* px = imgRow + x * 3;
                rsAll.push_back(px[0]); gsAll.push_back(px[1]); bsAll.push_back(px[2]);
                if (isHiQuality[fi]) {
                    rsHi.push_back(px[0]); gsHi.push_back(px[1]); bsHi.push_back(px[2]);
                }
            }

            // Prefer high-quality samples when at least 3 of them landed
            // here; otherwise use everything to keep coverage on sparse
            // captures. Both buckets are local references so downstream
            // medianInPlace operates on the chosen vector in place.
            constexpr int kHiPreferThreshold = 3;
            std::vector<uchar>& rs =
                    (rsHi.size() >= kHiPreferThreshold) ? rsHi : rsAll;
            std::vector<uchar>& gs =
                    (gsHi.size() >= kHiPreferThreshold) ? gsHi : gsAll;
            std::vector<uchar>& bs =
                    (bsHi.size() >= kHiPreferThreshold) ? bsHi : bsAll;

            const int n = static_cast<int>(rs.size());
            countRow[x] = static_cast<uchar>(std::min(n, 255));

            if (n == 0) ++hist0;
            else if (n <= 2) ++hist12;
            else if (n <= 9) ++hist39;
            else ++hist10p;

            // Adaptive routing: pixels with fewer than 3 samples have high
            // variance — the median over 1–2 samples is essentially "whatever
            // that one sample was," which on sparse buffers (fast/quiet
            // captures, static subject + static camera) means visible
            // patches of unreliable content.
            //
            // Mark them as "unfilled" so the downstream LaMa/Telea path
            // covers them with hallucinated content of consistent quality.
            // Still write the median as a *prior* (in case the AI inpaint
            // fails and we fall back to the stitcher's output), but the
            // unfilled mask says "this pixel's value isn't trustworthy."
            //
            // On good captures (10+ samples dominate, almost no 1–2 sample
            // pixels), this branch is rarely taken — no behavior change.
            // On sparse captures it's the difference between "patchy bad
            // median" and "consistent LaMa fill."
            constexpr int kMinReliableSamples = 3;
            if (n < kMinReliableSamples) {
                unfilledRow[x] = 255;
                if (n > 0) {
                    outRow[x] = cv::Vec3b(
                            medianInPlace(rs),
                            medianInPlace(gs),
                            medianInPlace(bs)
                    );
                }
                continue;
            }

            outRow[x] = cv::Vec3b(
                    medianInPlace(rs),
                    medianInPlace(gs),
                    medianInPlace(bs)
            );
            ++filled;
        }
    }

    result.realFillRatio = (totalHole > 0)
                           ? static_cast<float>(filled) / totalHole
                           : 0.f;
    result.noSampleRatio = (totalHole > 0)
                           ? static_cast<float>(hist0) / totalHole
                           : 0.f;

    LOGI("stitch: holePx=%d filledWithRealBg=%d unfilled=%d ratio=%.2f noSample=%.2f",
         totalHole, filled, totalHole - filled,
         result.realFillRatio, result.noSampleRatio);
    LOGI("stitch confidence: 0:%d  1-2:%d  3-9:%d  10+:%d",
         hist0, hist12, hist39, hist10p);

    return result;
}

// ----------------------------------------------------------------------------
// Stage 5 / Stage 6 post-stitch passes.
// ----------------------------------------------------------------------------

void matchStitchToReferenceTint(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask,
        int bandPx) {
    if (stitched.empty() || reference.empty() || holeMask.empty()) return;
    if (stitched.size() != reference.size() ||
        stitched.size() != holeMask.size()) return;

    // Build an outer ring (just outside the hole) and an inner ring (just inside).
    cv::Mat kern = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(2 * bandPx + 1, 2 * bandPx + 1));
    cv::Mat dilated, eroded, outerRing, innerRing;
    cv::dilate(holeMask, dilated, kern);
    cv::erode(holeMask, eroded, kern);
    cv::subtract(dilated, holeMask, outerRing);
    cv::subtract(holeMask, eroded, innerRing);

    const int outerCount = cv::countNonZero(outerRing);
    const int innerCount = cv::countNonZero(innerRing);
    if (outerCount < 200 || innerCount < 200) {
        // Tiny hole or hole hugging the image edge — bail rather than shift
        // the patch by a poorly-estimated diff.
        LOGI("matchTint: rings too thin (outer=%d inner=%d), skipping",
             outerCount, innerCount);
        return;
    }

    // Mean-only color match. Earlier iterations tried adding per-channel
    // variance scaling (histogram matching), but that requires sampling a
    // representative "natural" patch interior — which doesn't exist for
    // large holes that span multiple scene regions. In testing, the variance
    // sampling consistently picked the wrong reference (either the boundary
    // band with inflated edge variance, or the deep interior spanning
    // diverse content) and made things visibly worse by squashing variance
    // toward the clamp limits. The mean shift alone is well-behaved and
    // robust across scenes; visible patch artifacts that mean-only can't fix
    // (mostly large-hole selfies) need a different architectural fix (path C,
    // high-resolution capture), not more statistical post-processing.
    const cv::Scalar refMean = cv::mean(reference, outerRing);
    const cv::Scalar stitchMean = cv::mean(stitched, innerRing);
    const cv::Scalar diff = refMean - stitchMean;

    LOGI("matchTint: refRing=(%.0f,%.0f,%.0f) stitchRing=(%.0f,%.0f,%.0f) "
         "diff=(%.1f,%.1f,%.1f)",
         refMean[0], refMean[1], refMean[2],
         stitchMean[0], stitchMean[1], stitchMean[2],
         diff[0], diff[1], diff[2]);

    cv::Mat shifted32;
    stitched.convertTo(shifted32, CV_32FC3);
    cv::add(shifted32,
            cv::Scalar(diff[0], diff[1], diff[2], 0),
            shifted32);
    cv::Mat shifted8;
    shifted32.convertTo(shifted8, CV_8UC3);
    shifted8.copyTo(stitched, holeMask);
}

void featherStitchBoundary(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask,
        int bandPx) {
    if (stitched.empty() || reference.empty() || holeMask.empty()) return;
    if (stitched.size() != reference.size() ||
        stitched.size() != holeMask.size()) return;

    const int k = 2 * bandPx + 1;  // odd kernel for GaussianBlur

    // Soft alpha from the binary hole mask: ~1.0 deep inside hole, 0.0 far
    // outside, smooth gradient in the bandPx-wide ring at the boundary.
    cv::Mat alpha8;
    cv::GaussianBlur(holeMask, alpha8, cv::Size(k, k), 0);
    cv::Mat alphaF;
    alpha8.convertTo(alphaF, CV_32F, 1.0 / 255.0);

    cv::Mat stitched32, ref32;
    stitched.convertTo(stitched32, CV_32FC3);
    reference.convertTo(ref32, CV_32FC3);

    std::vector<cv::Mat> ach{alphaF, alphaF, alphaF};
    cv::Mat alpha3;
    cv::merge(ach, alpha3);

    cv::Mat blended = stitched32.mul(alpha3) +
                      ref32.mul(cv::Scalar::all(1) - alpha3);
    blended.convertTo(stitched, CV_8UC3);
}

void unsharpMaskInHole(
        cv::Mat& image,
        const cv::Mat& holeMask,
        float sigma,
        float amount) {
    if (image.empty() || holeMask.empty()) return;
    if (image.size() != holeMask.size()) return;
    if (sigma <= 0.f || amount <= 0.f) return;

    // Smooth-scene gate: if the area just outside the hole has very low
    // variance (white wall, sky, etc.), there's no high-frequency detail to
    // recover, but the eroded-boundary band creates a subtle sharpness
    // discontinuity that the eye picks up. Skip the pass entirely — adds
    // artifacts without adding detail.
    {
        cv::Mat outerKern = cv::getStructuringElement(
                cv::MORPH_RECT, cv::Size(13, 13));
        cv::Mat dilatedHole, outerRing;
        cv::dilate(holeMask, dilatedHole, outerKern);
        cv::subtract(dilatedHole, holeMask, outerRing);
        if (cv::countNonZero(outerRing) >= 200) {
            cv::Scalar mean, stddev;
            cv::meanStdDev(image, mean, stddev, outerRing);
            const double maxStd = std::max({stddev[0], stddev[1], stddev[2]});
            if (maxStd < 5.0) {
                LOGI("unsharp: smooth scene (maxStd=%.1f), skipping", maxStd);
                return;
            }
        }
    }

    // Pick an odd kernel large enough for the sigma. OpenCV's auto-sigma
    // formula expects ~3σ on each side; we use ceil(3σ)*2+1 for safety.
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    if (radius < 1) radius = 1;
    const int k = 2 * radius + 1;

    // Compute blurred version of the entire image (cheap enough at 240x320).
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(k, k), sigma);

    // sharpened = image + amount * (image - blurred), in float32 to avoid
    // wraparound on saturating channels.
    cv::Mat img32, blur32;
    image.convertTo(img32, CV_32FC3);
    blurred.convertTo(blur32, CV_32FC3);

    cv::Mat highFreq = img32 - blur32;
    cv::Mat sharpened32 = img32 + highFreq * amount;

    cv::Mat sharpened8;
    sharpened32.convertTo(sharpened8, CV_8UC3);   // saturating cast clamps to [0, 255]

    // Erode the application region by `boundarySkipPx` so the boundary band
    // is NOT sharpened. Reasons:
    //   - The unsharp gaussian extends ~3σ past each pixel; at the hole edge,
    //     that integral pulls in OUTSIDE-the-hole values, which biases the
    //     high-pass at the edge and creates a thin halo when written back.
    //   - The downstream feather pass smooths the boundary anyway; sharpening
    //     there just amplifies sub-pixel median misalignment that the eye
    //     would otherwise miss.
    // Pick the skip ≥ ceil(3σ) + 1 so the gaussian for sharpened pixels stays
    // entirely inside the hole. With σ=1.2 that's 5 px.
    const int boundarySkipPx = std::max(5, static_cast<int>(std::ceil(3.0f * sigma)) + 1);
    cv::Mat interiorMask;
    cv::Mat erodeKern = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * boundarySkipPx + 1, 2 * boundarySkipPx + 1));
    cv::erode(holeMask, interiorMask, erodeKern);

    sharpened8.copyTo(image, interiorMask);
}

void matchNoiseToReferenceSurround(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask,
        int bandPx) {
    if (stitched.empty() || reference.empty() || holeMask.empty()) return;
    if (stitched.size() != reference.size() ||
        stitched.size() != holeMask.size()) return;

    // Build the outer ring (band just outside the hole). The reference's
    // noise statistics there are what we'll inject into the patched region.
    cv::Mat kern = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(2 * bandPx + 1, 2 * bandPx + 1));
    cv::Mat dilated, outerRing;
    cv::dilate(holeMask, dilated, kern);
    cv::subtract(dilated, holeMask, outerRing);

    const int outerCount = cv::countNonZero(outerRing);
    if (outerCount < 500) {
        LOGI("matchNoise: outer ring too small (%d), skipping", outerCount);
        return;
    }

    // High-pass = reference − blur(reference). Its std-dev within the outer
    // ring is the noise floor (sensor noise + any high-freq texture). On a
    // smooth wall this is essentially just noise; on textured surfaces it
    // includes some texture contribution — that's fine, we want to match
    // whatever variation is in the surroundings.
    cv::Mat refBlurred;
    cv::GaussianBlur(reference, refBlurred, cv::Size(5, 5), 1.0);
    cv::Mat ref32, blur32;
    reference.convertTo(ref32, CV_32FC3);
    refBlurred.convertTo(blur32, CV_32FC3);
    cv::Mat highPass = ref32 - blur32;

    std::vector<cv::Mat> channels;
    cv::split(highPass, channels);
    double std[3] = {0, 0, 0};
    for (int c = 0; c < 3; ++c) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(channels[c], mean, stddev, outerRing);
        std[c] = stddev[0];
    }
    LOGI("matchNoise: stddev R=%.2f G=%.2f B=%.2f (outerRing=%d)",
         std[0], std[1], std[2], outerCount);

    // Skip if surroundings have essentially no noise (e.g. clean synthetic
    // inputs). Adding noise to a noise-free reference would make the patch
    // *more* visible, not less.
    if (std[0] < 0.5 && std[1] < 0.5 && std[2] < 0.5) {
        LOGI("matchNoise: noise floor too low, skipping");
        return;
    }

    // Generate per-channel gaussian noise at matching std-dev. cv::RNG::fill
    // with NORMAL distribution takes the mean and stddev as multi-channel
    // Scalars; a CV_32FC3 destination gets independent values per channel.
    cv::Mat noise(stitched.size(), CV_32FC3);
    cv::theRNG().fill(noise, cv::RNG::NORMAL,
                      cv::Scalar(0.0, 0.0, 0.0),
                      cv::Scalar(std[0], std[1], std[2]));

    // Add to stitched, saturating cast, write back only inside the hole.
    cv::Mat stitched32;
    stitched.convertTo(stitched32, CV_32FC3);
    cv::Mat noisy32 = stitched32 + noise;
    cv::Mat noisy8;
    noisy32.convertTo(noisy8, CV_8UC3);
    noisy8.copyTo(stitched, holeMask);
}

// ----------------------------------------------------------------------------
// Post-stitch ghost detector.
// ----------------------------------------------------------------------------

void detectGhostPixels(
        const cv::Mat& stitched,
        const cv::Mat& holeMask,
        cv::Mat& unfilled,
        const cv::Mat& sampleCount) {
    if (stitched.empty() || holeMask.empty() || unfilled.empty()) return;

    cv::Mat kern = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(33, 33));
    cv::Mat dilated, outerRing;
    cv::dilate(holeMask, dilated, kern);
    cv::subtract(dilated, holeMask, outerRing);
    if (cv::countNonZero(outerRing) < 200) return;

    cv::Mat gray;
    cv::cvtColor(stitched, gray, cv::COLOR_RGB2GRAY);
    cv::Scalar bandMean, bandStddev;
    cv::meanStdDev(gray, bandMean, bandStddev, outerRing);

    const auto refMean = static_cast<float>(bandMean[0]);
    const auto refStd  = static_cast<float>(bandStddev[0]);
    const float limit   = std::max(50.0f, refStd * 3.0f);

    int flagged = 0;
    const int H = gray.rows, W = gray.cols;
    for (int y = 0; y < H; ++y) {
        const uchar* gRow = gray.ptr<uchar>(y);
        const auto* hRow = holeMask.ptr<uchar>(y);
        auto*       uRow = unfilled.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            if (hRow[x] == 0) continue;
            if (uRow[x] != 0) continue;
            float diff = std::abs(static_cast<float>(gRow[x]) - refMean);
            if (diff > limit) {
                uRow[x] = 255;
                ++flagged;
            }
        }
    }
    LOGI("ghostDetect: flagged %d pixels (refMean=%.0f refStd=%.1f limit=%.1f)",
         flagged, refMean, refStd, limit);
}

// ----------------------------------------------------------------------------
// Multi-band (Laplacian pyramid) blend.
// ----------------------------------------------------------------------------

void multiBandBlendStitch(
        cv::Mat& stitched,
        const cv::Mat& reference,
        const cv::Mat& holeMask) {
    if (stitched.empty() || reference.empty() || holeMask.empty()) return;
    if (stitched.size() != reference.size() ||
        stitched.size() != holeMask.size()) return;
    if (stitched.type() != CV_8UC3 || reference.type() != CV_8UC3) return;
    if (holeMask.type() != CV_8UC1) return;

    const int W = stitched.cols;
    const int H = stitched.rows;

    // num_bands picked so the smallest band still has > 16 px in both dims.
    // Each pyramid level halves resolution; log2(minDim/16) is the upper
    // bound. Clamped to [2, 5] — fewer than 2 defeats the purpose; more than
    // 5 risks pyramid collapse on small inputs and inflates RAM.
    const int minDim = std::min(W, H);
    int numBands = static_cast<int>(std::floor(
            std::log2(std::max(1.0, static_cast<double>(minDim) / 16.0))));
    numBands = std::max(2, std::min(5, numBands));

    // Bail when the hole touches the image border on both inner and outer
    // mask boundaries — MultiBandBlender has been observed to produce
    // smeared border pixels in that degenerate case. The legacy
    // matchStitchToReferenceTint + featherStitchBoundary path is the
    // fallback (caller decides).
    cv::Mat refMask;
    cv::bitwise_not(holeMask, refMask);
    if (cv::countNonZero(refMask) < 200 || cv::countNonZero(holeMask) < 200) {
        LOGI("multiBand: degenerate mask coverage, skipping");
        return;
    }

    try {
        cv::detail::MultiBandBlender blender(/*try_gpu=*/false, numBands);
        cv::Rect roi(0, 0, W, H);
        blender.prepare(roi);

        cv::Mat ref16, stitched16;
        reference.convertTo(ref16, CV_16SC3);
        stitched.convertTo(stitched16, CV_16SC3);

        blender.feed(ref16, refMask, cv::Point(0, 0));
        blender.feed(stitched16, holeMask, cv::Point(0, 0));

        cv::Mat resultS, resultMask;
        blender.blend(resultS, resultMask);

        // resultS is CV_16SC3; convert back. Saturating cast clamps the
        // small overshoots that pyramid reconstruction can produce at
        // sharp-edge pixels.
        cv::Mat result8;
        resultS.convertTo(result8, CV_8UC3);
        result8.copyTo(stitched);

        LOGI("multiBand: %d bands, %dx%d", numBands, W, H);
    } catch (const cv::Exception& e) {
        LOGI("multiBand: cv exception (%s); leaving stitched unchanged",
             e.what());
    } catch (const std::exception& e) {
        LOGI("multiBand: std exception (%s); leaving stitched unchanged",
             e.what());
    }
}