#include "aligner.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>   // findTransformECC
#include <android/log.h>

#define LOG_TAG "PRNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace {

constexpr int   kOrbFeatures     = 1000;
// Lowered from 12 → 8. Scenes with mostly featureless walls/ceiling (selfie
// indoors, close subjects) struggle to hit 12 ORB inliers even when the
// alignment is genuinely correct. The homography determinant sanity check
// below (|det| ∈ [0.25, 4.0]) still rejects the truly degenerate fits.
constexpr int   kMinGoodMatches  = 4;      // lowered: imperfect alignment > no alignment
constexpr float kRansacReproj    = 5.0f;   // pixels — permissive RANSAC
// Safety margin around historical persons. Symmetric with the removal-mask
// dilation in jni_bridge.cpp (Size(61,61) ≈ 30 px radius). If we trust 30 px
// of slop around the *target* silhouette, we must trust the same around past
// silhouettes when deciding "is this pixel clean background" — otherwise a
// past-person halo leaks into the median and the stitched region carries
// faint face/hair tint.
constexpr int   kPersonMaskDilatePx = 40;

// Returns the inverse of a binary mask: where mask==0, output==255 (and vice-versa).
// Feeding this to ORB tells it "only detect features in the non-person regions."
cv::Mat invertMask(const cv::Mat& mask) {
    cv::Mat inv;
    if (mask.empty()) return inv;
    cv::bitwise_not(mask, inv);
    return inv;
}

}  // namespace

std::vector<AlignedFrame> alignToReference(
        const BufferedFrame& reference,
        const std::vector<BufferedFrame>& frames) {

    std::vector<AlignedFrame> out;
    out.reserve(frames.size());

    if (reference.image.empty()) {
        LOGD("alignToReference: empty reference");
        return out;
    }

    // ---- Reference frame features (compute once, reuse for every match).
    cv::Mat refGray;
    cv::cvtColor(reference.image, refGray, cv::COLOR_RGB2GRAY);

    // ORB at full resolution gets expensive once the analyzer feeds us 720×960
    // frames (≈9× the pixels of 240×320 → 4s+ ORB time per capture across the
    // buffer). Detect features at a ~240-wide downsampled view, then scale the
    // keypoints back to full coordinates so RANSAC, the reprojection-error
    // gate, and warping all still operate in full-resolution coordinate space
    // (no threshold tuning needed). Sub-pixel residual from this is ~1–2 full-
    // pixels, which the ECC refinement below pulls back down.
    const int detectScale = std::max(1, refGray.cols / 240);
    const cv::Size detectSize(refGray.cols / detectScale,
                              refGray.rows / detectScale);
    cv::Mat refGrayDetect;
    if (detectScale > 1) {
        cv::resize(refGray, refGrayDetect, detectSize, 0, 0, cv::INTER_AREA);
    } else {
        refGrayDetect = refGray;
    }
    cv::Mat refMaskFull = invertMask(reference.anyPersonMask);
    cv::Mat refMaskDetect;
    if (!refMaskFull.empty() && detectScale > 1) {
        cv::resize(refMaskFull, refMaskDetect, detectSize, 0, 0, cv::INTER_NEAREST);
    } else {
        refMaskDetect = refMaskFull;
    }

    // Tier 1: ORB at ~240px wide
    auto orb = cv::ORB::create(kOrbFeatures);
    std::vector<cv::KeyPoint> refKpsOrb;
    cv::Mat refDescOrb;
    orb->detectAndCompute(refGrayDetect, refMaskDetect, refKpsOrb, refDescOrb);

    if (refDescOrb.empty() && !refMaskDetect.empty()) {
        LOGD("alignToReference: no ORB features in non-person regions, retrying full frame");
        refKpsOrb.clear();
        orb->detectAndCompute(refGrayDetect, cv::noArray(), refKpsOrb, refDescOrb);
    }

    if (detectScale > 1 && !refDescOrb.empty()) {
        const float sx = static_cast<float>(detectScale);
        for (auto& kp : refKpsOrb) {
            kp.pt.x *= sx;
            kp.pt.y *= sx;
        }
    }

    // Tier 2: AKAZE at ~120px wide (coarser, more robust on texture-poor scenes)
    const int akaze_scale = std::max(1, refGray.cols / 120);
    const cv::Size akazeSize(refGray.cols / akaze_scale, refGray.rows / akaze_scale);
    cv::Mat refGrayAkaze;
    cv::resize(refGray, refGrayAkaze, akazeSize, 0, 0, cv::INTER_AREA);
    cv::Mat refMaskAkaze;
    if (!refMaskFull.empty()) {
        cv::resize(refMaskFull, refMaskAkaze, akazeSize, 0, 0, cv::INTER_NEAREST);
    }

    auto akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> refKpsAkaze;
    cv::Mat refDescAkaze;
    akaze->detectAndCompute(refGrayAkaze, refMaskAkaze, refKpsAkaze, refDescAkaze);
    if (refDescAkaze.empty() && !refMaskAkaze.empty()) {
        refKpsAkaze.clear();
        akaze->detectAndCompute(refGrayAkaze, cv::noArray(), refKpsAkaze, refDescAkaze);
    }
    if (akaze_scale > 1 && !refDescAkaze.empty()) {
        const float sx = static_cast<float>(akaze_scale);
        for (auto& kp : refKpsAkaze) {
            kp.pt.x *= sx;
            kp.pt.y *= sx;
        }
    }

    if (refDescOrb.empty() && refDescAkaze.empty()) {
        LOGD("alignToReference: reference has no features (ORB + AKAZE)");
        out.resize(frames.size());
        return out;
    }

    cv::BFMatcher matcherHamming(cv::NORM_HAMMING, /*crossCheck=*/true);
    const cv::Size refSize = reference.image.size();

    // AKAZE is ~10× slower than ORB per frame. Cap AKAZE attempts to the
    // newest 10 frames (most relevant for stitching) to keep total align
    // time under ~5s. Older frames that fail ORB are simply skipped.
    const size_t akazeStartIdx = frames.size() > 10 ? frames.size() - 10 : 0;

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const auto& f = frames[fi];
        AlignedFrame af{};
        af.valid = false;

        if (f.image.empty()) { out.push_back(std::move(af)); continue; }

        cv::Mat gray;
        cv::cvtColor(f.image, gray, cv::COLOR_RGB2GRAY);
        cv::Mat candMaskFull = invertMask(f.anyPersonMask);

        std::vector<cv::KeyPoint> kps;
        std::vector<cv::KeyPoint> refKpsUsed;
        std::vector<cv::DMatch> matches;
        int tier = 0;

        // Tier 1: ORB
        if (!refDescOrb.empty()) {
            cv::Mat grayDetect;
            if (detectScale > 1) {
                cv::resize(gray, grayDetect, detectSize, 0, 0, cv::INTER_AREA);
            } else {
                grayDetect = gray;
            }
            cv::Mat candMaskDetect;
            if (!candMaskFull.empty() && detectScale > 1) {
                cv::resize(candMaskFull, candMaskDetect, detectSize, 0, 0, cv::INTER_NEAREST);
            } else {
                candMaskDetect = candMaskFull;
            }

            cv::Mat desc;
            orb->detectAndCompute(grayDetect, candMaskDetect, kps, desc);
            if (desc.empty() && !candMaskDetect.empty()) {
                kps.clear();
                orb->detectAndCompute(grayDetect, cv::noArray(), kps, desc);
            }
            if (!desc.empty()) {
                if (detectScale > 1) {
                    const float sx = static_cast<float>(detectScale);
                    for (auto& kp : kps) { kp.pt.x *= sx; kp.pt.y *= sx; }
                }
                matcherHamming.match(desc, refDescOrb, matches);
                if ((int)matches.size() >= kMinGoodMatches) {
                    refKpsUsed = refKpsOrb;
                    tier = 1;
                }
            }
        }

        // Tier 2: AKAZE fallback (newest 10 frames only)
        if (tier == 0 && !refDescAkaze.empty() && fi >= akazeStartIdx) {
            cv::Mat grayAkaze;
            cv::resize(gray, grayAkaze, akazeSize, 0, 0, cv::INTER_AREA);
            cv::Mat candMaskAkaze;
            if (!candMaskFull.empty()) {
                cv::resize(candMaskFull, candMaskAkaze, akazeSize, 0, 0, cv::INTER_NEAREST);
            }

            kps.clear();
            cv::Mat desc;
            akaze->detectAndCompute(grayAkaze, candMaskAkaze, kps, desc);
            if (desc.empty() && !candMaskAkaze.empty()) {
                kps.clear();
                akaze->detectAndCompute(grayAkaze, cv::noArray(), kps, desc);
            }
            if (!desc.empty()) {
                if (akaze_scale > 1) {
                    const float sx = static_cast<float>(akaze_scale);
                    for (auto& kp : kps) { kp.pt.x *= sx; kp.pt.y *= sx; }
                }
                matches.clear();
                matcherHamming.match(desc, refDescAkaze, matches);
                if ((int)matches.size() >= kMinGoodMatches) {
                    refKpsUsed = refKpsAkaze;
                    tier = 2;
                }
            }
        }

        if (tier == 0) { out.push_back(std::move(af)); continue; }

        // 3. Build point correspondences for RANSAC homography.
        std::vector<cv::Point2f> srcPts, dstPts;
        srcPts.reserve(matches.size());
        dstPts.reserve(matches.size());
        for (const auto& m : matches) {
            srcPts.push_back(kps[m.queryIdx].pt);
            dstPts.push_back(refKpsUsed[m.trainIdx].pt);
        }

        cv::Mat inlierMask;
        cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, kRansacReproj, inlierMask);
        if (H.empty()) { out.push_back(std::move(af)); continue; }

        // findHomography returns a non-empty matrix even when RANSAC supported it
        // with very few inliers (sky / blank-wall scenes). Reject the result
        // unless enough matches actually agreed with the fitted model.
        int inliers = inlierMask.empty() ? 0 : cv::countNonZero(inlierMask);
        if (inliers < kMinGoodMatches) {
            LOGD("aligner: only %d inliers (need >= %d)", inliers, kMinGoodMatches);
            out.push_back(std::move(af));
            continue;
        }

        // Sanity-check the affine block's scale. A wildly extreme transform
        // (e.g. squashing the frame to a sliver) means RANSAC fit garbage; the
        // resulting warp would contribute nothing usable to the stitch.
        double a = H.at<double>(0, 0), b = H.at<double>(0, 1);
        double c = H.at<double>(1, 0), d = H.at<double>(1, 1);
        double scaleDet = std::abs(a * d - b * c);
        if (scaleDet < 0.05 || scaleDet > 20.0) {
            LOGD("aligner: rejecting H with extreme scale (det=%.3f)", scaleDet);
            out.push_back(std::move(af));
            continue;
        }

        // Reprojection-error gate: even with enough inliers and a sane
        // scale, a homography can be "approximately correct" — every inlier
        // sits within kRansacReproj (3 px) of the fitted model but the
        // average error is high. When 10+ such frames go into the temporal
        // median, each contributes its own sub-pixel drift in a different
        // direction, and the median smears into the wavy/blurred upper-area
        // patch we keep seeing.
        //
        // Compute mean inlier reprojection error; reject if > 1.5 px. That
        // cuts the worst marginal homographies (those near the RANSAC
        // ceiling) while keeping the well-fit ones (errors < 1 px).
        {
            std::vector<cv::Point2f> projected;
            cv::perspectiveTransform(srcPts, projected, H);
            double sumError = 0.0;
            int counted = 0;
            const uchar* maskPtr = inlierMask.ptr<uchar>(0);
            for (size_t i = 0; i < srcPts.size(); ++i) {
                if (maskPtr[i] == 0) continue;
                const double dx = projected[i].x - dstPts[i].x;
                const double dy = projected[i].y - dstPts[i].y;
                sumError += std::sqrt(dx * dx + dy * dy);
                ++counted;
            }
            const double meanError = (counted > 0) ? (sumError / counted) : 999.0;
            if (meanError > 5.0) {
                LOGD("aligner: rejecting H with mean reproj error %.2f px "
                     "(inliers=%d)", meanError, counted);
                out.push_back(std::move(af));
                continue;
            }
        }

        // Camera-motion gate: reject the frame if the homography requires too
        // much translation to align.
        //
        // The root cause of the residual ghosting in our captures is camera
        // parallax during the buffer window. Planar homography assumes either
        // (a) pure rotation or (b) a single-depth scene. Real handheld captures
        // have small translation (camera shifted between buffer frames), and
        // real scenes have multiple depths (ceiling lights, walls, foreground
        // objects). When both happen, a homography aligns one depth perfectly
        // but leaves other depths offset — and the median across ~30 such
        // frames smears those misaligned depths into ghost shapes.
        //
        // We can't fix the parallax algebraically with a homography, but we
        // can REJECT frames whose camera moved too much from the reference.
        // Small translations have small parallax for any depth → fine. Large
        // translations have visible parallax → drop. Trade-off: fewer samples
        // per pixel in the stitcher's median, but each surviving frame
        // contributes content from nearly the same camera viewpoint.
        //
        // Threshold scales with image width so it stays perceptually similar
        // across analyzer resolutions: 5% of width = ~36 px at 720 wide,
        // ~24 px at 480 wide, ~12 px at 240 wide. Above that the frame's
        // contribution to the median is more likely to be ghost than signal.
        {
            const double tx = H.at<double>(0, 2);
            const double ty = H.at<double>(1, 2);
            const double translation = std::sqrt(tx * tx + ty * ty);
            const double translationLimit = refSize.width * 0.15;
            if (translation > translationLimit) {
                LOGD("aligner: rejecting frame with translation %.1f px "
                     "(limit %.1f) — too much parallax risk",
                     translation, translationLimit);
                out.push_back(std::move(af));
                continue;
            }
        }

        // 4. Warp image + masks into reference space.
        // Dilate person masks BEFORE warping — warpPerspective distorts mask
        // edges, and post-warp dilation can't recover pixels that the warp
        // already exposed. Pre-warp dilation ensures the safety band survives
        // the perspective transform intact.
        cv::warpPerspective(f.image, af.image, H, refSize, cv::INTER_LINEAR);

        cv::Mat preDilatedMask;
        if (!f.anyPersonMask.empty() && kPersonMaskDilatePx > 0) {
            cv::Mat kern = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(2 * kPersonMaskDilatePx + 1, 2 * kPersonMaskDilatePx + 1));
            cv::dilate(f.anyPersonMask, preDilatedMask, kern);
        } else {
            preDilatedMask = f.anyPersonMask;
        }

        if (!preDilatedMask.empty()) {
            cv::warpPerspective(preDilatedMask, af.anyPersonMask, H, refSize, cv::INTER_NEAREST);
        } else {
            af.anyPersonMask = cv::Mat::zeros(refSize, CV_8UC1);
        }

        af.personMasks.reserve(f.personMasks.size());
        for (const auto& m : f.personMasks) {
            cv::Mat dilated, warped;
            if (kPersonMaskDilatePx > 0) {
                cv::Mat kern = cv::getStructuringElement(
                        cv::MORPH_ELLIPSE,
                        cv::Size(2 * kPersonMaskDilatePx + 1, 2 * kPersonMaskDilatePx + 1));
                cv::dilate(m, dilated, kern);
            } else {
                dilated = m;
            }
            cv::warpPerspective(dilated, warped, H, refSize, cv::INTER_NEAREST);
            af.personMasks.push_back(std::move(warped));
        }

        // 5. Build the validity mask: 255 where the warp landed real pixels,
        //    0 where it filled with black (out-of-frame). The stitcher uses this
        //    to avoid sampling spurious black pixels as "background".
        {
            cv::Mat ones(f.image.size(), CV_8UC1, cv::Scalar(255));
            cv::warpPerspective(ones, af.validMask, H, refSize, cv::INTER_NEAREST);
        }

        af.valid = true;
        out.push_back(std::move(af));
    }

    return out;
}
