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
constexpr int   kMinGoodMatches  = 8;
constexpr float kRansacReproj    = 3.0f;
// MORPH_RECT: OpenCV decomposes into separable 1D horizontal + vertical
// passes → O(w+h) per pixel instead of O(w×h) for MORPH_ELLIPSE. ~60×
// faster on the 121×121 kernel. No visual difference: this dilation is a
// hidden sampling safety band, not the visible output shape.
constexpr int   kPersonMaskDilatePx = 60;
constexpr int   kDilateShape = cv::MORPH_RECT;

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
    const int detectScale = std::max(1, refGray.cols / 360); // Downscale to 360px wide instead of 240px to reduce sub-pixel jitter
    const cv::Size detectSize(refGray.cols / detectScale,
                              refGray.rows / detectScale);
    cv::Mat refGrayDetect;
    if (detectScale > 1) {
        cv::resize(refGray, refGrayDetect, detectSize, 0, 0, cv::INTER_AREA);
    } else {
        refGrayDetect = refGray;
    }
    
    // Pre-dilate reference person mask by a moderate kernel to avoid boundary keypoint leakage
    cv::Mat dilatedRefPersonMask;
    if (!reference.anyPersonMask.empty()) {
        cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
        cv::dilate(reference.anyPersonMask, dilatedRefPersonMask, kern);
    } else {
        dilatedRefPersonMask = reference.anyPersonMask;
    }
    cv::Mat refMaskFull = invertMask(dilatedRefPersonMask);
    cv::Mat refMaskDetect;
    if (!refMaskFull.empty() && detectScale > 1) {
        // Use linear interpolation and strict thresholding to completely exclude person boundary pixels
        cv::resize(refMaskFull, refMaskDetect, detectSize, 0, 0, cv::INTER_LINEAR);
        cv::threshold(refMaskDetect, refMaskDetect, 240, 255, cv::THRESH_BINARY);
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
        const auto sx = static_cast<float>(detectScale);
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
        const auto sx = static_cast<float>(akaze_scale);
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
    // newest 5 frames (most relevant for stitching). Older frames that
    // fail ORB are simply skipped — the stitcher's dual-bucket quality
    // weighting ensures enough good samples from the recent frames.
    const size_t akazeStartIdx = frames.size() > 5 ? frames.size() - 5 : 0;

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const auto& f = frames[fi];
        AlignedFrame af{};
        af.valid = false;

        if (f.image.empty()) { out.push_back(std::move(af)); continue; }

        cv::Mat gray;
        cv::cvtColor(f.image, gray, cv::COLOR_RGB2GRAY);

        // Pre-dilate candidate person mask to avoid boundary keypoint leakage
        cv::Mat dilatedCandPersonMask;
        if (!f.anyPersonMask.empty()) {
            cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
            cv::dilate(f.anyPersonMask, dilatedCandPersonMask, kern);
        } else {
            dilatedCandPersonMask = f.anyPersonMask;
        }
        cv::Mat candMaskFull = invertMask(dilatedCandPersonMask);

        // Detect-scale gray + mask, hoisted out of the ORB block so the
        // post-RANSAC ECC refinement can reuse them without recomputing.
        // Cost: one INTER_AREA resize per frame regardless of tier reached.
        cv::Mat grayDetect;
        if (detectScale > 1) {
            cv::resize(gray, grayDetect, detectSize, 0, 0, cv::INTER_AREA);
        } else {
            grayDetect = gray;
        }
        cv::Mat candMaskDetect;
        if (!candMaskFull.empty() && detectScale > 1) {
            cv::resize(candMaskFull, candMaskDetect, detectSize, 0, 0, cv::INTER_LINEAR);
            cv::threshold(candMaskDetect, candMaskDetect, 240, 255, cv::THRESH_BINARY);
        } else {
            candMaskDetect = candMaskFull;
        }

        std::vector<cv::KeyPoint> kps;
        std::vector<cv::KeyPoint> refKpsUsed;
        std::vector<cv::DMatch> matches;
        int tier = 0;

        // Tier 1: ORB
        if (!refDescOrb.empty()) {
            cv::Mat desc;
            orb->detectAndCompute(grayDetect, candMaskDetect, kps, desc);
            if (desc.empty() && !candMaskDetect.empty()) {
                kps.clear();
                orb->detectAndCompute(grayDetect, cv::noArray(), kps, desc);
            }
            if (!desc.empty()) {
                if (detectScale > 1) {
                    const auto sx = static_cast<float>(detectScale);
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
                    const auto sx = static_cast<float>(akaze_scale);
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

        if (tier == 0) {
            // Gyro fallback: when no features matched (texture-poor scene)
            // but both ref and cand carry a gyro rotation, synthesize H
            // directly from the relative rotation. Pure-rotation model is
            // exact for distant scenes; for near-field scenes it produces
            // residual parallax but still beats "no alignment at all".
            //
            // Approximate intrinsics: focal_length ≈ max(W, H), principal
            // point at image center. Real K from CameraCharacteristics
            // would be more accurate but device-dependent and not always
            // exposed; the approximation is good enough for the typical
            // ~30° handheld FOV.
            if (!reference.rotation.empty() && !f.rotation.empty()) {
                cv::Mat Rrel = reference.rotation.t() * f.rotation;
                const auto fLen = static_cast<double>(
                        std::max(refSize.width, refSize.height));
                const double cx = refSize.width * 0.5;
                const double cy = refSize.height * 0.5;
                cv::Mat K = (cv::Mat_<double>(3, 3) <<
                             fLen, 0.0,  cx,
                             0.0,  fLen, cy,
                             0.0,  0.0,  1.0);
                cv::Mat Kinv = K.inv();
                cv::Mat H = K * Rrel * Kinv;

                cv::warpPerspective(f.image, af.image, H, refSize, cv::INTER_LINEAR);

                cv::Mat gyroKern = cv::getStructuringElement(
                        kDilateShape,
                        cv::Size(2 * kPersonMaskDilatePx + 1,
                                 2 * kPersonMaskDilatePx + 1));
                cv::Mat preDilatedMask;
                if (!f.anyPersonMask.empty()) {
                    cv::dilate(f.anyPersonMask, preDilatedMask, gyroKern);
                } else {
                    preDilatedMask = f.anyPersonMask;
                }
                if (!preDilatedMask.empty()) {
                    cv::warpPerspective(preDilatedMask, af.anyPersonMask, H,
                                        refSize, cv::INTER_NEAREST);
                } else {
                    af.anyPersonMask = cv::Mat::zeros(refSize, CV_8UC1);
                }
                af.personMasks.reserve(f.personMasks.size());
                af.personTrackIds = f.personTrackIds;
                for (const auto& m : f.personMasks) {
                    cv::Mat dilated, warped;
                    cv::dilate(m, dilated, gyroKern);
                    cv::warpPerspective(dilated, warped, H, refSize, cv::INTER_NEAREST);
                    af.personMasks.push_back(std::move(warped));
                }
                {
                    cv::Mat ones(f.image.size(), CV_8UC1, cv::Scalar(255));
                    cv::warpPerspective(ones, af.validMask, H, refSize, cv::INTER_NEAREST);
                }

                // Translation gate (same as feature-matched path): reject
                // gyro-derived warps that shift the center too far. Gyro
                // captures rotation but not translation; large displacement
                // under H means near-field parallax that the pure-rotation
                // model cannot fix.
                {
                    std::vector<cv::Point2f> cSrc = {
                        cv::Point2f(refSize.width * 0.5f, refSize.height * 0.5f)
                    };
                    std::vector<cv::Point2f> cDst;
                    cv::perspectiveTransform(cSrc, cDst, H);
                    const double dx = cDst[0].x - cSrc[0].x;
                    const double dy = cDst[0].y - cSrc[0].y;
                    const double tr = std::sqrt(dx * dx + dy * dy);
                    const double trLimit = refSize.width * 0.18;
                    if (tr > trLimit) {
                        LOGD("aligner: gyro tier-0 rejected — center shift %.1f px "
                             "(limit %.1f)", tr, trLimit);
                        af.valid = false;
                        out.push_back(std::move(af));
                        continue;
                    }
                }

                af.quality = 0.3f;
                af.valid = true;
                LOGD("aligner: tier=0 → using gyro-derived H fallback");
                out.push_back(std::move(af));
                continue;
            }
            out.push_back(std::move(af)); continue;
        }

        // 3. Build point correspondences for RANSAC homography.
        std::vector<cv::Point2f> srcPts, dstPts;
        srcPts.reserve(matches.size());
        dstPts.reserve(matches.size());
        for (const auto& m : matches) {
            srcPts.push_back(kps[m.queryIdx].pt);
            dstPts.push_back(refKpsUsed[m.trainIdx].pt);
        }

        cv::Mat inlierMask;
        // RANSAC iters capped at 500 (default 2000). Handheld photo domain
        // converges well within 500; the extra 1500 iters mostly burn CPU.
        // Confidence 0.995 balances reliability vs cost.
        cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, kRansacReproj,
                                       inlierMask, /*maxIters=*/500,
                                       /*confidence=*/0.995);
        if (H.empty()) { out.push_back(std::move(af)); continue; }

        // findHomography returns a non-empty matrix even when RANSAC supported it
        // with very few inliers (sky / blank-wall scenes). Reject the result
        // unless enough matches actually agreed with the fitted model.
        int inliers = cv::countNonZero(inlierMask);
        if (inliers < kMinGoodMatches) {
            LOGD("aligner: only %d inliers (need >= %d)", inliers, kMinGoodMatches);
            out.push_back(std::move(af));
            continue;
        }

        // Sanity-check the affine block's scale. Handheld photos within the
        // ring-buffer window have ~negligible zoom; det should sit in [0.85,
        // 1.15] for normal captures. Tightened range [0.5, 2.0] rejects
        // garbage RANSAC fits (extreme shear/perspective) without false-
        // positives on legitimate handheld motion.
        double a = H.at<double>(0, 0), b = H.at<double>(0, 1);
        double c = H.at<double>(1, 0), d = H.at<double>(1, 1);
        double scaleDet = std::abs(a * d - b * c);
        if (scaleDet < 0.5 || scaleDet > 2.0) {
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
        // Compute mean inlier reprojection error; reject if > 2.5 px. Noise
        // floor at detectScale=cols/360 is ~1.5–2 px when projected back to
        // full coordinates (sub-px ORB jitter × scale factor). Gate at 2.5
        // lets clean fits through while still cutting the genuinely bad
        // homo-graphies (extreme scale, low inlier, > 3 px residual).
        double meanReprojError = 0.0;
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
            meanReprojError = (counted > 0) ? (sumError / counted) : 999.0;
            if (meanReprojError > 3.0) {
                LOGD("aligner: rejecting H with mean reproj error %.2f px "
                     "(inliers=%d)", meanReprojError, counted);
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
            // Measure how far the IMAGE CENTER moves under H. The (0,2)/(1,2)
            // entries are shift at pixel (0,0) and conflate rotation/scale —
            // a frame with mild center motion + slight tilt can show
            // hundreds of px at the origin while the actual scene shift is
            // tiny. Center displacement reflects real parallax risk.
            std::vector<cv::Point2f> centerSrc = {
                cv::Point2f(refSize.width * 0.5f, refSize.height * 0.5f)
            };
            std::vector<cv::Point2f> centerDst;
            cv::perspectiveTransform(centerSrc, centerDst, H);
            const double dx = centerDst[0].x - centerSrc[0].x;
            const double dy = centerDst[0].y - centerSrc[0].y;
            const double translation = std::sqrt(dx * dx + dy * dy);
            const double translationLimit = refSize.width * 0.18;
            if (translation > translationLimit) {
                LOGD("aligner: rejecting frame with center shift %.1f px "
                     "(limit %.1f) — too much parallax risk",
                     translation, translationLimit);
                out.push_back(std::move(af));
                continue;
            }
        }

        // 3.5. ECC sub-pixel refinement. RANSAC's feature-driven H sits at
        // pixel-level accuracy; the per-feature sub-pixel jitter (≈ 0.5 px at
        // detect scale × scale factor when projected to full coords) becomes
        // the noise floor for downstream temporal-median sharpness. ECC
        // optimizes intensity correlation between the warped candidate and
        // the reference, refining H to sub-pixel accuracy.
        //
        // Run at detect scale to bound cost — full-res ECC on 1080×1920 is
        // too slow for ~15 frames per capture. Detect-scale gray + combined
        // (ref ∩ cand) non-person mask focuses ECC on bg pixels only.
        //
        // Skip for AKAZE tier (texture-poor scenes where ECC's intensity
        // correlation is unreliable). Skip on exception (no convergence).
        // Skip when buffer is large (>20 frames): ECC adds ~200-300ms per
        // frame; on a 30-frame buffer that's 6-9s of marginal benefit.
        // The RANSAC H is already sub-2px accurate at this point.
        if (tier == 1 && frames.size() <= 20) {
            cv::Mat Hdown = H.clone();
            const auto s = static_cast<double>(detectScale);
            if (detectScale > 1) {
                // H_down = S^-1 * H_full * S, where S = diag(s,s,1).
                Hdown.at<double>(0, 2) /= s;
                Hdown.at<double>(1, 2) /= s;
                Hdown.at<double>(2, 0) *= s;
                Hdown.at<double>(2, 1) *= s;
            }
            Hdown.convertTo(Hdown, CV_32F);

            // Mask pixels that are non-person in BOTH reference and candidate.
            cv::Mat eccMask;
            if (!refMaskDetect.empty() && !candMaskDetect.empty()) {
                cv::bitwise_and(refMaskDetect, candMaskDetect, eccMask);
            } else if (!candMaskDetect.empty()) {
                eccMask = candMaskDetect;
            } else if (!refMaskDetect.empty()) {
                eccMask = refMaskDetect;
            }

            try {
                cv::TermCriteria criteria(
                        cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                        /*maxCount=*/5, /*epsilon=*/0.001);
                cv::findTransformECC(refGrayDetect, grayDetect, Hdown,
                                     cv::MOTION_HOMOGRAPHY, criteria, eccMask, 5);

                Hdown.convertTo(Hdown, CV_64F);
                if (detectScale > 1) {
                    // Scale back: H_full = S * H_down * S^-1.
                    Hdown.at<double>(0, 2) *= s;
                    Hdown.at<double>(1, 2) *= s;
                    Hdown.at<double>(2, 0) /= s;
                    Hdown.at<double>(2, 1) /= s;
                }
                H = Hdown;
            } catch (const cv::Exception&) {
                // ECC failed to converge — keep the RANSAC H, which already
                // passed all gates above.
            }
        }

        // 4. Warp image + masks into reference space.
        // Dilate person masks BEFORE warping — warpPerspective distorts mask
        // edges, and post-warp dilation can't recover pixels that the warp
        // already exposed. Pre-warp dilation ensures the safety band survives
        // the perspective transform intact.
        cv::warpPerspective(f.image, af.image, H, refSize, cv::INTER_LINEAR);

        cv::Mat dilateKern = cv::getStructuringElement(
                kDilateShape,
                cv::Size(2 * kPersonMaskDilatePx + 1, 2 * kPersonMaskDilatePx + 1));
        cv::Mat preDilatedMask;
        if (!f.anyPersonMask.empty()) {
            cv::dilate(f.anyPersonMask, preDilatedMask, dilateKern);
        } else {
            preDilatedMask = f.anyPersonMask;
        }

        if (!preDilatedMask.empty()) {
            cv::warpPerspective(preDilatedMask, af.anyPersonMask, H, refSize, cv::INTER_NEAREST);
        } else {
            af.anyPersonMask = cv::Mat::zeros(refSize, CV_8UC1);
        }

        af.personMasks.reserve(f.personMasks.size());
        af.personTrackIds = f.personTrackIds;
        {
            for (const auto& m : f.personMasks) {
                cv::Mat dilated, warped;
                cv::dilate(m, dilated, dilateKern);
                cv::warpPerspective(dilated, warped, H, refSize, cv::INTER_NEAREST);
                af.personMasks.push_back(std::move(warped));
            }
        }

        // 5. Build the validity mask: 255 where the warp landed real pixels,
        //    0 where it filled with black (out-of-frame). The stitcher uses this
        //    to avoid sampling spurious black pixels as "background".
        {
            cv::Mat ones(f.image.size(), CV_8UC1, cv::Scalar(255));
            cv::warpPerspective(ones, af.validMask, H, refSize, cv::INTER_NEAREST);
        }

        // Quality ∈ (0, 1]. Falls off as RANSAC mean reprojection error
        // grows. At meanError=0 (perfect fit) quality=1; at 1px → 0.5; at
        // 2px → 0.33. ECC refinement after this point further tightens the
        // warp but we don't re-measure residual — quality remains a
        // conservative lower bound. Stitcher uses this to prefer
        // higher-quality samples when many candidates are available.
        af.quality = static_cast<float>(1.0 / (1.0 + meanReprojError));
        af.valid = true;
        out.push_back(std::move(af));
    }

    return out;
}
