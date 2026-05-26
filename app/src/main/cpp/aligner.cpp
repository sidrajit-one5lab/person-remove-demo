#include "aligner.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>     // cv::parallel_for_
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
        cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(151, 151));
        cv::dilate(reference.anyPersonMask, dilatedRefPersonMask, kern);
    } else {
        dilatedRefPersonMask = reference.anyPersonMask;
    }
    cv::Mat refMaskFull = invertMask(dilatedRefPersonMask);
    cv::Mat refMaskDetect;
    if (!refMaskFull.empty() && detectScale > 1) {
        // Use linear interpolation and strict thresholding to completely exclude person boundary pixels
        cv::resize(refMaskFull, refMaskDetect, detectSize, 0, 0, cv::INTER_NEAREST);
        cv::threshold(refMaskDetect, refMaskDetect, 128, 255, cv::THRESH_BINARY);
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

    const cv::Size refSize = reference.image.size();

    // AKAZE is ~10× slower than ORB per frame. Cap AKAZE attempts to the
    // newest 5 frames (most relevant for stitching). Older frames that
    // fail ORB are simply skipped — the stitcher's dual-bucket quality
    // weighting ensures enough good samples from the recent frames.
    const size_t akazeStartIdx = frames.size() > 5 ? frames.size() - 5 : 0;

    // -----------------------------------------------------------------
    // Two-phase alignment.
    //
    // Phase 1 (parallel): feature detection, matching, RANSAC, geometric
    // gates, and ECC refinement. All operates on detect-scale Mats
    // (~270×480), so per-worker memory stays small and several frames
    // can run concurrently. Produces a homography H per frame.
    //
    // Phase 2 (serial): full-resolution warpPerspective of the image
    // and every mask. Each iteration allocates one frame's worth of
    // full-res warps then releases them; running this serially caps
    // peak heap to a single warped frame at a time. Warping is cheap
    // relative to feature work (~10-15 ms vs ~70-90 ms), so the
    // serialization here costs little.
    //
    // Shared reads inside Phase 1 (refDescOrb, refDescAkaze, refKpsOrb,
    // refKpsAkaze, refGrayDetect, refMaskDetect, reference.rotation,
    // detectScale, detectSize, akaze_scale, akazeSize) are immutable
    // by this point. cv::ORB / cv::AKAZE / cv::BFMatcher hold mutable
    // internal state and are recreated per worker.
    // -----------------------------------------------------------------
    struct AlignWork {
        cv::Mat H;             // empty when invalid
        bool   valid   = false;
        float  quality = 0.f;
        int    tier    = 0;    // 0=gyro fallback, 1=ORB, 2=AKAZE
    };

    std::vector<AlignWork> works(frames.size());

    cv::parallel_for_(cv::Range(0, static_cast<int>(frames.size())),
        [&](const cv::Range& range) {
            // Per-worker feature instances — sharing across threads is unsafe.
            auto orbLocal   = cv::ORB::create(kOrbFeatures);
            auto akazeLocal = cv::AKAZE::create();
            cv::BFMatcher matcherLocal(cv::NORM_HAMMING, /*crossCheck=*/true);

            for (int idx = range.start; idx < range.end; ++idx) {
                const size_t fi = static_cast<size_t>(idx);
                const auto& f = frames[fi];
                AlignWork w;

                if (f.image.empty()) { works[fi] = w; continue; }

                cv::Mat gray;
                cv::cvtColor(f.image, gray, cv::COLOR_RGB2GRAY);

                cv::Mat dilatedCandPersonMask;
                if (!f.anyPersonMask.empty()) {
                    cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
                    cv::dilate(f.anyPersonMask, dilatedCandPersonMask, kern);
                } else {
                    dilatedCandPersonMask = f.anyPersonMask;
                }
                cv::Mat candMaskFull = invertMask(dilatedCandPersonMask);

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
                    orbLocal->detectAndCompute(grayDetect, candMaskDetect, kps, desc);
                    if (desc.empty() && !candMaskDetect.empty()) {
                        kps.clear();
                        orbLocal->detectAndCompute(grayDetect, cv::noArray(), kps, desc);
                    }
                    if (!desc.empty()) {
                        if (detectScale > 1) {
                            const auto sx = static_cast<float>(detectScale);
                            for (auto& kp : kps) { kp.pt.x *= sx; kp.pt.y *= sx; }
                        }
                        matcherLocal.match(desc, refDescOrb, matches);
                        if ((int)matches.size() >= kMinGoodMatches) {
                            refKpsUsed = refKpsOrb;
                            tier = 1;
                        }
                    }
                }

                // Tier 2: AKAZE fallback (newest 5 frames only)
                if (tier == 0 && !refDescAkaze.empty() && fi >= akazeStartIdx) {
                    cv::Mat grayAkaze;
                    cv::resize(gray, grayAkaze, akazeSize, 0, 0, cv::INTER_AREA);
                    cv::Mat candMaskAkaze;
                    if (!candMaskFull.empty()) {
                        cv::resize(candMaskFull, candMaskAkaze, akazeSize, 0, 0, cv::INTER_NEAREST);
                    }

                    kps.clear();
                    cv::Mat desc;
                    akazeLocal->detectAndCompute(grayAkaze, candMaskAkaze, kps, desc);
                    if (desc.empty() && !candMaskAkaze.empty()) {
                        kps.clear();
                        akazeLocal->detectAndCompute(grayAkaze, cv::noArray(), kps, desc);
                    }
                    if (!desc.empty()) {
                        if (akaze_scale > 1) {
                            const auto sx = static_cast<float>(akaze_scale);
                            for (auto& kp : kps) { kp.pt.x *= sx; kp.pt.y *= sx; }
                        }
                        matches.clear();
                        matcherLocal.match(desc, refDescAkaze, matches);
                        if ((int)matches.size() >= kMinGoodMatches) {
                            refKpsUsed = refKpsAkaze;
                            tier = 2;
                        }
                    }
                }

                if (tier == 0) {
                    // Gyro fallback: synthesize H from relative rotation.
                    // Pure-rotation model is exact for distant scenes; for
                    // near-field scenes it leaves residual parallax but
                    // still beats no alignment.
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

                        // Translation gate: reject gyro-derived warps that
                        // shift the center too far. Gyro captures rotation
                        // but not translation; large displacement under H
                        // means near-field parallax the pure-rotation model
                        // cannot fix.
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
                            works[fi] = w;
                            continue;
                        }

                        w.H = H;
                        w.tier = 0;
                        w.quality = 0.3f;
                        w.valid = true;
                        works[fi] = w;
                        LOGD("aligner: tier=0 → using gyro-derived H fallback");
                        continue;
                    }
                    works[fi] = w;
                    continue;
                }

                // Build correspondences for RANSAC homography.
                std::vector<cv::Point2f> srcPts, dstPts;
                srcPts.reserve(matches.size());
                dstPts.reserve(matches.size());
                for (const auto& m : matches) {
                    srcPts.push_back(kps[m.queryIdx].pt);
                    dstPts.push_back(refKpsUsed[m.trainIdx].pt);
                }

                cv::Mat inlierMask;
                // RANSAC iters capped at 500 (default 2000). Handheld photo
                // domain converges well within 500; extra iters mostly burn CPU.
                cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, kRansacReproj,
                                               inlierMask, /*maxIters=*/500,
                                               /*confidence=*/0.995);
                if (H.empty()) { works[fi] = w; continue; }

                int inliers = cv::countNonZero(inlierMask);
                if (inliers < kMinGoodMatches) {
                    LOGD("aligner: only %d inliers (need >= %d)", inliers, kMinGoodMatches);
                    works[fi] = w;
                    continue;
                }

                // Sanity-check the affine block's scale to reject garbage
                // RANSAC fits (extreme shear/perspective).
                double a = H.at<double>(0, 0), b = H.at<double>(0, 1);
                double c = H.at<double>(1, 0), d = H.at<double>(1, 1);
                double scaleDet = std::abs(a * d - b * c);
                if (scaleDet < 0.5 || scaleDet > 2.0) {
                    LOGD("aligner: rejecting H with extreme scale (det=%.3f)", scaleDet);
                    works[fi] = w;
                    continue;
                }

                // Mean inlier reprojection error gate. Even with enough
                // inliers and a sane scale, an "approximately correct"
                // homography contributes per-frame sub-pixel drift that
                // smears into the temporal median. Reject > 3 px.
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
                        works[fi] = w;
                        continue;
                    }
                }

                // Camera-motion gate: reject frames whose homography
                // requires large center translation. Large translation
                // implies parallax-driven misalignment that planar
                // homography cannot fix; survivors come from near-identical
                // viewpoints.
                {
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
                        works[fi] = w;
                        continue;
                    }
                }

                // ECC sub-pixel refinement. Detect-scale only, ORB tier
                // only, small-buffer only. RANSAC H is already sub-2 px;
                // ECC pulls residual jitter below the noise floor.
                if (tier == 1 && frames.size() <= 20) {
                    cv::Mat Hdown = H.clone();
                    const auto s = static_cast<double>(detectScale);
                    if (detectScale > 1) {
                        Hdown.at<double>(0, 2) /= s;
                        Hdown.at<double>(1, 2) /= s;
                        Hdown.at<double>(2, 0) *= s;
                        Hdown.at<double>(2, 1) *= s;
                    }
                    Hdown.convertTo(Hdown, CV_32F);

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
                            Hdown.at<double>(0, 2) *= s;
                            Hdown.at<double>(1, 2) *= s;
                            Hdown.at<double>(2, 0) /= s;
                            Hdown.at<double>(2, 1) /= s;
                        }
                        H = Hdown;
                    } catch (const cv::Exception&) {
                        // Keep the RANSAC H; it already passed all gates.
                    }
                }

                w.H = H;
                w.tier = tier;
                w.quality = static_cast<float>(1.0 / (1.0 + meanReprojError));
                w.valid = true;
                works[fi] = w;
            }
        },
        /*nstripes=*/3);  // Cap workers to bound transient memory. ECC
                          // internally also uses parallel_for_; nesting
                          // beyond ~3 outer workers oversubscribes the
                          // big cores on typical arm64 devices.

    // -----------------------------------------------------------------
    // Phase 2 — apply homographies serially. One frame's full-res warps
    // live at a time; bounded peak heap regardless of buffer size.
    // -----------------------------------------------------------------
    out.resize(frames.size());

    cv::Mat dilateKern = cv::getStructuringElement(
            kDilateShape,
            cv::Size(2 * kPersonMaskDilatePx + 1, 2 * kPersonMaskDilatePx + 1));

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const auto& w = works[fi];
        if (!w.valid || w.H.empty()) continue;   // default AlignedFrame stays invalid

        const auto& f = frames[fi];
        AlignedFrame af{};
        af.valid = true;
        af.quality = w.quality;

        // Image warp.
        cv::warpPerspective(f.image, af.image, w.H, refSize, cv::INTER_LINEAR);

        // Person masks: dilate BEFORE warping. warpPerspective distorts mask
        // edges, and post-warp dilation can't recover pixels the warp already
        // exposed. Pre-warp dilation ensures the safety band survives intact.
        cv::Mat preDilatedMask;
        if (!f.anyPersonMask.empty()) {
            cv::dilate(f.anyPersonMask, preDilatedMask, dilateKern);
        }
        if (!preDilatedMask.empty()) {
            cv::warpPerspective(preDilatedMask, af.anyPersonMask, w.H, refSize, cv::INTER_NEAREST);
        } else {
            af.anyPersonMask = cv::Mat::zeros(refSize, CV_8UC1);
        }

        af.personMasks.reserve(f.personMasks.size());
        af.personTrackIds = f.personTrackIds;
        for (const auto& m : f.personMasks) {
            cv::Mat dilated, warped;
            cv::dilate(m, dilated, dilateKern);
            cv::warpPerspective(dilated, warped, w.H, refSize, cv::INTER_NEAREST);
            af.personMasks.push_back(std::move(warped));
        }

        // Validity mask: 255 where the warp landed real pixels, 0 where it
        // filled with black (out-of-frame). The stitcher uses this to avoid
        // sampling spurious black pixels as "background".
        cv::Mat ones(f.image.size(), CV_8UC1, cv::Scalar(255));
        cv::warpPerspective(ones, af.validMask, w.H, refSize, cv::INTER_NEAREST);

        out[fi] = std::move(af);
    }

    return out;
}
