// JNI bridge for the PersonRemoval native engine.
//
// Step 6a: nativeOpenCvVersion()           — sanity check.
// Step 6b: native Session = RingBuffer.    — push frames + masks from Kotlin into a
//                                            native ring buffer for later capture-time use.
// Step 7+: alignment, stitching, blending will be added here.

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <chrono>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>
#include "buffer.h"
#include "aligner.h"
#include "stitcher.h"
#include "blender.h"

#define LOG_TAG "PRNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

/**
 * A native session owns a single ring buffer plus any scratch state we'll add later.
 * The Kotlin side keeps a `long handle` that's a pointer to this struct.
 *
 * `lastThumb` is a 32×24 grayscale thumbnail of the last frame actually pushed
 * into the buffer. Used by nativePushFrame to skip frames that are nearly
 * identical to the previous one — without this, a stationary scene fills the
 * 80-slot ring with redundant copies and we lose the older frames that DO
 * have parallax information.
 */
struct Session {
    std::unique_ptr<RingBuffer> buffer;
    cv::Mat lastThumb;
    // Heartbeat: monotonic ms timestamp of the last frame actually pushed.
    // The static-scene skip uses this to ensure that even on a perfectly still
    // scene, we still ingest a frame every ~1s — otherwise an "out of frame"
    // period (e.g. user steps away from the front camera) gets discarded as
    // "static" and the buffer never collects clean-background samples for
    // the eventual capture.
    int64_t lastPushMs = 0;
    // Detection-miss guard: fraction of pixels that were person in the last
    // pushed frame. If previous frame had significant person coverage but
    // current frame has zero AND the scene didn't change (low thumbnail diff),
    // YOLO likely missed detection → skip frame to prevent ghost silhouettes.
    float lastPersonCoverage = 0.f;
};

Session* asSession(jlong h) { return reinterpret_cast<Session*>(h); }

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeOpenCvVersion(
        JNIEnv* env, jclass /*clazz*/) {
    std::string v = cv::getVersionString();
    return env->NewStringUTF(v.c_str());
}

JNIEXPORT jlong JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeCreate(
        JNIEnv* /*env*/, jclass /*clazz*/, jint capacity) {
    auto* s = new Session();
    s->buffer = std::make_unique<RingBuffer>(static_cast<size_t>(capacity));
    LOGI("Session created, ringBuffer capacity=%d", capacity);
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeDestroy(
        JNIEnv* /*env*/, jclass /*clazz*/, jlong handle) {
    if (auto* s = asSession(handle)) {
        delete s;
        LOGI("Session destroyed");
    }
}

JNIEXPORT jint JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeBufferSize(
        JNIEnv* /*env*/, jclass /*clazz*/, jlong handle) {
    auto* s = asSession(handle);
    if (!s) return 0;
    return static_cast<jint>(s->buffer->size());
}

JNIEXPORT void JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeClear(
        JNIEnv* /*env*/, jclass /*clazz*/, jlong handle) {
    if (auto* s = asSession(handle)) {
        s->buffer->clear();
        // Also reset the static-scene-skip thumbnail and heartbeat. After a
        // buffer clear the next pushed frame is from a fundamentally different
        // context (camera flip, scene change), and we must not skip it as
        // "same as before".
        s->lastThumb.release();
        s->lastPushMs = 0;
    }
}

/**
 * Pushes one camera frame + its per-person masks into the ring buffer.
 *
 * @param rgba      width*height*4 bytes (CameraX RGBA_8888 layout)
 * @param width,height   frame dimensions
 * @param personMasks byte[][]; each inner array is width*height bytes (0 = bg, !=0 = person)
 * @param timestampMs frame capture time
 */
JNIEXPORT void JNICALL
Java_com_one5_personremoval_core_NativeSession_nativePushFrame(
        JNIEnv* env, jclass /*clazz*/,
        jlong handle,
        jbyteArray rgba, jint width, jint height,
        jobjectArray personMasks,
        jlong timestampMs,
        jfloatArray rotation9,
        jintArray trackIds) {

    auto* s = asSession(handle);
    if (!s) {
        LOGE("nativePushFrame: invalid handle");
        return;
    }

    BufferedFrame f;
    f.timestampMs = timestampMs;

    // Optional gyro rotation, row-major float[9]. Stored as CV_64FC1 3x3
    // so aligner can use double-precision arithmetic when composing with
    // intrinsics. Empty when caller passes null (no gyro available).
    if (rotation9 != nullptr && env->GetArrayLength(rotation9) >= 9) {
        jfloat* rp = env->GetFloatArrayElements(rotation9, nullptr);
        if (rp != nullptr) {
            cv::Mat R(3, 3, CV_64FC1);
            for (int i = 0; i < 9; ++i) R.at<double>(i / 3, i % 3) = rp[i];
            f.rotation = R;
            env->ReleaseFloatArrayElements(rotation9, rp, JNI_ABORT);
        }
    }

    // ---- 1. RGBA → RGB cv::Mat (drop alpha; we never use it).
    {
        jbyte* rgbaPtr = env->GetByteArrayElements(rgba, nullptr);
        if (!rgbaPtr) return;
        cv::Mat rgbaMat(height, width, CV_8UC4, rgbaPtr);
        cv::Mat rgbMat;
        cv::cvtColor(rgbaMat, rgbMat, cv::COLOR_RGBA2RGB);
        f.image = std::move(rgbMat);   // owns its own buffer (cvtColor allocates new)
        env->ReleaseByteArrayElements(rgba, rgbaPtr, JNI_ABORT);
    }

    // ---- 2. Person masks + union.
    f.anyPersonMask = cv::Mat::zeros(height, width, CV_8UC1);
    if (personMasks != nullptr) {
        jsize n = env->GetArrayLength(personMasks);
        f.personMasks.reserve(n);
        for (jsize i = 0; i < n; ++i) {
            auto arr = static_cast<jbyteArray>(env->GetObjectArrayElement(personMasks, i));
            if (!arr) continue;
            jbyte* mp = env->GetByteArrayElements(arr, nullptr);
            if (!mp) {
                // OOM / pending exception — skip this mask rather than crash on
                // cv::Mat(nullptr) inside threshold.
                env->DeleteLocalRef(arr);
                continue;
            }
            cv::Mat maskView(height, width, CV_8UC1, mp);
            // Normalize to {0, 255} so OpenCV bitwise ops behave.
            cv::Mat mask;
            cv::threshold(maskView, mask, 0, 255, cv::THRESH_BINARY);
            f.personMasks.push_back(mask);
            cv::bitwise_or(f.anyPersonMask, mask, f.anyPersonMask);
            env->ReleaseByteArrayElements(arr, mp, JNI_ABORT);
            env->DeleteLocalRef(arr);
        }
    }

    // Store per-person track IDs (parallel to personMasks) so the stitcher
    // can distinguish REMOVE vs KEEP persons at capture time.
    if (trackIds != nullptr) {
        jsize idCount = env->GetArrayLength(trackIds);
        jint* idPtr = env->GetIntArrayElements(trackIds, nullptr);
        if (idPtr) {
            f.personTrackIds.reserve(idCount);
            for (jsize i = 0; i < idCount; ++i) {
                f.personTrackIds.push_back(idPtr[i]);
            }
            env->ReleaseIntArrayElements(trackIds, idPtr, JNI_ABORT);
        }
    }

    // Detection-miss guard: if the previous pushed frame had significant
    // person coverage but this frame has zero, AND the scene hasn't changed
    // much (person is still in view but YOLO missed), skip the frame.
    // Without this, unmasked person pixels enter the median → ghost.
    {
        const int totalPx = width * height;
        const int personPx = cv::countNonZero(f.anyPersonMask);
        const float coverage = (totalPx > 0) ? static_cast<float>(personPx) / totalPx : 0.f;
        if (s->lastPersonCoverage > 0.02f && coverage < 0.001f && !s->lastThumb.empty()) {
            cv::Mat gray, thumb;
            cv::cvtColor(f.image, gray, cv::COLOR_RGB2GRAY);
            cv::resize(gray, thumb, cv::Size(32, 24), 0, 0, cv::INTER_AREA);
            cv::Mat diff;
            cv::absdiff(thumb, s->lastThumb, diff);
            if (cv::mean(diff)[0] < 6.0) {
                LOGI("pushFrame: detection miss guard — prev coverage %.1f%% "
                     "but current 0%%, scene unchanged, skipping",
                     s->lastPersonCoverage * 100);
                return;
            }
        }
        s->lastPersonCoverage = coverage;
    }

    // Rate gate: cap ingest at ~10 fps during active scenes. Without this,
    // fast YOLO cycles flood the buffer with near-identical frames and evict
    // older parallax-diverse samples.
    {
        const int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (s->lastPushMs > 0 && nowMs - s->lastPushMs < 100) {
            return;
        }
    }

    // Static-scene skip + heartbeat: compare a 32×24 grayscale thumbnail of
    // this frame against the last pushed thumbnail. If the mean abs diff is
    // below threshold AND we pushed something within the last ~1s, skip —
    // the ring buffer would just rotate fresh redundant copies and we'd
    // discard older parallax frames.
    //
    // The heartbeat clause is critical: without it, a perfectly still scene
    // (user out of frame, camera on tripod) gets entirely discarded, so when
    // the user comes back the buffer has zero clean-background samples for
    // the stitch.
    {
        cv::Mat gray, thumb;
        cv::cvtColor(f.image, gray, cv::COLOR_RGB2GRAY);
        cv::resize(gray, thumb, cv::Size(32, 24), 0, 0, cv::INTER_AREA);
        const int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (!s->lastThumb.empty() && s->lastThumb.size() == thumb.size()) {
            cv::Mat diff;
            cv::absdiff(thumb, s->lastThumb, diff);
            const double meanDiff = cv::mean(diff)[0];
            const bool scenedStill = (meanDiff < 3.0);
            const bool recentPush = (nowMs - s->lastPushMs < 500);
            if (scenedStill && recentPush) {
                return;
            }
        }
        s->lastThumb = thumb;
        s->lastPushMs = nowMs;
        f.thumbnail = thumb.clone();
    }

    s->buffer->push(std::move(f));
}

} // extern "C"

// -------------------------------------------------------------------------- //
// Phase 9 — Stitch-for-inpaint and OpenCV fallback inpaint.
// -------------------------------------------------------------------------- //

namespace {

/**
 * Helper: align + stitch using the caller-supplied REMOVE hole mask. Populates
 * a 6-slot java.lang.Object[]:
 *   [0] = byte[] stitched RGB           (w*h*3) — reference with hole filled by real samples
 *   [1] = byte[] original reference RGB (w*h*3) — pre-stitch (needed by finalize for seamless cloning)
 *   [2] = byte[] unfilled mask          (w*h, 255 = LaMa should fill)
 *   [3] = byte[] full hole mask         (w*h, 255 = blender should cross-blend) — already dilated
 *   [4] = int[]{w, h}
 *   [5] = float[]{fillRatio}
 *
 * Returns true on success.
 */
bool runStitchForInpaint(Session* s, JNIEnv* env,
                         jbyteArray jRemoveMask, jint maskW, jint maskH,
                         jintArray jRemoveTrackIds,
                         jintArray jKeepTrackIds,
                         jobjectArray outArr) {
    auto snap = s->buffer->snapshot();
    if (snap.empty()) {
        LOGI("stitchForInpaint: buffer empty");
        return false;
    }
    // Find the newest frame by timestamp — content-based eviction can place
    // it at any slot, so snap.back() is NOT reliable as the reference.
    size_t newestIdx = 0;
    for (size_t i = 1; i < snap.size(); ++i) {
        if (snap[i].timestampMs > snap[newestIdx].timestampMs)
            newestIdx = i;
    }
    BufferedFrame reference = std::move(snap[newestIdx]);
    snap.erase(snap.begin() + static_cast<std::ptrdiff_t>(newestIdx));

    // The REMOVE mask must match the reference frame size. If not, the caller's
    // detection is stale (frame size changed mid-capture) — fail rather than warp.
    if (maskW != reference.image.cols || maskH != reference.image.rows) {
        LOGW("stitchForInpaint: REMOVE mask %dx%d != reference %dx%d",
             maskW, maskH, reference.image.cols, reference.image.rows);
        return false;
    }

    // Pull the REMOVE mask bytes (Kotlin sends 0/1) and threshold to 0/255 so
    // bitwise / morph ops behave.
    jbyte* maskPtr = env->GetByteArrayElements(jRemoveMask, nullptr);
    if (!maskPtr) return false;
    cv::Mat removeHole;
    {
        cv::Mat view(maskH, maskW, CV_8UC1, maskPtr);
        cv::threshold(view, removeHole, 0, 255, cv::THRESH_BINARY);
    }
    env->ReleaseByteArrayElements(jRemoveMask, maskPtr, JNI_ABORT);

    if (cv::countNonZero(removeHole) == 0) {
        LOGI("stitchForInpaint: REMOVE hole is empty");
        return false;
    }

    // Viewpoint coherence pre-filter: discard frames where the CAMERA
    // moved significantly (walk-away scenario). Uses gyro rotation to
    // distinguish camera motion from scene-content change (person stepping
    // aside). A large thumbnail diff with SMALL gyro displacement means
    // the person moved, not the camera — those frames are valuable and
    // must be kept. Only discard when both thumbnail AND gyro confirm
    // the camera itself moved to a very different position.
    {
        size_t before = snap.size();
        const bool refHasRot = !reference.rotation.empty();
        cv::Mat refThumb;
        {
            cv::Mat refGray;
            cv::cvtColor(reference.image, refGray, cv::COLOR_RGB2GRAY);
            cv::resize(refGray, refThumb, cv::Size(32, 24), 0, 0, cv::INTER_AREA);
        }
        snap.erase(
            std::remove_if(snap.begin(), snap.end(),
                [&reference, refHasRot, &refThumb](const BufferedFrame& f) {
                    if (refHasRot && !f.rotation.empty()) {
                        cv::Mat Rrel = reference.rotation.t() * f.rotation;
                        double tr = cv::trace(Rrel)[0];
                        double cosAngle = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
                        double angleDeg = std::acos(cosAngle) * 180.0 / CV_PI;
                        return angleDeg > 25.0;
                    }
                    if (f.thumbnail.empty()) return false;
                    cv::Mat diff;
                    cv::absdiff(f.thumbnail, refThumb, diff);
                    return cv::mean(diff)[0] > 40.0;
                }),
            snap.end());
        LOGI("stitchForInpaint: viewpoint filter kept %zu/%zu frames",
             snap.size(), before);
    }

    // Two-stage dilation:
    // 1. Uniform 30px elliptical dilation (silhouette undercoverage: hair, jaw, hands)
    // 2. Directional downward expansion by 60px (cast shadow on floor/wall below person)
    cv::Mat dilatedHole;
    {
        cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(61, 61));
        cv::dilate(removeHole, dilatedHole, kern);

        // Shadow expansion: extend downward using a kernel that's tall
        // (reach below the feet) and roughly as wide as a foot's footprint
        // (cover the cast shadow's actual width). The old 61×3 strip was
        // effectively a centerline column — real shadows fall under the
        // entire foot width, so most of the shadow leaked past the hole
        // and reappeared as ghosting in the stitched result.
        cv::Mat shadowKern = cv::Mat::zeros(61, 31, CV_8UC1);
        // Fill only the bottom half of the kernel → dilates downward only.
        // Width stays full (31 px) so the dilation covers the foot footprint.
        shadowKern(cv::Rect(0, 30, 31, 31)).setTo(255);
        cv::dilate(dilatedHole, dilatedHole, shadowKern);
    }

    // Parse REMOVE + KEEP track IDs. The REMOVE set gates whether we rebuild
    // the exclusion masks at all; the KEEP set decides which person pixels may
    // still serve as background (see the rebuild below).
    std::unordered_set<int> removeIds;
    if (jRemoveTrackIds != nullptr) {
        jsize idCount = env->GetArrayLength(jRemoveTrackIds);
        jint* idPtr = env->GetIntArrayElements(jRemoveTrackIds, nullptr);
        if (idPtr) {
            for (jsize i = 0; i < idCount; ++i) removeIds.insert(idPtr[i]);
            env->ReleaseIntArrayElements(jRemoveTrackIds, idPtr, JNI_ABORT);
        }
    }
    std::unordered_set<int> keepIds;
    if (jKeepTrackIds != nullptr) {
        jsize kc = env->GetArrayLength(jKeepTrackIds);
        jint* kp = env->GetIntArrayElements(jKeepTrackIds, nullptr);
        if (kp) {
            for (jsize i = 0; i < kc; ++i) keepIds.insert(kp[i]);
            env->ReleaseIntArrayElements(jKeepTrackIds, kp, JNI_ABORT);
        }
    }

    // Early-abort: if the buffer contains frames with very large camera
    // rotation relative to the reference (user walking with phone), abort
    // before the expensive alignment pass. Uses gyro rotation matrices
    // which are already stored per frame — zero compute cost.
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    if (!reference.rotation.empty()) {
        int largeMotion = 0;
        for (const auto& f : snap) {
            if (f.rotation.empty()) continue;
            cv::Mat Rrel = reference.rotation.t() * f.rotation;
            double tr = cv::trace(Rrel)[0];
            double cosA = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
            double deg = std::acos(cosA) * 180.0 / CV_PI;
            if (deg > 20.0) ++largeMotion;
        }
        if (largeMotion > static_cast<int>(snap.size()) * 3 / 4) {
            LOGI("stitchForInpaint: early abort — %d/%zu frames have >20° "
                 "camera rotation. Phone moved too much.",
                 largeMotion, snap.size());
            return false;
        }
    }
    auto aligned = alignToReference(reference, snap);

    // Rebuild each aligned frame's background-exclusion mask. The aligner
    // already used the full anyPersonMask for feature masking (correct: no
    // keypoints on ANY person). Now the stitcher needs to know which person
    // pixels may still serve as background.
    //
    // Rule: exclude any person whose track id is NOT a current KEEP id. That
    // covers the REMOVE target AND — critically — any stale-track instance of
    // a person who left the frame for >1s and returned with a fresh id
    // (Tracker.maxGapMs drops the old track). Those earlier frames still sit
    // in the ring buffer carrying the OLD id, which is neither the current
    // REMOVE id nor a KEEP id. The old "exclude only REMOVE ids" rule treated
    // them as valid background, so the departed subject leaked back into the
    // temporal median as a translucent ghost. Excluding by "not KEEP" closes
    // that leak while still admitting genuine KEEP persons, so overlapping
    // people (remove the front one, reveal the kept one behind) still work.
    if (!removeIds.empty()) {
        const cv::Size refSize = reference.image.size();
        for (auto& af : aligned) {
            if (!af.valid) continue;
            cv::Mat excludeMask = cv::Mat::zeros(refSize, CV_8UC1);
            const size_t nm = std::min(af.personMasks.size(),
                                       af.personTrackIds.size());
            for (size_t mi = 0; mi < nm; ++mi) {
                if (keepIds.count(af.personTrackIds[mi]) == 0) {
                    cv::bitwise_or(excludeMask, af.personMasks[mi], excludeMask);
                }
            }
            af.anyPersonMask = excludeMask;
        }
    }

    auto t1 = clk::now();
    auto result = stitch(reference, dilatedHole, aligned);

    // Ghost detector disabled. The detection-miss guard in pushFrame
    // prevents the primary ghost source (YOLO detection drops). The
    // ref-similarity check false-positives on background pixels whose
    // luminance naturally matches the person → forces unnecessary LaMa
    // → produces worse artifacts than the ghosts it was meant to catch.

    auto t2 = clk::now();

    // Sharpness recovery: unsharp mask inside the hole. The temporal median
    // averages sub-pixel-misaligned samples, which softens edges and texture.
    // Applied BEFORE the multi-band blend so the high-freq detail enters the
    // pyramid intact; the blend's top band carries it through to output.
    // boundarySkipPx erodes the application region so unsharp doesn't reach
    // outside-hole pixels (which would create a halo).
    unsharpMaskInHole(result.image, dilatedHole);

    // Noise floor matching: the median averages out sensor noise, so the
    // patched area reads as artificially smooth on featureless surfaces
    // (white walls / sky). Re-inject gaussian noise at the same std-dev the
    // reference has just outside the hole. Done before the blend so the
    // injected noise propagates through pyramid levels at the correct scale.
    matchNoiseToReferenceSurround(result.image, reference.image, dilatedHole);

    // Stage 5+6 combined (lighting + seam): multi-band Laplacian pyramid
    // blend. Fuses the stitched hole region with the reference outside via
    // Burt-Adelson pyramid blending. The low-freq bands carry mean/tint —
    // they automatically match the surrounding context, replacing the
    // constant-offset matchStitchToReferenceTint that produced visible
    // bright/dark boxes on multi-region holes. The high-freq bands carry
    // texture/detail. The boundary blends smoothly across the pyramid's
    // built-in Gaussian-weighted mask, replacing featherStitchBoundary.
    multiBandBlendStitch(result.image, reference.image, dilatedHole);
    auto t3 = clk::now();

    auto alignMs  = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto stitchMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto polishMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    LOGI("stitchForInpaint: align=%lldms stitch=%lldms polish=%lldms fill=%.2f",
         (long long)alignMs, (long long)stitchMs, (long long)polishMs,
         result.realFillRatio);

    if (result.image.empty()) return false;

    const int W = result.image.cols;
    const int H = result.image.rows;
    const size_t rgbLen = static_cast<size_t>(W) * H * 3;
    const size_t maskLen = static_cast<size_t>(W) * H;

    // Allocate all 6 outputs up front; bail if any fail (Java refs leak only as
    // local refs and are cleaned up on JNI return).
    jbyteArray jrgb       = env->NewByteArray(static_cast<jsize>(rgbLen));
    jbyteArray joriginal  = env->NewByteArray(static_cast<jsize>(rgbLen));
    jbyteArray junfilled  = env->NewByteArray(static_cast<jsize>(maskLen));
    jbyteArray jhole      = env->NewByteArray(static_cast<jsize>(maskLen));
    jintArray  jdims      = env->NewIntArray(2);
    // [0] = realFillRatio, [1] = noSampleRatio (hist0 / totalHole).
    jfloatArray jratio    = env->NewFloatArray(2);
    if (!jrgb || !joriginal || !junfilled || !jhole || !jdims || !jratio) return false;

    env->SetByteArrayRegion(jrgb, 0, static_cast<jsize>(rgbLen),
                            reinterpret_cast<const jbyte*>(result.image.data));
    env->SetByteArrayRegion(joriginal, 0, static_cast<jsize>(rgbLen),
                            reinterpret_cast<const jbyte*>(reference.image.data));
    if (!result.stillUnfilled.empty()) {
        env->SetByteArrayRegion(junfilled, 0, static_cast<jsize>(maskLen),
                                reinterpret_cast<const jbyte*>(result.stillUnfilled.data));
    }
    env->SetByteArrayRegion(jhole, 0, static_cast<jsize>(maskLen),
                            reinterpret_cast<const jbyte*>(dilatedHole.data));

    jint dims[2] = { W, H };
    env->SetIntArrayRegion(jdims, 0, 2, dims);
    jfloat ratios[2] = { result.realFillRatio, result.noSampleRatio };
    env->SetFloatArrayRegion(jratio, 0, 2, ratios);

    env->SetObjectArrayElement(outArr, 0, jrgb);
    env->SetObjectArrayElement(outArr, 1, joriginal);
    env->SetObjectArrayElement(outArr, 2, junfilled);
    env->SetObjectArrayElement(outArr, 3, jhole);
    env->SetObjectArrayElement(outArr, 4, jdims);
    env->SetObjectArrayElement(outArr, 5, jratio);
    return true;
}

}  // namespace

extern "C" {

/**
 * Phase 9 entry point. Runs stitching and returns the four pieces of data the
 * Kotlin side needs to drive the inpaint stage:
 *   rgb bytes, unfilled mask bytes, [w,h] int[], fillRatio float[].
 *
 * Caller passes an Object[4] which we populate. Returns true on success.
 */
JNIEXPORT jboolean JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeStitchForInpaint(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jbyteArray removeMask, jint maskW, jint maskH,
        jintArray removeTrackIds,
        jintArray keepTrackIds,
        jobjectArray outArr) {
    auto* s = asSession(handle);
    if (!s) return JNI_FALSE;
    if (!outArr || env->GetArrayLength(outArr) < 6) return JNI_FALSE;
    if (!removeMask || maskW <= 0 || maskH <= 0) return JNI_FALSE;
    if (env->GetArrayLength(removeMask) < (jsize)((size_t)maskW * maskH)) return JNI_FALSE;
    return runStitchForInpaint(s, env, removeMask, maskW, maskH,
                               removeTrackIds, keepTrackIds, outArr) ? JNI_TRUE : JNI_FALSE;
}

/**
 * OpenCV cv::inpaint fallback used if LaMa fails. Uses Navier-Stokes method —
 * solves a fluid-flow PDE that propagates isophotes (lines of equal intensity)
 * into the hole. Smoother and less streaky than Telea (cv::INPAINT_TELEA) on
 * large holes, especially across textured boundaries. Slightly slower than
 * Telea but well within budget. The whole point is to always produce *some*
 * result so the app never ships an empty pixel region.
 *
 * @param rgb   raw RGB bytes, w*h*3
 * @param mask  raw mask bytes, w*h, non-zero = inpaint here
 * @param w,h   dimensions
 * @return  raw RGB bytes (w*h*3) of inpainted image
 */
JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeOpencvInpaint(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray rgb, jbyteArray mask, jint w, jint h) {

    if (w <= 0 || h <= 0) {
        LOGW("nativeOpencvInpaint: invalid dimensions %dx%d", w, h);
        return nullptr;
    }
    const size_t rgbLen = static_cast<size_t>(w) * h * 3;
    const size_t maskLen = static_cast<size_t>(w) * h;
    if (env->GetArrayLength(rgb) < (jsize)rgbLen ||
        env->GetArrayLength(mask) < (jsize)maskLen) {
        LOGW("nativeOpencvInpaint: input size mismatch");
        return nullptr;
    }

    jbyte* rgbPtr = env->GetByteArrayElements(rgb, nullptr);
    jbyte* maskPtr = env->GetByteArrayElements(mask, nullptr);
    if (!rgbPtr || !maskPtr) {
        if (rgbPtr)  env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);
        if (maskPtr) env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);
        return nullptr;
    }

    cv::Mat rgbMat(h, w, CV_8UC3, rgbPtr);
    cv::Mat maskMat(h, w, CV_8UC1, maskPtr);

    // cv::inpaint wants BGR.
    cv::Mat bgr;
    cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);

    cv::Mat inpainted;
    try {
        cv::inpaint(bgr, maskMat, inpainted, 3.0, cv::INPAINT_NS);
    } catch (const cv::Exception& e) {
        LOGE("cv::inpaint threw: %s", e.what());
        env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);
        env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);
        return nullptr;
    }

    cv::Mat outRgb;
    cv::cvtColor(inpainted, outRgb, cv::COLOR_BGR2RGB);

    env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);

    jbyteArray out = env->NewByteArray(static_cast<jsize>(rgbLen));
    if (!out) return nullptr;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(rgbLen),
                            reinterpret_cast<const jbyte*>(outRgb.data));
    return out;
}

/**
 * Phase 10 — final polish pass. Takes the (already-stitched-and-maybe-inpainted)
 * `filled` image plus the original `reference` and the hole mask, returns the
 * blended/color-corrected/feathered final image as raw RGB bytes.
 *
 *   referenceRgb : raw RGB, w*h*3 — the capture-moment frame
 *   filledRgb    : raw RGB, w*h*3 — stitched + inpainted result
 *   holeMask     : w*h bytes, non-zero where fill happened
 *
 * Returns the finalized RGB bytes (w*h*3), or null on failure.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeFinalize(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray referenceRgb, jbyteArray filledRgb, jbyteArray holeMask,
        jint w, jint h) {

    if (w <= 0 || h <= 0) {
        LOGW("nativeFinalize: invalid dimensions %dx%d", w, h);
        return nullptr;
    }
    const size_t rgbLen  = static_cast<size_t>(w) * h * 3;
    const size_t maskLen = static_cast<size_t>(w) * h;
    if (env->GetArrayLength(referenceRgb) < (jsize)rgbLen ||
        env->GetArrayLength(filledRgb) < (jsize)rgbLen ||
        env->GetArrayLength(holeMask) < (jsize)maskLen) {
        LOGW("nativeFinalize: input size mismatch");
        return nullptr;
    }

    jbyte* refPtr  = env->GetByteArrayElements(referenceRgb, nullptr);
    jbyte* fillPtr = env->GetByteArrayElements(filledRgb, nullptr);
    jbyte* maskPtr = env->GetByteArrayElements(holeMask, nullptr);
    if (!refPtr || !fillPtr || !maskPtr) {
        if (refPtr)  env->ReleaseByteArrayElements(referenceRgb, refPtr, JNI_ABORT);
        if (fillPtr) env->ReleaseByteArrayElements(filledRgb, fillPtr, JNI_ABORT);
        if (maskPtr) env->ReleaseByteArrayElements(holeMask, maskPtr, JNI_ABORT);
        return nullptr;
    }

    cv::Mat refMat (h, w, CV_8UC3, refPtr);
    cv::Mat fillMat(h, w, CV_8UC3, fillPtr);
    cv::Mat maskMat(h, w, CV_8UC1, maskPtr);

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    cv::Mat finalImg;
    try {
        finalImg = finalize(refMat, fillMat, maskMat);
    } catch (const cv::Exception& e) {
        LOGW("nativeFinalize: cv exception: %s", e.what());
        finalImg = fillMat.clone();    // best-effort: ship the unblended fill
    }
    auto t1 = clk::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    LOGI("finalize: %lldms", (long long)ms);

    env->ReleaseByteArrayElements(referenceRgb, refPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(filledRgb, fillPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(holeMask, maskPtr, JNI_ABORT);

    if (finalImg.empty()) return nullptr;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(rgbLen));
    if (!out) return nullptr;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(rgbLen),
                            reinterpret_cast<const jbyte*>(finalImg.data));
    return out;
}

/**
 * Phase 10 — encode raw RGB to JPEG (so Kotlin can save via MediaStore).
 */
JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeEncodeJpeg(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray rgb, jint w, jint h, jint quality) {

    if (w <= 0 || h <= 0) {
        LOGW("nativeEncodeJpeg: invalid dimensions %dx%d", w, h);
        return nullptr;
    }
    const size_t rgbLen = static_cast<size_t>(w) * h * 3;
    if (env->GetArrayLength(rgb) < (jsize)rgbLen) return nullptr;

    jbyte* rgbPtr = env->GetByteArrayElements(rgb, nullptr);
    if (!rgbPtr) return nullptr;
    cv::Mat rgbMat(h, w, CV_8UC3, rgbPtr);

    cv::Mat bgr;
    cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);
    env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);

    std::vector<uchar> jpeg;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", bgr, jpeg, params)) return nullptr;

    jbyteArray out = env->NewByteArray(static_cast<jsize>(jpeg.size()));
    if (!out) return nullptr;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(jpeg.size()),
                            reinterpret_cast<const jbyte*>(jpeg.data()));
    return out;
}

/**
 * YOLO input preprocessing: letterbox resize + RGBA→RGB + optional float32
 * normalization, using OpenCV's SIMD-optimized kernels instead of Kotlin
 * per-pixel loops.
 *
 * Writes directly into a DirectByteBuffer that backs the TFLite input tensor.
 * Layout: [1, 640, 640, 3] HWC, either float32 ([0,1]) or uint8 ([0,255]).
 * Padding value: 114/255 for float32, 114 for uint8.
 */
JNIEXPORT void JNICALL
Java_com_one5_personremoval_ml_YoloSegmenter_nativeFillInputBuffer(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray rgba, jint srcW, jint srcH,
        jint dstW, jint dstH, jint padX, jint padY,
        jobject inputBuffer, jboolean isFloat32) {

    constexpr int INPUT_SIZE = 640;

    jbyte* rgbaPtr = env->GetByteArrayElements(rgba, nullptr);
    if (!rgbaPtr) return;

    // RGBA → RGB
    cv::Mat rgbaMat(srcH, srcW, CV_8UC4, rgbaPtr);
    cv::Mat rgbMat;
    cv::cvtColor(rgbaMat, rgbMat, cv::COLOR_RGBA2RGB);
    env->ReleaseByteArrayElements(rgba, rgbaPtr, JNI_ABORT);

    // Nearest-neighbor resize to letterbox dimensions
    cv::Mat resized;
    cv::resize(rgbMat, resized, cv::Size(dstW, dstH), 0, 0, cv::INTER_NEAREST);

    void* bufPtr = env->GetDirectBufferAddress(inputBuffer);
    if (!bufPtr) return;

    if (isFloat32) {
        // Build padded 640×640 float32 mat, pre-filled with 114/255
        const float padVal = 114.0f / 255.0f;
        cv::Mat padded(INPUT_SIZE, INPUT_SIZE, CV_32FC3, cv::Scalar(padVal, padVal, padVal));

        // Convert resized uint8 → float32 normalized [0,1]
        cv::Mat resizedF;
        resized.convertTo(resizedF, CV_32FC3, 1.0 / 255.0);

        // Copy into the padded region
        resizedF.copyTo(padded(cv::Rect(padX, padY, dstW, dstH)));

        // Copy to DirectByteBuffer (HWC layout — continuous, same as Mat layout)
        memcpy(bufPtr, padded.data, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
    } else {
        // uint8 path
        cv::Mat padded(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(padX, padY, dstW, dstH)));
        memcpy(bufPtr, padded.data, INPUT_SIZE * INPUT_SIZE * 3);
    }
}

/**
 * In-place tint harmonization for the LaMa-filled region. Computes the
 * mean RGB shift between the unfilled mask interior and the 1-px ring of
 * real pixels immediately outside, then applies the (clamped) shift to
 * every masked pixel. Faithful port of CaptureUseCase.harmonizeLamaRegion;
 * the 41×41 mask-presence sampling on a 4-px grid is preserved verbatim.
 *
 * Cost on the Kotlin side was 150-250 ms (double full-image loop). The
 * native version drops it to ~5-15 ms — same algorithm, but raw pointer
 * loops + sequential RGB indexing keep it inside the L1 cache.
 */
JNIEXPORT void JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeHarmonizeLamaRegion(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray rgb, jbyteArray mask, jint w, jint h) {
    if (w <= 0 || h <= 0) return;
    const size_t rgbLen  = static_cast<size_t>(w) * h * 3;
    const size_t maskLen = static_cast<size_t>(w) * h;
    if (env->GetArrayLength(rgb)  < (jsize)rgbLen)  return;
    if (env->GetArrayLength(mask) < (jsize)maskLen) return;

    jbyte* rgbPtr  = env->GetByteArrayElements(rgb, nullptr);
    jbyte* maskPtr = env->GetByteArrayElements(mask, nullptr);
    if (!rgbPtr || !maskPtr) {
        if (rgbPtr)  env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);
        if (maskPtr) env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);
        return;
    }

    auto* rgbU  = reinterpret_cast<uint8_t*>(rgbPtr);
    auto* maskU = reinterpret_cast<const uint8_t*>(maskPtr);

    constexpr int BAND_PX = 20;
    int64_t innerR = 0, innerG = 0, innerB = 0, innerN = 0;
    int64_t outerR = 0, outerG = 0, outerB = 0, outerN = 0;

    for (int y = 0; y < h; ++y) {
        const int rowBase = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = rowBase + x;
            if (maskU[i]) {
                const int p = i * 3;
                innerR += rgbU[p];
                innerG += rgbU[p + 1];
                innerB += rgbU[p + 2];
                ++innerN;
                continue;
            }
            const bool nearMask =
                    (x > 0     && maskU[i - 1]) ||
                    (x < w - 1 && maskU[i + 1]) ||
                    (y > 0     && maskU[i - w]) ||
                    (y < h - 1 && maskU[i + w]);
            if (!nearMask) continue;
            const int yLo = std::max(0, y - BAND_PX);
            const int yHi = std::min(h - 1, y + BAND_PX);
            const int xLo = std::max(0, x - BAND_PX);
            const int xHi = std::min(w - 1, x + BAND_PX);
            bool inBand = false;
            for (int by = yLo; by <= yHi && !inBand; by += 4) {
                const int br = by * w;
                for (int bx = xLo; bx <= xHi; bx += 4) {
                    if (maskU[br + bx]) { inBand = true; break; }
                }
            }
            if (!inBand) continue;
            const int p = i * 3;
            outerR += rgbU[p];
            outerG += rgbU[p + 1];
            outerB += rgbU[p + 2];
            ++outerN;
        }
    }

    auto release = [&](){
        env->ReleaseByteArrayElements(rgb,  rgbPtr,  0);  // commit rgb writes
        env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);
    };

    if (outerN < 500 || innerN < 200) { release(); return; }

    // Tone clamp widened 15 → 25: the fill on a low-fill hole is a flat blob
    // whose mean can sit further off the surround than ±15. A wider cap lets
    // the shift actually reach the surrounding tone (killing the visible
    // silhouette) while still bounding a runaway correction from a bad ring
    // measurement. Only captures whose mismatch exceeds 15 change at all;
    // well-matched fills (diff < 15) are unaffected.
    const auto clampTone = [](int v) {
        return v < -25 ? -25 : (v > 25 ? 25 : v);
    };
    const int dr = clampTone(static_cast<int>(
            static_cast<float>(outerR) / outerN -
            static_cast<float>(innerR) / innerN));
    const int dg = clampTone(static_cast<int>(
            static_cast<float>(outerG) / outerN -
            static_cast<float>(innerG) / innerN));
    const int db = clampTone(static_cast<int>(
            static_cast<float>(outerB) / outerN -
            static_cast<float>(innerB) / innerN));

    if (dr == 0 && dg == 0 && db == 0) { release(); return; }

    LOGI("harmonize: shift R=%d G=%d B=%d (outer=%lld inner=%lld)",
         dr, dg, db, (long long)outerN, (long long)innerN);

    const auto clamp255 = [](int v) {
        return v < 0 ? uint8_t(0) : (v > 255 ? uint8_t(255) : uint8_t(v));
    };
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        if (!maskU[i]) continue;
        const int p = i * 3;
        rgbU[p]     = clamp255(static_cast<int>(rgbU[p])     + dr);
        rgbU[p + 1] = clamp255(static_cast<int>(rgbU[p + 1]) + dg);
        rgbU[p + 2] = clamp255(static_cast<int>(rgbU[p + 2]) + db);
    }

    release();
}

/**
 * Post-fill grain match for the inpainted region. LaMa (and Telea) produce a
 * smooth, low-frequency fill; against the grainy real surround that reads as
 * an unnaturally clean patch the eye locks onto. This injects sensor noise
 * matching the band just OUTSIDE the hole into the filled pixels so the patch
 * sits in the same grain.
 *
 * Reuses matchNoiseToReferenceSurround with the image as its own reference —
 * it reads the outside-hole band for noise stats and writes only inside the
 * hole, so the single-buffer aliasing is safe. The helper self-skips when the
 * surround is too thin (<500 px) or too clean (noise floor < 0.5) to measure,
 * so this is a no-op on synthetic / already-clean captures and can never
 * degrade a good result. Modifies [rgb] in place.
 *
 * Call AFTER finalize(): the Poisson clone there would otherwise smooth the
 * injected noise back out.
 */
JNIEXPORT void JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeTextureLamaRegion(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray rgb, jbyteArray mask, jint w, jint h) {
    if (w <= 0 || h <= 0) return;
    const size_t rgbLen  = static_cast<size_t>(w) * h * 3;
    const size_t maskLen = static_cast<size_t>(w) * h;
    if (env->GetArrayLength(rgb)  < (jsize)rgbLen)  return;
    if (env->GetArrayLength(mask) < (jsize)maskLen) return;

    jbyte* rgbPtr  = env->GetByteArrayElements(rgb, nullptr);
    jbyte* maskPtr = env->GetByteArrayElements(mask, nullptr);
    if (!rgbPtr || !maskPtr) {
        if (rgbPtr)  env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);
        if (maskPtr) env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);
        return;
    }

    cv::Mat rgbMat(h, w, CV_8UC3, rgbPtr);
    cv::Mat maskMat(h, w, CV_8UC1, maskPtr);
    cv::Mat maskBin;
    cv::threshold(maskMat, maskBin, 0, 255, cv::THRESH_BINARY);

    matchNoiseToReferenceSurround(rgbMat, rgbMat, maskBin);

    env->ReleaseByteArrayElements(rgb,  rgbPtr,  0);          // commit rgb writes
    env->ReleaseByteArrayElements(mask, maskPtr, JNI_ABORT);
}

/**
 * Fill an existing ARGB_8888 bitmap from a packed RGB byte array. Bitmap
 * dimensions must match (w, h). Alpha is set to 0xFF for every pixel.
 * Uses AndroidBitmap_lockPixels + cv::cvtColor (SIMD on arm64). Replaces
 * the Kotlin Bitmap.setPixels(IntArray) path in LamaInpainter that
 * allocated an IntArray + per-pixel pack on every inpaint invocation.
 */
JNIEXPORT jboolean JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeFillBitmapFromRgb(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray rgb, jint w, jint h, jobject bitmap) {
    if (w <= 0 || h <= 0 || !bitmap) return JNI_FALSE;
    const size_t rgbLen = static_cast<size_t>(w) * h * 3;
    if (env->GetArrayLength(rgb) < (jsize)rgbLen) return JNI_FALSE;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return JNI_FALSE;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        (int)info.width != w || (int)info.height != h) {
        return JNI_FALSE;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
        !pixels) {
        return JNI_FALSE;
    }

    jbyte* rgbPtr = env->GetByteArrayElements(rgb, nullptr);
    if (!rgbPtr) {
        AndroidBitmap_unlockPixels(env, bitmap);
        return JNI_FALSE;
    }
    cv::Mat rgbMat(h, w, CV_8UC3, rgbPtr);
    cv::Mat rgbaMat(h, w, CV_8UC4, pixels, (size_t)info.stride);
    cv::cvtColor(rgbMat, rgbaMat, cv::COLOR_RGB2RGBA);
    env->ReleaseByteArrayElements(rgb, rgbPtr, JNI_ABORT);
    AndroidBitmap_unlockPixels(env, bitmap);
    return JNI_TRUE;
}

/**
 * Read RGB bytes (drop alpha) from an ARGB_8888 bitmap into a fresh
 * byte array. SIMD cvtColor in place of per-pixel IntArray unpack.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeReadRgbFromBitmap(
        JNIEnv* env, jclass /*clazz*/, jobject bitmap) {
    if (!bitmap) return nullptr;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return nullptr;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return nullptr;

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
        !pixels) {
        return nullptr;
    }

    cv::Mat rgbaMat((int)info.height, (int)info.width, CV_8UC4, pixels,
                    (size_t)info.stride);
    cv::Mat rgbMat;
    cv::cvtColor(rgbaMat, rgbMat, cv::COLOR_RGBA2RGB);
    AndroidBitmap_unlockPixels(env, bitmap);

    const size_t rgbLen =
            static_cast<size_t>(info.width) * info.height * 3;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(rgbLen));
    if (!out) return nullptr;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(rgbLen),
                            reinterpret_cast<const jbyte*>(rgbMat.data));
    return out;
}

JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeDilateMask(
        JNIEnv* env, jclass /*clazz*/,
        jbyteArray mask, jint w, jint h, jint radiusPx) {
    if (w <= 0 || h <= 0 || radiusPx <= 0) return mask;
    const size_t len = static_cast<size_t>(w) * h;
    if (env->GetArrayLength(mask) < (jsize)len) return mask;

    jbyte* ptr = env->GetByteArrayElements(mask, nullptr);
    if (!ptr) return mask;
    cv::Mat src(h, w, CV_8UC1, ptr);
    cv::Mat bin;
    cv::threshold(src, bin, 0, 255, cv::THRESH_BINARY);
    env->ReleaseByteArrayElements(mask, ptr, JNI_ABORT);

    int kernSz = radiusPx * 2 + 1;
    cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernSz, kernSz));
    cv::Mat dilated;
    cv::dilate(bin, dilated, kern);

    jbyteArray out = env->NewByteArray(static_cast<jsize>(len));
    if (!out) return mask;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(len),
                            reinterpret_cast<const jbyte*>(dilated.data));
    return out;
}

}