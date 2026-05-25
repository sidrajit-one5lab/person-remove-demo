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
        jlong timestampMs) {

    auto* s = asSession(handle);
    if (!s) {
        LOGE("nativePushFrame: invalid handle");
        return;
    }

    BufferedFrame f;
    f.timestampMs = timestampMs;

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
            // ~3/255 = scene is essentially the same. Sensor noise alone runs
            // ~1–2 on a still subject, so 3 is a gentle floor.
            //
            // Heartbeat: even on a perfectly still scene we push a frame
            // every 500 ms so the smaller (30-frame) buffer still covers a
            // useful real-time window during "leave and come back" workflows.
            // (Previously 1000 ms when the buffer was 80 frames deep.)
            const bool scenedStill = (meanDiff < 3.0);
            const bool recentPush = (nowMs - s->lastPushMs < 500);
            if (scenedStill && recentPush) {
                return;  // skip push
            }
        }
        s->lastThumb = thumb;       // owns its own buffer (cv::resize allocates)
        s->lastPushMs = nowMs;
        f.thumbnail = thumb.clone();
    }

    s->buffer->push(std::move(f));
}

/**
 * Phase 7 verification: snapshot the buffer, use the newest frame as the reference,
 * align every older frame to it, log how many succeeded and how long it took.
 * Returns the success count (or -1 on error).
 */
JNIEXPORT jint JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeTestAlignment(
        JNIEnv* /*env*/, jclass /*clazz*/, jlong handle) {

    auto* s = asSession(handle);
    if (!s) return -1;

    auto snap = s->buffer->snapshot();
    if (snap.size() < 2) {
        LOGI("testAlignment: buffer has only %zu frames, need >= 2", snap.size());
        return 0;
    }

    // Newest frame = reference; everything before it = candidates.
    BufferedFrame reference = std::move(snap.back());
    snap.pop_back();

    LOGI("testAlignment: aligning %zu frames to reference %dx%d...",
         snap.size(), reference.image.cols, reference.image.rows);

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto aligned = alignToReference(reference, snap);
    auto t1 = clk::now();

    int ok = 0;
    for (const auto& a : aligned) if (a.valid) ++ok;

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    LOGI("testAlignment: aligned %d/%zu frames in %lld ms (%.1f ms/frame)",
         ok, aligned.size(), (long long)totalMs,
         aligned.empty() ? 0.0 : (double)totalMs / aligned.size());

    return ok;
}

/**
 * Phase 8 verification.
 *   1. Snapshot buffer, newest frame = reference.
 *   2. Build hole mask = union of all per-person masks in the reference (treat
 *      every detected person as REMOVE for testing purposes).
 *   3. Align older frames to reference.
 *   4. Run stitching.
 *   5. Encode the stitched RGB image as a JPEG and return the bytes.
 *
 * Returns null if the buffer is empty or no people were detected in the reference.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeTestStitch(
        JNIEnv* env, jclass /*clazz*/, jlong handle) {

    auto* s = asSession(handle);
    if (!s) return nullptr;

    auto snap = s->buffer->snapshot();
    if (snap.empty()) {
        LOGI("testStitch: buffer empty");
        return nullptr;
    }

    BufferedFrame reference = std::move(snap.back());
    snap.pop_back();

    if (reference.anyPersonMask.empty() ||
        cv::countNonZero(reference.anyPersonMask) == 0) {
        LOGI("testStitch: no people detected in reference frame");
        return nullptr;
    }

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto aligned = alignToReference(reference, snap);
    auto t1 = clk::now();
    auto result = stitch(reference, reference.anyPersonMask, aligned);
    auto t2 = clk::now();

    auto alignMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto stitchMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    LOGI("testStitch: align=%lldms stitch=%lldms fill=%.2f",
         (long long)alignMs, (long long)stitchMs, result.realFillRatio);

    if (result.image.empty()) return nullptr;

    // OpenCV's JPEG encoder wants BGR; our images are RGB.
    cv::Mat bgr;
    cv::cvtColor(result.image, bgr, cv::COLOR_RGB2BGR);

    std::vector<uchar> jpeg;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 90};
    if (!cv::imencode(".jpg", bgr, jpeg, params)) {
        LOGI("testStitch: imencode failed");
        return nullptr;
    }

    jbyteArray out = env->NewByteArray(static_cast<jsize>(jpeg.size()));
    if (!out) return nullptr;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(jpeg.size()),
                            reinterpret_cast<const jbyte*>(jpeg.data()));
    return out;
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
                         jobjectArray outArr) {
    auto snap = s->buffer->snapshot();
    if (snap.empty()) {
        LOGI("stitchForInpaint: buffer empty");
        return false;
    }
    BufferedFrame reference = std::move(snap.back());
    snap.pop_back();

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

    // No pre-trim: let the aligner decide per-frame via translation/reproj
    // gates. Old frames from when the person wasn't in frame are the most
    // valuable — they have clean background. Trimming them defeats the purpose
    // for "hold steady → step out → step back → capture" workflows.
    LOGI("stitchForInpaint: using all %zu candidate frames", snap.size());

    // Two-stage dilation:
    // 1. Uniform 30px elliptical dilation (silhouette undercoverage: hair, jaw, hands)
    // 2. Directional downward expansion by 60px (cast shadow on floor/wall below person)
    cv::Mat dilatedHole;
    {
        cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(61, 61));
        cv::dilate(removeHole, dilatedHole, kern);

        // Shadow expansion: extend downward only using a tall narrow kernel.
        // Shadows fall below the person's feet; sideways expansion is unnecessary
        // and would clip nearby KEEP persons.
        cv::Mat shadowKern = cv::Mat::zeros(61, 3, CV_8UC1);
        // Fill only the bottom half of the kernel → dilates downward
        shadowKern(cv::Rect(0, 30, 3, 31)).setTo(255);
        cv::dilate(dilatedHole, dilatedHole, shadowKern);
    }

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto aligned = alignToReference(reference, snap);
    auto t1 = clk::now();
    auto result = stitch(reference, dilatedHole, aligned);
    auto t2 = clk::now();

    // Stage 5 (lighting / exposure drift): shift the patched region's mean
    // toward the reference's boundary mean. Fixes the "blueish tint" caused
    // by AWB drift across the buffer window. No-op if the rings are too thin.
    // Wider band (24px) to average over more surrounding context — reduces
    // bias from localized shadow gradients at the hole boundary.
    matchStitchToReferenceTint(result.image, reference.image, dilatedHole, 24);

    // Sharpness recovery: unsharp mask inside the hole. The temporal median
    // averages sub-pixel-misaligned samples, which softens edges and texture.
    // This restores most of it without amplifying noise. Applied BEFORE the
    // boundary feather so the feather's smooth transition isn't ringed by
    // sharpening.
    unsharpMaskInHole(result.image, dilatedHole);

    // Noise floor matching: the median also averages out sensor noise, so the
    // patched area reads as artificially smooth on featureless surfaces (white
    // walls / sky). Re-inject gaussian noise at the same std-dev the reference
    // has just outside the hole. Without this, large patches on smooth surfaces
    // look "too clean" and the eye picks them out even when everything else
    // is correct.
    matchNoiseToReferenceSurround(result.image, reference.image, dilatedHole);

    // Stage 6 (seam blending): soft feather across the hole boundary so the
    // stitched→original transition isn't a hard cut. The dilation already
    // pushed the boundary well past the silhouette, so no person leaks in.
    // Wider feather (25px) to smooth the larger dilation boundary.
    featherStitchBoundary(result.image, reference.image, dilatedHole, 25);
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
    jfloatArray jratio    = env->NewFloatArray(1);
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
    jfloat ratio = result.realFillRatio;
    env->SetFloatArrayRegion(jratio, 0, 1, &ratio);

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
        jobjectArray outArr) {
    auto* s = asSession(handle);
    if (!s) return JNI_FALSE;
    if (!outArr || env->GetArrayLength(outArr) < 6) return JNI_FALSE;
    if (!removeMask || maskW <= 0 || maskH <= 0) return JNI_FALSE;
    if (env->GetArrayLength(removeMask) < (jsize)((size_t)maskW * maskH)) return JNI_FALSE;
    return runStitchForInpaint(s, env, removeMask, maskW, maskH, outArr) ? JNI_TRUE : JNI_FALSE;
}

/**
 * OpenCV cv::inpaint fallback used if LaMa fails. Uses Telea's method — fast,
 * acceptable for small holes, visibly bad for big ones. The whole point is to
 * always produce *some* result so the app never ships an empty pixel region.
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
        cv::inpaint(bgr, maskMat, inpainted, 3.0, cv::INPAINT_TELEA);
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
 * Phase 11 / Level 2 — high-resolution output.
 *
 * Composites the low-resolution stitched patch into a full-sensor high-res
 * Bitmap captured by CameraX ImageCapture. The bulk of the output is the
 * sharp high-res sensor pixels; only the patched region is the upscaled
 * low-res stitched content, blended at the dilated-hole boundary with a
 * gaussian feather.
 *
 *   highResBitmap : Android Bitmap (ARGB_8888 expected) from ImageCapture →
 *                   captureHighRes(), already rotated upright.
 *   lowResStitched: w*h*3 RGB bytes — the stitcher's output (post matchTint /
 *                   unsharp / noise / feather).
 *   lowResMask    : w*h bytes — the dilated REMOVE hole used by the stitcher
 *                   (stitch.fullHoleMask in Kotlin).
 *   lowW, lowH    : low-res dimensions.
 *
 * Returns a JPEG-encoded ByteArray ready for GallerySaver, or null on any
 * failure (bitmap lock, decode, encode). All cv::Mats are stack-scoped via
 * RAII — no leaks if any intermediate step fails.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_one5_personremoval_core_NativeSession_nativeCompositeHighRes(
        JNIEnv* env, jclass /*clazz*/,
        jobject highResBitmap,
        jbyteArray lowResStitched, jbyteArray lowResMask,
        jint lowW, jint lowH) {

    if (lowW <= 0 || lowH <= 0) {
        LOGW("compositeHighRes: invalid low-res dims %dx%d", lowW, lowH);
        return nullptr;
    }
    const size_t lowRgbLen = static_cast<size_t>(lowW) * lowH * 3;
    const size_t lowMaskLen = static_cast<size_t>(lowW) * lowH;
    if (env->GetArrayLength(lowResStitched) < (jsize)lowRgbLen ||
        env->GetArrayLength(lowResMask) < (jsize)lowMaskLen) {
        LOGW("compositeHighRes: low-res input size mismatch");
        return nullptr;
    }

    // ---- 1. Lock the Android Bitmap and wrap its pixels as a cv::Mat. -----
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, highResBitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGW("compositeHighRes: AndroidBitmap_getInfo failed");
        return nullptr;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGW("compositeHighRes: bitmap not RGBA_8888 (got %d)", (int)info.format);
        return nullptr;
    }
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, highResBitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
        !pixels) {
        LOGW("compositeHighRes: lockPixels failed");
        return nullptr;
    }

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    // Bitmap storage is row-major RGBA. Wrap directly — no copy.
    cv::Mat highResRgba((int)info.height, (int)info.width, CV_8UC4, pixels,
                        (size_t)info.stride);
    cv::Mat highRes;  // BGR for OpenCV downstream + JPEG encode
    cv::cvtColor(highResRgba, highRes, cv::COLOR_RGBA2BGR);

    // We can unlock as soon as we've copied OUT of the Bitmap's memory.
    AndroidBitmap_unlockPixels(env, highResBitmap);

    // ---- 2. Wrap the low-res inputs. --------------------------------------
    jbyte* lowStPtr = env->GetByteArrayElements(lowResStitched, nullptr);
    jbyte* lowMaskPtr = env->GetByteArrayElements(lowResMask, nullptr);
    if (!lowStPtr || !lowMaskPtr) {
        if (lowStPtr)   env->ReleaseByteArrayElements(lowResStitched, lowStPtr, JNI_ABORT);
        if (lowMaskPtr) env->ReleaseByteArrayElements(lowResMask, lowMaskPtr, JNI_ABORT);
        return nullptr;
    }
    cv::Mat lowRgb(lowH, lowW, CV_8UC3, lowStPtr);
    cv::Mat lowMask(lowH, lowW, CV_8UC1, lowMaskPtr);

    // Convert low-res to BGR to match high-res's color order.
    cv::Mat lowBgr;
    cv::cvtColor(lowRgb, lowBgr, cv::COLOR_RGB2BGR);

    // ---- 3. Upscale low-res stitched + mask to high-res dimensions. -------
    cv::Mat upStitched, upMask;
    cv::resize(lowBgr,  upStitched, highRes.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(lowMask, upMask,     highRes.size(), 0, 0, cv::INTER_LINEAR);
    cv::threshold(upMask, upMask, 127, 255, cv::THRESH_BINARY);

    env->ReleaseByteArrayElements(lowResStitched, lowStPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(lowResMask, lowMaskPtr, JNI_ABORT);

    // ---- 4. Composite with a feathered boundary. --------------------------
    // The feather kernel scales with the high-res dimensions: at 12 MP the
    // patch is ~2000 px across, so a 31-px feather is proportionally similar
    // to the 15-px feather we use at low-res (240-wide). On smaller images
    // it floors to a sensible minimum.
    const int featherRadius = std::max(15, std::min(highRes.cols, highRes.rows) / 64);
    const int featherKsz = 2 * featherRadius + 1;
    cv::Mat alpha8;
    cv::GaussianBlur(upMask, alpha8, cv::Size(featherKsz, featherKsz), 0);
    cv::Mat alphaF;
    alpha8.convertTo(alphaF, CV_32F, 1.0 / 255.0);

    cv::Mat highRes32, up32;
    highRes.convertTo(highRes32, CV_32FC3);
    upStitched.convertTo(up32, CV_32FC3);
    std::vector<cv::Mat> ach{alphaF, alphaF, alphaF};
    cv::Mat alpha3;
    cv::merge(ach, alpha3);
    cv::Mat blended32 = up32.mul(alpha3) + highRes32.mul(cv::Scalar::all(1) - alpha3);

    cv::Mat result;
    blended32.convertTo(result, CV_8UC3);

    // ---- 5. JPEG-encode. --------------------------------------------------
    std::vector<uchar> jpegOut;
    std::vector<int> jpegParams{cv::IMWRITE_JPEG_QUALITY, 95};
    if (!cv::imencode(".jpg", result, jpegOut, jpegParams)) {
        LOGW("compositeHighRes: imencode failed");
        return nullptr;
    }

    auto t1 = clk::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    LOGI("compositeHighRes: high=%dx%d low=%dx%d feather=%d %lldms jpegBytes=%zu",
         highRes.cols, highRes.rows, lowW, lowH, featherRadius,
         (long long)ms, jpegOut.size());

    jbyteArray result_out = env->NewByteArray(static_cast<jsize>(jpegOut.size()));
    if (!result_out) return nullptr;
    env->SetByteArrayRegion(result_out, 0, static_cast<jsize>(jpegOut.size()),
                            reinterpret_cast<const jbyte*>(jpegOut.data()));
    return result_out;
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

}