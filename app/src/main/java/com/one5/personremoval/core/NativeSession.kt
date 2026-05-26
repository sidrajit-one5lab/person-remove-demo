package com.one5.personremoval.core

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Kotlin facade for the native C++ engine. One session owns one ring buffer in native memory.
 *
 * Step 6b: ring buffer push/snapshot/clear. Capture/align/stitch follow in later steps.
 *
 * Always call [close] (use it as a Compose DisposableEffect resource). Forgetting will leak
 * the native session and its ~250 MB of frame buffers.
 */
class NativeSession(capacity: Int = 80) : AutoCloseable {

    // Serializes JNI calls against close(). Without this, a capture coroutine on
    // Dispatchers.Default can be mid-stitch while the composable is disposed on
    // Main and nativeDestroy is invoked — the in-flight JNI thread would then
    // dereference a freed Session*. close() acquires the lock and waits for any
    // in-flight op to finish before destroying the native session.
    private val opLock = ReentrantLock()

    /** Pointer to the native Session struct. 0 means closed. Mutated under [opLock]. */
    private var handle: Long = nativeCreate(capacity)

    @Volatile var frozen: Boolean = false

    /**
     * Push one camera frame into the ring buffer.
     *
     * @param rotation9 optional 9-element row-major device rotation matrix
     *                  (from GyroIntegrator) captured at frame time. When
     *                  provided, the native aligner can use it as a
     *                  feature-free fallback on texture-poor scenes.
     */
    fun pushFrame(
        rgba: ByteArray,
        width: Int,
        height: Int,
        personMasks: Array<ByteArray>,
        timestampMs: Long,
        rotation9: FloatArray? = null,
        trackIds: IntArray? = null
    ) {
        if (frozen) return
        if (!opLock.tryLock()) return
        try {
            if (handle != 0L) nativePushFrame(
                handle, rgba, width, height, personMasks, timestampMs, rotation9, trackIds
            )
        } finally {
            opLock.unlock()
        }
    }

    fun bufferSize(): Int = opLock.withLock {
        if (handle == 0L) 0 else nativeBufferSize(handle)
    }

    fun clearBuffer(): Unit = opLock.withLock {
        if (handle != 0L) nativeClear(handle)
    }

    /**
     * Phase 7 verification: aligns every buffered frame (except the newest, used as reference)
     * to the reference frame's viewpoint. Returns the number that aligned successfully.
     * Result and timing are logged via LOG_TAG="PRNative".
     */
    fun testAlignment(): Int = opLock.withLock {
        if (handle == 0L) -1 else nativeTestAlignment(handle)
    }

    /**
     * Phase 9 entry: runs stitching and returns everything the Kotlin side needs
     * to drive both the inpaint stage and the final seamless-clone blend.
     *  - `rgb`           : stitched RGB (reference with hole filled by real samples)
     *  - `originalRgb`   : raw reference RGB before stitching — used by finalize()
     *                      so seamless cloning runs across the FULL stitched boundary
     *                      (not just the LaMa boundary)
     *  - `unfilledMask`  : per-pixel byte; non-zero = the inpainter should fill
     *  - `fullHoleMask`  : the dilated hole used by the stitcher (full REMOVE region)
     *  - `width/height`  : image size
     *  - `fillRatio`     : fraction of the hole filled with real (non-AI) pixels, 0..1
     *
     * Returns null if there's nothing to stitch (empty buffer, empty REMOVE mask,
     * or mask dimensions don't match the latest buffered frame).
     */
    data class StitchOutput(
        val rgb: ByteArray,
        val originalRgb: ByteArray,
        val unfilledMask: ByteArray,
        val fullHoleMask: ByteArray,
        val width: Int,
        val height: Int,
        val fillRatio: Float,
        // Fraction of hole pixels with ZERO clean-bg samples in the buffer.
        // Distinct from (1 - fillRatio): that also counts pixels with 1-2
        // marginal samples (which inpaint can still leverage). A high
        // noSampleRatio means subject occluded those pixels for the entire
        // buffer window — the UI should prompt "ask subject to step aside
        // briefly" because no amount of waiting will recover them.
        val noSampleRatio: Float
    )

    /**
     * @param removeMask byte[width*height], non-zero where the user wants pixels removed
     */
    fun stitchForInpaint(
        removeMask: ByteArray,
        maskWidth: Int,
        maskHeight: Int,
        removeTrackIds: IntArray = intArrayOf()
    ): StitchOutput? = opLock.withLock {
        if (handle == 0L) return@withLock null
        // outArr: [0]=rgb, [1]=originalRgb, [2]=unfilledMask, [3]=fullHoleMask,
        //         [4]=int[]{w,h}, [5]=float[]{ratio}
        val outArr = arrayOfNulls<Any>(6)
        @Suppress("UNCHECKED_CAST")
        val ok = nativeStitchForInpaint(
            handle, removeMask, maskWidth, maskHeight, removeTrackIds, outArr as Array<Any?>
        )
        if (!ok) return@withLock null
        val rgb         = outArr[0] as? ByteArray ?: return@withLock null
        val originalRgb = outArr[1] as? ByteArray ?: return@withLock null
        val unfilled    = outArr[2] as? ByteArray ?: return@withLock null
        val fullHole    = outArr[3] as? ByteArray ?: return@withLock null
        val dims        = outArr[4] as? IntArray ?: return@withLock null
        val ratio       = outArr[5] as? FloatArray ?: return@withLock null
        if (dims.size < 2 || ratio.size < 2) return@withLock null
        StitchOutput(
            rgb, originalRgb, unfilled, fullHole,
            dims[0], dims[1],
            fillRatio = ratio[0],
            noSampleRatio = ratio[1]
        )
    }

    // opencvInpaint / finalize / encodeJpeg deliberately stay outside opLock:
    // they don't touch the Session* (no handle arg), so they can't race with
    // nativeDestroy.

    /**
     * Classical OpenCV inpaint (Telea). Fallback used when LaMa fails.
     * Safe to call on any thread. Returns null on failure.
     */
    fun opencvInpaint(rgb: ByteArray, mask: ByteArray, width: Int, height: Int): ByteArray? {
        return nativeOpencvInpaint(rgb, mask, width, height)
    }

    /**
     * Phase 10 — final polish: Poisson seamless cloning + CLAHE + feather.
     * `referenceRgb` is the original capture-moment frame; `filledRgb` is the
     * post-stitch (and post-inpaint) result. Returns the polished RGB bytes.
     */
    fun finalize(
        referenceRgb: ByteArray,
        filledRgb: ByteArray,
        holeMask: ByteArray,
        width: Int,
        height: Int
    ): ByteArray? = nativeFinalize(referenceRgb, filledRgb, holeMask, width, height)

    /** Encode raw RGB bytes to a JPEG byte array via OpenCV. */
    fun encodeJpeg(rgb: ByteArray, width: Int, height: Int, quality: Int = 95): ByteArray? =
        nativeEncodeJpeg(rgb, width, height, quality)

    /**
     * Drop the alpha channel from a CameraX RGBA_8888 byte array. Pure
     * pixel reshape; uses OpenCV cv::cvtColor (SIMD on arm64).
     */
    fun rgbaToRgb(rgba: ByteArray, width: Int, height: Int): ByteArray? =
        nativeRgbaToRgb(rgba, width, height)

    /**
     * In-place tint harmonization for the LaMa-filled region. Shifts the
     * mean RGB of the masked pixels toward the 1-px ring of real pixels
     * just outside, clamped to ±15 per channel. Modifies [rgb] in place.
     */
    fun harmonizeLamaRegion(
        rgb: ByteArray,
        mask: ByteArray,
        width: Int,
        height: Int
    ): Unit = nativeHarmonizeLamaRegion(rgb, mask, width, height)

    /**
     * Composite the low-res stitched patch into a high-res camera capture.
     * The low-res patch is bicubic-upscaled to the bitmap's dimensions; the
     * low-res hole mask is bilinear-upscaled and re-thresholded; the two are
     * alpha-blended with a gaussian-feathered boundary so the patch sharpness
     * discontinuity isn't a hard edge.
     *
     * Pixels outside the upscaled hole come straight from [hiResBitmap]
     * (full sensor quality). Pixels inside come from [stitchedLoRgb]
     * upscaled (the patched region — softer than the rest, but plausible).
     *
     * The bitmap must be in [Bitmap.Config.ARGB_8888] format (what CameraX
     * ImageCapture decodes to). Native code locks the bitmap pixels directly
     * (via android/bitmap.h jnigraphics) — no Kotlin-side pixel copy.
     *
     * Returns the JPEG-encoded composited result, or null on size mismatch /
     * locking failure / OpenCV failure.
     */
    fun compositeHighRes(
        hiResBitmap: android.graphics.Bitmap,
        stitchedLoRgb: ByteArray, holeLo: ByteArray, loW: Int, loH: Int
    ): ByteArray? =
        nativeCompositeHighRes(hiResBitmap, stitchedLoRgb, holeLo, loW, loH)

    override fun close() = opLock.withLock {
        if (handle != 0L) {
            val h = handle
            handle = 0L
            nativeDestroy(h)
        }
    }

    companion object {
        init { System.loadLibrary("personremoval") }

        /** Returns the linked OpenCV version. Used by the sanity-check log on startup. */
        fun openCvVersion(): String = nativeOpenCvVersion()

        /**
         * Pack an RGB byte array into an existing ARGB_8888 Bitmap (alpha
         * set to 0xFF). Returns true on success. Fast path replaces the
         * Bitmap.setPixels(IntArray) per-pixel pack used by LamaInpainter
         * and MattingRefiner.
         */
        fun fillBitmapFromRgb(
            rgb: ByteArray, w: Int, h: Int, bmp: android.graphics.Bitmap
        ): Boolean = nativeFillBitmapFromRgb(rgb, w, h, bmp)

        /**
         * Unpack RGB bytes from an ARGB_8888 Bitmap. Returns null on
         * failure (wrong format, lock failure, OOM); caller should fall
         * back to Bitmap.getPixels().
         */
        fun readRgbFromBitmap(bmp: android.graphics.Bitmap): ByteArray? =
            nativeReadRgbFromBitmap(bmp)

        @JvmStatic private external fun nativeOpenCvVersion(): String
        @JvmStatic private external fun nativeFillBitmapFromRgb(
            rgb: ByteArray, w: Int, h: Int, bmp: android.graphics.Bitmap
        ): Boolean
        @JvmStatic private external fun nativeReadRgbFromBitmap(
            bmp: android.graphics.Bitmap
        ): ByteArray?
        @JvmStatic private external fun nativeCreate(capacity: Int): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
        @JvmStatic private external fun nativeBufferSize(handle: Long): Int
        @JvmStatic private external fun nativeClear(handle: Long)
        @JvmStatic private external fun nativePushFrame(
            handle: Long,
            rgba: ByteArray,
            width: Int,
            height: Int,
            personMasks: Array<ByteArray>,
            timestampMs: Long,
            rotation9: FloatArray?,
            trackIds: IntArray?
        )
        @JvmStatic private external fun nativeTestAlignment(handle: Long): Int
        @JvmStatic private external fun nativeStitchForInpaint(
            handle: Long,
            removeMask: ByteArray, maskW: Int, maskH: Int,
            removeTrackIds: IntArray,
            out: Array<Any?>
        ): Boolean
        @JvmStatic private external fun nativeOpencvInpaint(
            rgb: ByteArray, mask: ByteArray, width: Int, height: Int
        ): ByteArray?
        @JvmStatic private external fun nativeFinalize(
            referenceRgb: ByteArray, filledRgb: ByteArray, holeMask: ByteArray,
            width: Int, height: Int
        ): ByteArray?
        @JvmStatic private external fun nativeEncodeJpeg(
            rgb: ByteArray, width: Int, height: Int, quality: Int
        ): ByteArray?
        @JvmStatic private external fun nativeCompositeHighRes(
            hiResBitmap: android.graphics.Bitmap,
            stitchedLoRgb: ByteArray, holeLo: ByteArray, loW: Int, loH: Int
        ): ByteArray?
        @JvmStatic private external fun nativeRgbaToRgb(
            rgba: ByteArray, width: Int, height: Int
        ): ByteArray?
        @JvmStatic private external fun nativeHarmonizeLamaRegion(
            rgb: ByteArray, mask: ByteArray, width: Int, height: Int
        )
    }
}
