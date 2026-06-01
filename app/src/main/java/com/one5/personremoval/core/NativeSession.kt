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
     * Phase 9 entry: runs stitching and returns everything the Kotlin side needs
     * to drive both the inpaint stage and the final seamless-clone blend.
     *  - `rgb`           : stitched RGB (reference with hole filled by real samples)
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
        removeTrackIds: IntArray = intArrayOf(),
        keepTrackIds: IntArray = intArrayOf()
    ): StitchOutput? = opLock.withLock {
        if (handle == 0L) return@withLock null
        // outArr: [0]=rgb, [1]=unfilledMask, [2]=fullHoleMask,
        //         [3]=int[]{w,h}, [4]=float[]{ratio, noSampleRatio}
        val outArr = arrayOfNulls<Any>(5)
        @Suppress("UNCHECKED_CAST")
        val ok = nativeStitchForInpaint(
            handle, removeMask, maskWidth, maskHeight,
            removeTrackIds, keepTrackIds, outArr as Array<Any?>
        )
        if (!ok) return@withLock null
        val rgb         = outArr[0] as? ByteArray ?: return@withLock null
        val unfilled    = outArr[1] as? ByteArray ?: return@withLock null
        val fullHole    = outArr[2] as? ByteArray ?: return@withLock null
        val dims        = outArr[3] as? IntArray ?: return@withLock null
        val ratio       = outArr[4] as? FloatArray ?: return@withLock null
        if (dims.size < 2 || ratio.size < 2) return@withLock null
        StitchOutput(
            rgb, unfilled, fullHole,
            dims[0], dims[1],
            fillRatio = ratio[0],
            noSampleRatio = ratio[1]
        )
    }

    // opencvInpaint / finalize / encodeJpeg deliberately stay outside opLock:
    // they don't touch the Session* (no handle arg), so they can't race with
    // nativeDestroy.

    /**
     * Classical OpenCV inpaint (Navier-Stokes, cv::INPAINT_NS). Fallback when
     * MI-GAN is unavailable or fails. Safe to call on any thread. Null on failure.
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
     * In-place grain match for the inpainted region. Injects sensor noise
     * matching the band just outside the hole into the filled pixels so the
     * smooth inpaint/Telea patch sits in the same grain as the surround.
     * Self-skips when the surround is too thin/clean to measure (no-op on
     * synthetic inputs). Call AFTER [finalize] so the Poisson clone doesn't
     * smooth the injected noise out. Modifies [rgb] in place.
     *
     * (Native fn keeps the legacy "Lama" name — it's engine-agnostic grain.)
     */
    fun textureLamaRegion(
        rgb: ByteArray,
        mask: ByteArray,
        width: Int,
        height: Int
    ): Unit = nativeTextureLamaRegion(rgb, mask, width, height)

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
         * set to 0xFF). Returns true on success. Fast path replaces a
         * Bitmap.setPixels(IntArray) per-pixel pack.
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
        @JvmStatic private external fun nativeStitchForInpaint(
            handle: Long,
            removeMask: ByteArray, maskW: Int, maskH: Int,
            removeTrackIds: IntArray,
            keepTrackIds: IntArray,
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
        @JvmStatic private external fun nativeTextureLamaRegion(
            rgb: ByteArray, mask: ByteArray, width: Int, height: Int
        )
        @JvmStatic external fun nativeDilateMask(
            mask: ByteArray, w: Int, h: Int, radiusPx: Int
        ): ByteArray
    }
}
