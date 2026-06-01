package com.one5.personremoval.core

import android.content.Context
import android.util.Log
import com.one5.personremoval.ml.MiGanInpainter
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * App-scoped holder for heavy ML models.
 *
 * Constructed once by [com.one5.personremoval.PersonRemovalApp] at process start
 * and outlives any single screen / composable / orientation change. UI code reads
 * from here instead of constructing models inline, so the user doesn't pay the
 * load cost on first capture.
 *
 * Currently holds [MiGanInpainter] — the on-device removal/inpainting model
 * (~28 MB ONNX, single forward pass). [YoloSegmenter] stays in screen scope
 * ([com.one5.personremoval.ui.CaptureScreen]) because its lifecycle is tightly
 * coupled to the analyzer pipeline there.
 *
 * Concurrency: a [Mutex] guards init. First caller wins; subsequent callers see
 * the cached instance. If construction throws (OOM, missing asset), we log and
 * return null — callers must handle null and fall back to the classical OpenCV
 * (Telea/NS) inpaint path.
 */
class MlRepository(private val appContext: Context) {

    @Volatile private var miganInstance: MiGanInpainter? = null
    // Sticky "we already tried" flag, so a failed init isn't retried (and
    // re-logged) by every caller — they see the cached result instead.
    @Volatile private var miganAttempted: Boolean = false
    private val miganMutex = Mutex()

    /**
     * Returns the singleton [MiGanInpainter], constructing on first call.
     * Returns null if construction failed (cached). Callers must handle null
     * and fall back to the OpenCV inpaint path.
     */
    suspend fun migan(): MiGanInpainter? = miganMutex.withLock {
        if (miganAttempted) return@withLock miganInstance
        miganAttempted = true
        try {
            MiGanInpainter(appContext).also {
                miganInstance = it
                // Dummy inference so ORT JIT / delegate init happens off the
                // user-visible capture path.
                it.warmUp()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "MI-GAN init failed; capture will fall back to OpenCV", t)
            null
        }
    }

    /**
     * Eagerly construct all models. Called by [PersonRemovalApp.onCreate] in a
     * background coroutine so the first user capture doesn't pay the load cost.
     */
    suspend fun preload() {
        migan()
    }

    /**
     * Release all loaded models. Optional — the OS reclaims native memory on
     * process kill anyway, and Application doesn't have a reliable teardown
     * hook on Android. Kept for completeness / testing.
     */
    fun closeAll() {
        miganInstance?.close()
        miganInstance = null
        miganAttempted = false
    }

    private companion object {
        const val TAG = "MlRepository"
    }
}