package com.one5.personremoval.core

import android.content.Context
import android.util.Log
import com.one5.personremoval.ml.LamaInpainter
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
 * Currently holds only [LamaInpainter] because that's the expensive one:
 *  - ~100 MB asset to extract from APK to internal storage on first launch
 *  - ~500 ms ORT session creation
 *  - Held native memory between captures
 *
 * [YoloSegmenter] is left in screen scope ([com.one5.personremoval.ui.CaptureScreen])
 * because its lifecycle is tightly coupled to the analyzer pipeline that lives
 * there. Moving it here is a separate refactor.
 *
 * Concurrency: each model has its own [Mutex]. First caller wins; subsequent
 * callers see the cached instance. If construction throws (OOM, missing asset),
 * we log and return null — callers must handle the null and fall back to a
 * classical path (OpenCV Telea for LaMa).
 */
class MlRepository(private val appContext: Context) {

    @Volatile private var lamaInstance: LamaInpainter? = null
    // Sticky "we already tried" flag. Without this, every caller of lama()
    // after a failure would re-attempt the construction (and pay the same
    // crash + log spam). Once an init attempt completes — success or failure
    // — subsequent callers see the cached result without retrying.
    @Volatile private var lamaAttempted: Boolean = false
    private val lamaMutex = Mutex()

    /**
     * Returns the singleton [LamaInpainter], constructing on first call.
     * Returns null if construction failed (cached) or is in flight elsewhere.
     * Callers must handle null and fall back to OpenCV Telea.
     */
    suspend fun lama(): LamaInpainter? = lamaMutex.withLock {
        if (lamaAttempted) return@withLock lamaInstance
        lamaAttempted = true
        try {
            LamaInpainter(appContext).also {
                lamaInstance = it
                // Run a dummy inference so ORT JIT / delegate init happens
                // off the user-visible capture path. Cheap (~few hundred ms
                // at 64×64) and saves that latency on the first real capture.
                it.warmUp()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "LaMa init failed; capture pipeline will fall back to OpenCV Telea", t)
            null
        }
    }

    /**
     * Eagerly construct all models. Called by [PersonRemovalApp.onCreate] in a
     * background coroutine so the first user capture doesn't pay the load cost.
     */
    suspend fun preload() {
        lama()
    }

    /**
     * Release all loaded models. Optional — the OS reclaims native memory on
     * process kill anyway, and Application doesn't have a reliable teardown
     * hook on Android. Kept for completeness / testing.
     */
    fun closeAll() {
        lamaInstance?.close()
        lamaInstance = null
        lamaAttempted = false   // allow re-construction after explicit close
    }

    private companion object {
        const val TAG = "MlRepository"
    }
}
