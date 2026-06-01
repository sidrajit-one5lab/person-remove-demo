package com.one5.personremoval

import android.app.Application
import android.util.Log
import com.one5.personremoval.core.MlRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Application entry point. Owns the app-scoped [MlRepository] and kicks off
 * eager model preloading on process start so the user doesn't pay the
 * MI-GAN load cost on their first capture.
 *
 * Wired into [AndroidManifest.xml] via `android:name=".PersonRemovalApp"`.
 *
 * Lifecycle notes:
 *  - Application.onCreate runs once per process, before any Activity / Composable.
 *  - The preload runs in [appScope], a SupervisorJob-backed scope so a single
 *    model failure doesn't cancel the others.
 *  - There's no reliable Application.onDestroy on Android; native model memory
 *    is reclaimed when the OS kills the process. That's fine for our use case.
 */
class PersonRemovalApp : Application() {

    /** App-wide ML model repository. Read by Composables via the Application context. */
    val mlRepository: MlRepository by lazy { MlRepository(this) }

    /**
     * Coroutine scope tied to the application lifetime. Survives every Activity,
     * Fragment, and Composable in the process. Use for "load once at start"
     * background work like model preloading.
     */
    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onCreate() {
        super.onCreate()
        logAssetAvailability()
        // Preload heavy ML models in the background. The user-visible UI (camera
        // preview) can come up immediately while this runs. Any individual model
        // failure is logged inside the repository and doesn't bubble up — the
        // capture pipeline has classical fallbacks for every model.
        appScope.launch {
            try {
                mlRepository.preload()
                Log.i(TAG, "ML model preload complete")
            } catch (t: Throwable) {
                Log.w(TAG, "preload failed", t)
            }
        }
    }

    /**
     * Explicit startup audit of every ML asset bundled in the APK. Surfacing
     * a yes/no per asset at app start makes dormancy visible in logcat.
     */
    private fun logAssetAvailability() {
        val checks = listOf(
            "yolov8n-seg.tflite"     to "YOLO segmenter",
            "migan_pipeline_v2.onnx" to "MI-GAN inpainter"
        )
        for ((file, label) in checks) {
            val present = try { assets.openFd(file).use { true } } catch (_: Throwable) {
                try { assets.open(file).use { true } } catch (_: Throwable) { false }
            }
            Log.i(TAG, "asset $label ($file) available: $present")
        }
    }

    private companion object {
        const val TAG = "PersonRemovalApp"
    }
}
