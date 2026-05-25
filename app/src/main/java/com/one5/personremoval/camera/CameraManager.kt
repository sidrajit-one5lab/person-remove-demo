package com.one5.personremoval.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import android.util.Size
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.transform
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.resume

/**
 * Owns the CameraX pipeline and exposes analysis frames as a SharedFlow<ImageProxy>.
 * Consumers MUST call ImageProxy.close() when done with each frame.
 *
 * The current camera selector can be swapped at runtime via [rebind] (e.g. front <-> back).
 */
class CameraManager(private val context: Context) {

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Single-slot "latest frame" mailbox. The analyzer atomically swaps in the
    // newest proxy and closes whatever was there. The consumer pulls via the
    // signal flow below. This replaces a previous SharedFlow<ImageProxy> with
    // DROP_OLDEST that leaked dropped proxies — every dropped frame closes here.
    private val latest = AtomicReference<ImageProxy?>(null)

    // Wake-up signal. DROP_OLDEST coalesces wakes; each wake pulls whatever is
    // currently in `latest`. If nothing is there (consumer already took it), the
    // transform simply emits nothing.
    private val _signal = MutableSharedFlow<Unit>(
        replay = 0,
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val frames: Flow<ImageProxy> = _signal.transform {
        latest.getAndSet(null)?.let { emit(it) }
    }

    // Remember the lifecycle owner and preview view so [rebind] can swap selectors later.
    private var currentOwner: LifecycleOwner? = null
    private var currentPreviewView: PreviewView? = null
    private var currentSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    // High-resolution ImageCapture use case. Bound alongside Preview + Analysis.
    // The Analysis pipeline keeps running at 240x320 (fast YOLO + small buffer
    // memory); ImageCapture is only invoked when the user hits the shutter and
    // produces a full-sensor JPEG that we composite the patched region into.
    private var imageCapture: ImageCapture? = null

    fun bind(
        owner: LifecycleOwner,
        previewView: PreviewView,
        selector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    ) {
        currentOwner = owner
        currentPreviewView = previewView
        currentSelector = selector
        doBind()
    }

    /** Switch to a different camera selector (e.g. front <-> back). Safe to call any time after [bind]. */
    fun rebind(selector: CameraSelector) {
        currentSelector = selector
        doBind()
    }

    /** The selector currently being used. */
    fun currentSelector(): CameraSelector = currentSelector

    private fun doBind() {
        val owner = currentOwner ?: return
        val previewView = currentPreviewView ?: return
        val selector = currentSelector

        // Drop any frame buffered from a previous binding — it's from a different
        // camera / different bound state and the consumer would just close it.
        latest.getAndSet(null)?.close()

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Analyzer resolution at ~FHD. The high-res ImageCapture composite
            // upscales the analyzer-resolution patch to ~2460×3280 (full
            // sensor). At 720-wide analyzer that's ~3.4× upscale → visibly
            // soft patch on smooth surfaces. Bumping the analyzer to 1080-
            // wide drops the upscale factor to ~2.3×, which on smooth content
            // is the difference between "noticeably soft" and "essentially
            // matches the surrounding sensor pixels."
            //
            // Cost: per-frame RGBA buffer grows to ~8 MB. Ring buffer is
            // reduced to 30 frames (in NativeSession constructor) to keep
            // total under ~250 MB — still fits in largeHeap.
            //
            // FALLBACK_HIGHER means devices that don't expose 1080×1920 as
            // an analyzer profile will land at the next larger supported size.
            // Aligner's auto-downsample (detectScale = cols/240) adapts so
            // ORB still operates on ~270-wide gray; no retuning needed.
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1080, 1920), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setResolutionSelector(resolutionSelector)
                // Rotate the analyzer buffer to match the display, so YOLO sees upright people.
                .setOutputImageRotationEnabled(true)
                .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                .build()
                .also { ia ->
                    ia.setAnalyzer(analysisExecutor) { proxy ->
                        // Atomic swap: install the new proxy and close the one being
                        // displaced (if any). Guarantees every analyzer-delivered
                        // ImageProxy is closed exactly once — either here when
                        // replaced, or by the consumer's finally block when taken.
                        latest.getAndSet(proxy)?.close()
                        _signal.tryEmit(Unit)
                    }
                }

            // ImageCapture at full sensor resolution. CAPTURE_MODE_MAXIMIZE_QUALITY
            // trades a little shutter latency for a higher-quality JPEG (multi-
            // frame noise reduction on most devices). Format defaults to JPEG; we
            // decode to Bitmap in captureHighRes().
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                .build()
            imageCapture = capture

            provider.unbindAll()
            try {
                provider.bindToLifecycle(owner, selector, preview, analysis, capture)
            } catch (t: Throwable) {
                // Some devices restrict the combination of bound use cases.
                // Fall back to Preview + Analysis only — captureHighRes() will
                // return null and the pipeline will use the low-res result.
                Log.w(TAG, "bind with ImageCapture failed (${t.message}), retrying without it")
                imageCapture = null
                provider.bindToLifecycle(owner, selector, preview, analysis)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Trigger a one-shot full-resolution photo. Returns a Bitmap in the same
     * orientation as the preview (auto-rotated), or null if ImageCapture isn't
     * bound on this device or the capture failed.
     *
     * Safe to call from any coroutine. Does NOT interact with the analyzer
     * pipeline; analyzer keeps producing frames during the capture call.
     */
    suspend fun captureHighRes(): Bitmap? = suspendCancellableCoroutine { cont ->
        val capture = imageCapture
        if (capture == null) {
            cont.resume(null)
            return@suspendCancellableCoroutine
        }
        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val bmp = jpegProxyToBitmap(image)
                        cont.resume(bmp)
                    } catch (t: Throwable) {
                        Log.w(TAG, "high-res decode failed", t)
                        cont.resume(null)
                    } finally {
                        image.close()
                    }
                }

                override fun onError(e: ImageCaptureException) {
                    Log.w(TAG, "high-res capture error: ${e.message}", e)
                    cont.resume(null)
                }
            }
        )
    }

    /**
     * Decode the JPEG bytes from a CameraX ImageCapture ImageProxy into an
     * upright Bitmap. The proxy's rotationDegrees indicates how much to rotate
     * the JPEG-decoded bitmap so it matches the preview orientation.
     */
    private fun jpegProxyToBitmap(image: ImageProxy): Bitmap? {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        val raw = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        val rot = image.imageInfo.rotationDegrees
        if (rot == 0) return raw
        val m = Matrix().apply { postRotate(rot.toFloat()) }
        val rotated = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
        if (rotated !== raw) raw.recycle()
        return rotated
    }

    private companion object {
        const val TAG = "CameraManager"
    }

    fun shutdown() {
        // Close any frame still parked in the mailbox so its underlying buffer is
        // returned to CameraX before the executor goes away.
        latest.getAndSet(null)?.close()
        analysisExecutor.shutdown()
    }
}
