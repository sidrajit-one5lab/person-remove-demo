package com.one5.personremoval.ui

import android.app.Application
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.one5.personremoval.PersonRemovalApp
import com.one5.personremoval.camera.CameraManager
import com.one5.personremoval.core.CaptureUseCase
import com.one5.personremoval.core.DetectionResult
import com.one5.personremoval.core.GallerySaver
import com.one5.personremoval.core.NativeSession
import com.one5.personremoval.core.PersonState
import com.one5.personremoval.core.Tracker
import com.one5.personremoval.ml.YoloSegmenter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch

data class ToastEvent(val message: String, val long: Boolean = false)

class CaptureViewModel(application: Application) : AndroidViewModel(application) {

    val cameraManager = CameraManager(application)
    private val segmenter = YoloSegmenter(application)
    val tracker = Tracker()
    val nativeSession = NativeSession(capacity = 45)
    private val mlRepository = (application as PersonRemovalApp).mlRepository
    // Gyroscope integrator: provides a per-frame device rotation matrix so
    // the native aligner can fall back to a gyro-derived homography when
    // ORB/AKAZE feature matching produces no inliers (texture-poor scenes:
    // walls, sky, ceiling). No-ops on devices without a gyroscope.
    private val gyroIntegrator: com.one5.personremoval.sensors.GyroIntegrator =
        com.one5.personremoval.sensors.GyroIntegrator(application)
    private val captureUseCase = CaptureUseCase(nativeSession)

    // Reusable analyzer RGBA buffer. pushFrame copies into the native ring
    // buffer synchronously, so a single slot is safe to overwrite next frame.
    // Avoids the per-frame ~8 MB allocation that dominated analyzer-thread
    // GC pressure on low-end devices.
    private var frameSlot: ByteArray? = null
    private var frameSlotBytes: Int = 0

    // AE re-lock on large camera motion. When the user physically moves
    // (walks to different position), exposure/WB lock becomes stale. Track
    // the gyro rotation at lock time; if angular displacement exceeds the
    // threshold, trigger relockExposure() so AE reconverges for the new
    // viewpoint. The onExposureLocked callback flushes the buffer, so only
    // exposure-consistent frames from the new position enter the stitch.
    @Volatile private var lockedRotation: FloatArray? = null
    @Volatile private var lastRelockMs: Long = 0L

    private val _hasPermission = MutableStateFlow(false)
    val hasPermission: StateFlow<Boolean> = _hasPermission.asStateFlow()

    private val _detection = MutableStateFlow<DetectionResult?>(null)
    val detection: StateFlow<DetectionResult?> = _detection.asStateFlow()

    private val _personStates = MutableStateFlow<Map<Int, PersonState>>(emptyMap())
    val personStates: StateFlow<Map<Int, PersonState>> = _personStates.asStateFlow()

    private val _stitchPreview = MutableStateFlow<Bitmap?>(null)
    val stitchPreview: StateFlow<Bitmap?> = _stitchPreview.asStateFlow()

    private val _lastResultText = MutableStateFlow<String?>(null)
    val lastResultText: StateFlow<String?> = _lastResultText.asStateFlow()

    private val _cameraSelector = MutableStateFlow(CameraSelector.DEFAULT_BACK_CAMERA)
    val cameraSelector: StateFlow<CameraSelector> = _cameraSelector.asStateFlow()

    private val _bufSize = MutableStateFlow(0)
    val bufSize: StateFlow<Int> = _bufSize.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private val _toastEvents = MutableSharedFlow<ToastEvent>(extraBufferCapacity = 1)
    val toastEvents: SharedFlow<ToastEvent> = _toastEvents.asSharedFlow()

    init {
        viewModelScope.launch {
            captureUseCase.lamaInpainter = mlRepository.lama()
        }
        // Start gyro accumulation. The rotation matrix resets here so all
        // subsequent buffer frames are expressed in a common reference
        // frame (the moment the VM was constructed).
        gyroIntegrator.start()

        // After CameraManager applies AE/AWB lock, flush the ring buffer.
        // Pre-lock frames captured varying exposure/WB; mixing them into the
        // temporal-median stitch shifts the patched region's tint away from
        // the post-lock reference frame, producing the visible "bright box"
        // halo on multi-region holes. Discarding pre-lock frames means only
        // exposure-consistent frames feed the stitcher.
        cameraManager.onExposureLocked = {
            nativeSession.clearBuffer()
            _bufSize.value = 0
            lockedRotation = if (gyroIntegrator.isAvailable) gyroIntegrator.snapshot() else null
            lastRelockMs = System.currentTimeMillis()
        }

        viewModelScope.launch(Dispatchers.Default) {
            cameraManager.frames.collectLatest { proxy ->
                proxy.use { proxy ->
                    if (_isProcessing.value) {
                        return@collectLatest
                    }
                    val w = proxy.width
                    val h = proxy.height
                    val plane = proxy.planes[0]
                    val buf = plane.buffer
                    val rowStride = plane.rowStride
                    val needed = w * h * 4
                    // Lazy (re)allocation: only on first frame, or when the
                    // analyzer resolution actually changes (e.g. camera flip
                    // returns a different ImageAnalysis profile). Steady-state
                    // path hits neither branch.
                    if (frameSlotBytes != needed) {
                        frameSlot = ByteArray(needed)
                        frameSlotBytes = needed
                    }
                    val rgba = frameSlot!!
                    if (rowStride == w * 4) {
                        buf.get(rgba)
                    } else {
                        val rowBytes = w * 4
                        val tmp = ByteArray(rowStride)
                        for (y in 0 until h) {
                            buf.position(y * rowStride)
                            buf.get(tmp, 0, rowStride)
                            System.arraycopy(tmp, 0, rgba, y * rowBytes, rowBytes)
                        }
                    }
                    val raw = try {
                        segmenter.detect(rgba, w, h)
                    } catch (t: Throwable) {
                        // YOLO inference failure on a single frame must not
                        // tear down the analyzer pipeline. Skip this frame,
                        // log, and continue with the next one.
                        Log.w(TAG, "YOLO detect threw, skipping frame", t)
                        return@collectLatest
                    }
                    val nowMs = System.currentTimeMillis()
                    val (tracked, states) = tracker.update(raw.persons, nowMs)

                    val masks: Array<ByteArray> = Array(tracked.size) { i -> tracked[i].mask }
                    val trackIds = IntArray(tracked.size) { i -> tracked[i].trackId }
                    val rot = if (gyroIntegrator.isAvailable) gyroIntegrator.snapshot() else null
                    nativeSession.pushFrame(rgba, w, h, masks, nowMs, rot, trackIds)

                    _bufSize.value = nativeSession.bufferSize()
                    _detection.value = raw.copy(persons = tracked)
                    _personStates.value = states
                    checkMotionRelock(rot)
                }
            }
        }

        viewModelScope.launch {
            _cameraSelector.drop(1).collect { selector ->
                cameraManager.rebind(selector)
                nativeSession.clearBuffer()
                tracker.reset()
                _stitchPreview.value = null
                _personStates.value = emptyMap()
            }
        }
    }

    fun checkPermission(context: android.content.Context) {
        _hasPermission.value = ContextCompat.checkSelfPermission(
            context, android.Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    fun onPermissionResult(granted: Boolean) {
        _hasPermission.value = granted
    }

    fun bindCamera(owner: LifecycleOwner, previewView: PreviewView) {
        cameraManager.bind(owner, previewView, _cameraSelector.value)
    }

    fun onTap(offsetX: Float, offsetY: Float, canvasWidth: Int, canvasHeight: Int) {
        val d = _detection.value ?: return
        if (d.sourceWidth == 0 || d.sourceHeight == 0) return

        val sx = d.sourceWidth.toFloat() / canvasWidth
        val sy = d.sourceHeight.toFloat() / canvasHeight
        val isFront = _cameraSelector.value == CameraSelector.DEFAULT_FRONT_CAMERA
        val rawX = (offsetX * sx).toInt().coerceIn(0, d.sourceWidth - 1)
        val fx = if (isFront) d.sourceWidth - 1 - rawX else rawX
        val fy = (offsetY * sy).toInt().coerceIn(0, d.sourceHeight - 1)
        val idx = fy * d.sourceWidth + fx

        var best = -1
        var bestArea = Int.MAX_VALUE
        for (p in d.persons) {
            if (p.maskWidth != d.sourceWidth || p.maskHeight != d.sourceHeight) continue
            if (idx >= p.mask.size) continue
            if (p.mask[idx].toInt() == 0) continue
            var area = 0
            for (b in p.mask) if (b.toInt() != 0) area++
            if (area < bestArea) {
                bestArea = area; best = p.trackId
            }
        }
        if (best >= 0) tracker.toggle(best)
    }

    fun onFlipCamera() {
        if (_isProcessing.value) return
        _cameraSelector.value = if (_cameraSelector.value == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
    }


    fun onCapture() {
        if (_isProcessing.value) return
        _isProcessing.value = true
        viewModelScope.launch(Dispatchers.Default) {
            try {
                // Stillness gate. Replaces the old fixed 1.5 s post-tap
                // delay with an adaptive wait: poll the gyro and freeze
                // the buffer as soon as the phone stops moving, capped
                // at STILLNESS_MAX_WAIT_MS. The cap covers devices with
                // no gyro (recentAngularVelocityRadPerSec returns +∞) and
                // shaky-handed captures that never settle. On still phones
                // the gate completes in one poll cycle (~25 ms) — the user
                // sees an immediate shutter.
                val stillnessStart = System.currentTimeMillis()
                while (true) {
                    val omega = gyroIntegrator.recentAngularVelocityRadPerSec()
                    val elapsed = System.currentTimeMillis() - stillnessStart
                    if (omega < STILLNESS_THRESHOLD_RAD_PER_SEC) {
                        Log.i(TAG, "stillness ok after ${elapsed}ms (ω=$omega rad/s)")
                        break
                    }
                    if (elapsed >= STILLNESS_MAX_WAIT_MS) {
                        Log.i(TAG, "stillness timeout at ${elapsed}ms (ω=$omega rad/s) — capturing anyway")
                        break
                    }
                    delay(STILLNESS_POLL_MS)
                }

                // freeze before launching the pipeline. New pushFrame() calls bail
                // early on `frozen == true`, and stitchForInpaint() blocks on the
                // same opLock that pushFrame holds, so any in-flight push completes
                // before stitching reads the buffer. No additional wait needed.
                nativeSession.frozen = true

                val d = _detection.value
                val states = tracker.getStates()
                val (result, status) = if (d != null) {
                    captureUseCase.execute(d.persons, states, d.sourceWidth, d.sourceHeight)
                } else {
                    null to "no detection yet"
                }
                if (result == null) {
                    _toastEvents.tryEmit(ToastEvent(status))
                    _lastResultText.value = status
                    return@launch
                }

                val jpeg = nativeSession.encodeJpeg(result.rgb, result.width, result.height)
                if (jpeg == null) {
                    _toastEvents.tryEmit(ToastEvent("JPEG encode failed"))
                    return@launch
                }
                val uri = GallerySaver.save(getApplication(), jpeg)
                val bmp = android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)

                val isTricky = "tricky scene" in status
                val keepSteady = "keep phone steady" in status
                val toastMsg = when {
                    uri == null -> "Save failed"
                    keepSteady -> "Saved — keep phone steady, ask subject to step aside"
                    isTricky -> "Saved — tricky scene, try stepping out of frame and retake"
                    else -> "Saved to Gallery"
                }
                _stitchPreview.value = bmp
                _lastResultText.value = status
                _toastEvents.tryEmit(ToastEvent(toastMsg, long = isTricky || keepSteady))
            } finally {
                nativeSession.frozen = false
                _isProcessing.value = false
            }
        }
    }

    private fun checkMotionRelock(currentRotation: FloatArray?) {
        if (nativeSession.frozen) return
        val locked = lockedRotation ?: return
        val current = currentRotation ?: return
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastRelockMs < 3000L) return
        val angle = angularDisplacement(locked, current)
        if (angle > RELOCK_ANGLE_DEG) {
            lastRelockMs = nowMs
            cameraManager.relockExposure()
        }
    }

    override fun onCleared() {
        super.onCleared()
        gyroIntegrator.stop()
        cameraManager.shutdown()
        segmenter.close()
        nativeSession.close()
    }

    companion object {
        private const val TAG = "CaptureViewModel"
        // Stillness gate parameters. 0.15 rad/s ≈ 8.6°/s — typical "holding
        // phone reasonably still" angular speed. Tighter (e.g. 0.05) waits
        // out micro-tremor at the cost of frequent timeouts in normal use;
        // looser would let blur sneak in. 500 ms hard cap so the shutter
        // never feels broken.
        private const val STILLNESS_THRESHOLD_RAD_PER_SEC = 0.15f
        private const val STILLNESS_MAX_WAIT_MS = 500L
        private const val STILLNESS_POLL_MS = 25L
    }
}

private const val RELOCK_ANGLE_DEG = 30f

private fun angularDisplacement(r1: FloatArray, r2: FloatArray): Float {
    // trace(R2 * R1^T) = Frobenius inner product of R1 and R2
    var trace = 0f
    for (i in 0 until 9) trace += r1[i] * r2[i]
    val cosAngle = ((trace - 1f) / 2f).coerceIn(-1f, 1f)
    return Math.toDegrees(kotlin.math.acos(cosAngle.toDouble())).toFloat()
}

private fun rgbBytesToBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap {
    val bmp = createBitmap(w, h)
    val pixels = IntArray(w * h)
    for (i in 0 until w * h) {
        val r = rgb[i * 3].toInt() and 0xFF
        val g = rgb[i * 3 + 1].toInt() and 0xFF
        val b = rgb[i * 3 + 2].toInt() and 0xFF
        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    bmp.setPixels(pixels, 0, w, 0, 0, w, h)
    return bmp
}