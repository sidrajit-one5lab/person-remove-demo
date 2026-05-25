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
import kotlinx.coroutines.async
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
    val nativeSession = NativeSession(capacity = 30)
    private val mlRepository = (application as PersonRemovalApp).mlRepository
    private val captureUseCase = CaptureUseCase(nativeSession)

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

        viewModelScope.launch(Dispatchers.Default) {
            cameraManager.frames.collectLatest { proxy ->
                try {
                    val w = proxy.width
                    val h = proxy.height
                    val plane = proxy.planes[0]
                    val buf = plane.buffer
                    val rowStride = plane.rowStride
                    val rgba = ByteArray(w * h * 4)
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
                    val raw = segmenter.detect(rgba, w, h)
                    val nowMs = System.currentTimeMillis()
                    val (tracked, states) = tracker.update(raw.persons, nowMs)

                    val masks: Array<ByteArray> = Array(tracked.size) { i -> tracked[i].mask }
                    nativeSession.pushFrame(rgba, w, h, masks, nowMs)

                    _bufSize.value = nativeSession.bufferSize()
                    _detection.value = raw.copy(persons = tracked)
                    _personStates.value = states
                } finally {
                    proxy.close()
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
        val fx = (offsetX * sx).toInt().coerceIn(0, d.sourceWidth - 1)
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

    fun onTestStitch() {
        if (_isProcessing.value) return
        _isProcessing.value = true
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val d = _detection.value
                val states = tracker.getStates()
                val (result, status) = if (d != null) {
                    captureUseCase.execute(d.persons, states, d.sourceWidth, d.sourceHeight)
                } else {
                    null to "no detection yet"
                }
                _lastResultText.value = status
                _stitchPreview.value = result?.let {
                    rgbBytesToBitmap(it.rgb, it.width, it.height)
                }
            } finally {
                _isProcessing.value = false
            }
        }
    }

    fun onCapture() {
        if (_isProcessing.value) return
        _isProcessing.value = true
        viewModelScope.launch(Dispatchers.Default) {
            try {
                delay(1500L)
                val hiResDeferred = async { cameraManager.captureHighRes() }

                val d = _detection.value
                val states = tracker.getStates()
                val (result, status) = if (d != null) {
                    captureUseCase.execute(d.persons, states, d.sourceWidth, d.sourceHeight)
                } else {
                    null to "no detection yet"
                }
                if (result == null) {
                    hiResDeferred.cancel()
                    _toastEvents.tryEmit(ToastEvent(status))
                    _lastResultText.value = status
                    return@launch
                }

                val hiResBitmap = hiResDeferred.await()

                val (jpeg, finalStatus) =
                    if (hiResBitmap != null) {
                        val tComp = System.currentTimeMillis()
                        val composited = nativeSession.compositeHighRes(
                            hiResBitmap,
                            result.rgb, result.holeMask, result.width, result.height
                        )
                        val compMs = System.currentTimeMillis() - tComp
                        if (composited != null) {
                            Log.i("PRPipeline",
                                "hi-res composite ${hiResBitmap.width}x${hiResBitmap.height} in ${compMs}ms")
                            composited to "$status  +hi-res"
                        } else {
                            Log.w("PRPipeline", "hi-res composite returned null, falling back to low-res")
                            val lo = nativeSession.encodeJpeg(result.rgb, result.width, result.height)
                            lo to status
                        }
                    } else {
                        val lo = nativeSession.encodeJpeg(result.rgb, result.width, result.height)
                        lo to status
                    }
                hiResBitmap?.recycle()

                if (jpeg == null) {
                    _toastEvents.tryEmit(ToastEvent("JPEG encode failed"))
                    return@launch
                }
                val uri = GallerySaver.save(getApplication(), jpeg)
                val bmp = android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)

                val isTricky = "tricky scene" in finalStatus
                val toastMsg = when {
                    uri == null -> "Save failed"
                    isTricky -> "Saved — tricky scene, try stepping out of frame and retake"
                    else -> "Saved to Gallery"
                }
                _stitchPreview.value = bmp
                _lastResultText.value = finalStatus
                _toastEvents.tryEmit(ToastEvent(toastMsg, long = isTricky))
            } finally {
                _isProcessing.value = false
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        cameraManager.shutdown()
        segmenter.close()
        nativeSession.close()
    }
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