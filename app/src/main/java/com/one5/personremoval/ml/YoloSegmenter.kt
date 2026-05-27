package com.one5.personremoval.ml

import android.content.Context
import android.content.res.AssetManager
import android.graphics.RectF
import android.util.Log
import com.one5.personremoval.core.DetectionResult
import com.one5.personremoval.core.Person
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8-seg inference for the "person" COCO class.
 *
 * Expects an Ultralytics-exported FP16 TFLite model at assets/yolov8n-seg.tflite.
 * Input shape: [1, 640, 640, 3], float32, normalized to [0, 1].
 * Outputs (NHWC export):
 *   output0 = [1, 8400, 116]   → 4 bbox (cx,cy,w,h in 640 space) + 80 class scores + 32 mask coefficients
 *   output1 = [1, 160, 160, 32] → mask prototypes
 *
 * Final per-detection mask = sigmoid( prototypes · coefficients ), thresholded.
 */
class YoloSegmenter(
    context: Context,
    private val modelAsset: String = "yolov8n-seg.tflite",
    // 0.25 (rather than the more common 0.4) so we catch weaker detections of
    // partially-occluded body parts (a person behind a counter, hands-only on a
    // workbench, etc.). The COCO `person` class has very low false-positive
    // density at this level, so the cost is just a few stray chair/mannequin
    // boxes that the user can ignore.
    private val confidenceThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.5f,
    private val personClassIndex: Int = 0
) : AutoCloseable {

    companion object {
        init { System.loadLibrary("personremoval") }

        @JvmStatic
        private external fun nativeFillInputBuffer(
            rgba: ByteArray, srcW: Int, srcH: Int,
            dstW: Int, dstH: Int, padX: Int, padY: Int,
            inputBuffer: ByteBuffer, isFloat32: Boolean
        )

        private const val TAG = "YoloSegmenter"
        private const val INPUT_SIZE = 640
        private const val NUM_CLASSES = 80
        private const val NUM_MASK_COEFFS = 32
        private const val MASK_PROTO_SIZE = 160
        private const val NUM_ANCHORS = 8400
        // Adaptive threshold range: low-confidence detections (0.3) → 0.20
        // (aggressive, catches soft edges), high-confidence (0.9) → 0.38
        // (tight, less background puff).
        private const val MASK_THRESH_MIN = 0.12f
        private const val MASK_THRESH_MAX = 0.30f
        private const val BBOX_MASK_PAD = 0.05f
        private const val EDGE_SNAP_FRAC = 0.03f
    }

    private val interpreter: Interpreter
    private val ownedDelegate: Delegate?

    // True if the .tflite was exported with int8/uint8 input quantization
    // (e.g. `yolo export ... int8=True`). When true, the model expects 1 byte
    // per channel (raw uint8 in [0, 255]) instead of 4 bytes per channel
    // (float32 in [0, 1]). We detect this at init and size [inputBuffer]
    // accordingly, then dispatch in [fillInputBuffer].
    private val isQuantizedInput: Boolean

    // Reusable input buffer. Sized 1 byte/channel for int8 models, 4 bytes/channel
    // for fp32 — quantized buffer is 4× smaller and skips the /255 normalization.
    private val inputBuffer: ByteBuffer

    // [1, 116, 8400] — Ultralytics exports features-first.
    // Output is fp32 even for int8 input models (Ultralytics keeps output fp32
    // for downstream simplicity). If a future export emits uint8 outputs we'd
    // need a dequantization path here — not the case today.
    private val output0 = Array(1) { Array(4 + NUM_CLASSES + NUM_MASK_COEFFS) { FloatArray(NUM_ANCHORS) } }
    // [1, 160, 160, 32]
    private val output1 =
        Array(1) { Array(MASK_PROTO_SIZE) { Array(MASK_PROTO_SIZE) { FloatArray(NUM_MASK_COEFFS) } } }

    init {
        val model = loadModelOrThrow(context.assets, modelAsset)

        // Try delegates in order: NNAPI → GPU → CPU. The Interpreter constructor itself can
        // throw if a delegate accepts construction but rejects the model graph, so we wrap
        // each attempt and fall through on any failure.
        val (it, dlg) = tryBuildInterpreter(model)
        interpreter = it
        ownedDelegate = dlg

        // Detect the model's input dtype. Ultralytics fp16/fp32 exports give
        // FLOAT32 input; int8 exports give UINT8 input. We support both — they
        // differ only in pre-processing (normalize vs raw) and buffer size.
        val inputTensor = interpreter.getInputTensor(0)
        isQuantizedInput = when (inputTensor.dataType()) {
            DataType.FLOAT32 -> false
            DataType.UINT8, DataType.INT8 -> true
            else -> throw IllegalStateException(
                "Unsupported YOLO input dtype: ${inputTensor.dataType()}"
            )
        }
        val bytesPerChannel = if (isQuantizedInput) 1 else 4
        inputBuffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * bytesPerChannel)
            .order(ByteOrder.nativeOrder())

        Log.i(TAG, "YOLO loaded; input=${inputTensor.shape().toList()} " +
                "dtype=${inputTensor.dataType()} quantized=$isQuantizedInput " +
                "out0=${interpreter.getOutputTensor(0).shape().toList()} " +
                "out1=${interpreter.getOutputTensor(1).shape().toList()}")
    }

    /**
     * Runs detection on an RGBA byte buffer of size [width*height*4].
     * Returns persons with masks in the source frame's coordinate space.
     */
    fun detect(rgba: ByteArray, width: Int, height: Int): DetectionResult {
        val t0 = System.currentTimeMillis()

        // 1. Letterbox: scale the longest side to 640, pad shorter to center.
        val scale = INPUT_SIZE.toFloat() / max(width, height)
        val newW = (width * scale).toInt()
        val newH = (height * scale).toInt()
        val padX = (INPUT_SIZE - newW) / 2
        val padY = (INPUT_SIZE - newH) / 2

        fillInputBuffer(rgba, width, height, newW, newH, padX, padY)

        // 2. Inference
        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(inputBuffer),
            mapOf(0 to output0, 1 to output1)
        )

        // 3. Decode detections (filter by person class).
        // Output layout is [116][8400]: feature index then anchor index.
        val feat = output0[0]   // feat[f][i] -> value of feature f for anchor i
        val protos = output1[0]

        data class Det(
            val cx: Float, val cy: Float, val w: Float, val h: Float,
            val conf: Float, val coeffs: FloatArray
        ) {
            override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (javaClass != other?.javaClass) return false

                other as Det

                if (cx != other.cx) return false
                if (cy != other.cy) return false
                if (w != other.w) return false
                if (h != other.h) return false
                if (conf != other.conf) return false
                if (!coeffs.contentEquals(other.coeffs)) return false

                return true
            }

            override fun hashCode(): Int {
                var result = cx.hashCode()
                result = 31 * result + cy.hashCode()
                result = 31 * result + w.hashCode()
                result = 31 * result + h.hashCode()
                result = 31 * result + conf.hashCode()
                result = 31 * result + coeffs.contentHashCode()
                return result
            }
        }

        val candidates = ArrayList<Det>(64)
        val personRow = feat[4 + personClassIndex]
        for (i in 0 until NUM_ANCHORS) {
            val score = personRow[i]
            if (score < confidenceThreshold) continue
            // Check person is the top class for this anchor
            var maxClass = score
            var maxIdx = personClassIndex
            for (c in 0 until NUM_CLASSES) {
                val s = feat[4 + c][i]
                if (s > maxClass) { maxClass = s; maxIdx = c }
            }
            if (maxIdx != personClassIndex) continue

            val coeffs = FloatArray(NUM_MASK_COEFFS)
            for (k in 0 until NUM_MASK_COEFFS) coeffs[k] = feat[4 + NUM_CLASSES + k][i]
            // Ultralytics' TFLite export emits bbox normalized to [0, 1]. Scale up to 640-space.
            candidates.add(
                Det(
                    cx = feat[0][i] * INPUT_SIZE,
                    cy = feat[1][i] * INPUT_SIZE,
                    w = feat[2][i] * INPUT_SIZE,
                    h = feat[3][i] * INPUT_SIZE,
                    conf = score,
                    coeffs = coeffs
                )
            )
        }

        // 4. NMS
        candidates.sortByDescending { it.conf }
        val kept = ArrayList<Det>()
        val suppressed = BooleanArray(candidates.size)
        for (i in candidates.indices) {
            if (suppressed[i]) continue
            kept.add(candidates[i])
            val a = candidates[i]
            for (j in i + 1 until candidates.size) {
                if (suppressed[j]) continue
                val b = candidates[j]
                if (iouXywh(a.cx, a.cy, a.w, a.h, b.cx, b.cy, b.w, b.h) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        // 5. Build per-detection masks in source-image space.
        val persons = ArrayList<Person>(kept.size)
        for (det in kept) {
            // Bbox in 640-space → undo letterbox → source space
            val x1_640 = det.cx - det.w / 2f
            val y1_640 = det.cy - det.h / 2f
            val x2_640 = det.cx + det.w / 2f
            val y2_640 = det.cy + det.h / 2f

            var x1 = ((x1_640 - padX) / scale).coerceIn(0f, width - 1f)
            var y1 = ((y1_640 - padY) / scale).coerceIn(0f, height - 1f)
            var x2 = ((x2_640 - padX) / scale).coerceIn(0f, width - 1f)
            var y2 = ((y2_640 - padY) / scale).coerceIn(0f, height - 1f)

            // Snap bbox to frame edge when close — YOLO often predicts slightly
            // short of the boundary for persons cut off by the frame.
            val edgeX = width * EDGE_SNAP_FRAC
            val edgeY = height * EDGE_SNAP_FRAC
            if (x1 < edgeX) x1 = 0f
            if (y1 < edgeY) y1 = 0f
            if (x2 > width - 1f - edgeX) x2 = width - 1f
            if (y2 > height - 1f - edgeY) y2 = height - 1f

            val bbox = RectF(x1, y1, x2, y2)

            val mask = buildMask(det.coeffs, protos, bbox, width, height, scale, padX, padY, det.conf)
            persons.add(
                Person(
                    trackId = -1,
                    bBox = bbox,
                    mask = mask,
                    maskWidth = width,
                    maskHeight = height,
                    confidence = det.conf
                )
            )
        }

        val ms = System.currentTimeMillis() - t0
        if (persons.isNotEmpty()) {
            val p = persons[0]
            Log.d(TAG,
                "src=${width}x${height} pad=$padX,$padY scale=$scale " +
                "first bbox=(${p.bBox.left.toInt()},${p.bBox.top.toInt()})-" +
                "(${p.bBox.right.toInt()},${p.bBox.bottom.toInt()}) conf=${p.confidence}")
        }
        return DetectionResult(persons, width, height, ms)
    }

    // --------------------------------------------------------------------- //

    private fun fillInputBuffer(
        rgba: ByteArray, srcW: Int, srcH: Int,
        dstW: Int, dstH: Int, padX: Int, padY: Int
    ) {
        inputBuffer.rewind()
        try {
            nativeFillInputBuffer(rgba, srcW, srcH, dstW, dstH, padX, padY,
                inputBuffer, !isQuantizedInput)
        } catch (t: Throwable) {
            Log.w(TAG, "native preprocessing failed, falling back to Kotlin: ${t.message}")
            if (isQuantizedInput) {
                fillInputBufferUint8(rgba, srcW, srcH, dstW, dstH, padX, padY)
            } else {
                fillInputBufferFloat32(rgba, srcW, srcH, dstW, dstH, padX, padY)
            }
        }
        inputBuffer.rewind()
    }

    private fun fillInputBufferFloat32(
        rgba: ByteArray, srcW: Int, srcH: Int,
        dstW: Int, dstH: Int, padX: Int, padY: Int
    ) {
        inputBuffer.rewind()
        val padVal = 114f / 255f
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            inputBuffer.putFloat(padVal); inputBuffer.putFloat(padVal); inputBuffer.putFloat(padVal)
        }
        for (y in 0 until dstH) {
            val srcY = (y.toFloat() * srcH / dstH).toInt().coerceIn(0, srcH - 1)
            val rowOffset = ((padY + y) * INPUT_SIZE + padX) * 3 * 4
            for (x in 0 until dstW) {
                val srcX = (x.toFloat() * srcW / dstW).toInt().coerceIn(0, srcW - 1)
                val srcIdx = (srcY * srcW + srcX) * 4
                val r = (rgba[srcIdx].toInt() and 0xFF) / 255f
                val g = (rgba[srcIdx + 1].toInt() and 0xFF) / 255f
                val b = (rgba[srcIdx + 2].toInt() and 0xFF) / 255f
                val dstPos = rowOffset + x * 3 * 4
                inputBuffer.putFloat(dstPos, r)
                inputBuffer.putFloat(dstPos + 4, g)
                inputBuffer.putFloat(dstPos + 8, b)
            }
        }
        inputBuffer.rewind()
    }

    private fun fillInputBufferUint8(
        rgba: ByteArray, srcW: Int, srcH: Int,
        dstW: Int, dstH: Int, padX: Int, padY: Int
    ) {
        inputBuffer.rewind()
        val padVal: Byte = 114.toByte()
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            inputBuffer.put(padVal); inputBuffer.put(padVal); inputBuffer.put(padVal)
        }
        for (y in 0 until dstH) {
            val srcY = (y.toFloat() * srcH / dstH).toInt().coerceIn(0, srcH - 1)
            val rowOffset = ((padY + y) * INPUT_SIZE + padX) * 3
            for (x in 0 until dstW) {
                val srcX = (x.toFloat() * srcW / dstW).toInt().coerceIn(0, srcW - 1)
                val srcIdx = (srcY * srcW + srcX) * 4
                val dstPos = rowOffset + x * 3
                inputBuffer.put(dstPos,     rgba[srcIdx])
                inputBuffer.put(dstPos + 1, rgba[srcIdx + 1])
                inputBuffer.put(dstPos + 2, rgba[srcIdx + 2])
            }
        }
        inputBuffer.rewind()
    }

    /**
     * Computes mask = sigmoid(prototypes · coeffs), thresholded to binary,
     * then resampled into the source-image space inside the bbox.
     */
    private fun buildMask(
        coeffs: FloatArray,
        protos: Array<Array<FloatArray>>,        // [160][160][32]
        bbox: RectF,
        srcW: Int, srcH: Int,
        scale: Float, padX: Int, padY: Int,
        confidence: Float
    ): ByteArray {
        val protoScale = MASK_PROTO_SIZE.toFloat() / INPUT_SIZE.toFloat()  // 0.25
        val maxP = MASK_PROTO_SIZE - 1

        // Expand sampling region beyond bbox to catch extremities (feet, hands)
        // that the proto-mask covers but the bbox doesn't fully enclose.
        val padW = (bbox.width() * BBOX_MASK_PAD).toInt()
        val padH = (bbox.height() * BBOX_MASK_PAD).toInt()
        val ix1 = (bbox.left.toInt() - padW).coerceAtLeast(0)
        val iy1 = (bbox.top.toInt() - padH).coerceAtLeast(0)
        val ix2 = (bbox.right.toInt() + padW).coerceAtMost(srcW - 1)
        val iy2 = (bbox.bottom.toInt() + padH).coerceAtMost(srcH - 1)

        val py1 = ((iy1 * scale + padY) * protoScale).toInt().coerceIn(0, maxP)
        val py2 = ((iy2 * scale + padY) * protoScale).toInt().coerceIn(0, maxP)
        val px1 = ((ix1 * scale + padX) * protoScale).toInt().coerceIn(0, maxP)
        val px2 = ((ix2 * scale + padX) * protoScale).toInt().coerceIn(0, maxP)

        val protoY1 = (py1 - 1).coerceAtLeast(0)
        val protoY2 = (py2 + 1).coerceAtMost(maxP)
        val protoX1 = (px1 - 1).coerceAtLeast(0)
        val protoX2 = (px2 + 1).coerceAtMost(maxP)

        // Sigmoid only within bbox region of proto space
        val regionW = protoX2 - protoX1 + 1
        val regionH = protoY2 - protoY1 + 1
        val protoMask = FloatArray(regionH * regionW)
        for (y in protoY1..protoY2) {
            val v = protos[y]
            val rowOff = (y - protoY1) * regionW
            for (x in protoX1..protoX2) {
                var sum = 0f
                val proto = v[x]
                for (k in 0 until NUM_MASK_COEFFS) sum += proto[k] * coeffs[k]
                protoMask[rowOff + (x - protoX1)] = sigmoid(sum)
            }
        }

        // Adaptive threshold: loose for low-conf, tight for high-conf
        val maskThreshold = MASK_THRESH_MIN + (MASK_THRESH_MAX - MASK_THRESH_MIN) * confidence

        // Bilinear upsample from proto mask into output within bbox
        val out = ByteArray(srcW * srcH)
        for (y in iy1..iy2) {
            val pfY = (y * scale + padY) * protoScale
            val outRow = y * srcW
            for (x in ix1..ix2) {
                val pfX = (x * scale + padX) * protoScale

                val x0 = pfX.toInt().coerceIn(protoX1, protoX2)
                val y0 = pfY.toInt().coerceIn(protoY1, protoY2)
                val x1 = (x0 + 1).coerceAtMost(protoX2)
                val y1 = (y0 + 1).coerceAtMost(protoY2)

                val fx = pfX - x0
                val fy = pfY - y0

                val rx0 = x0 - protoX1
                val rx1 = x1 - protoX1
                val ry0 = (y0 - protoY1) * regionW
                val ry1 = (y1 - protoY1) * regionW

                val v00 = protoMask[ry0 + rx0]
                val v10 = protoMask[ry0 + rx1]
                val v01 = protoMask[ry1 + rx0]
                val v11 = protoMask[ry1 + rx1]

                val value = v00 * (1 - fx) * (1 - fy) +
                            v10 * fx * (1 - fy) +
                            v01 * (1 - fx) * fy +
                            v11 * fx * fy

                if (value > maskThreshold) {
                    out[outRow + x] = 1
                }
            }
        }
        return out
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + Math.exp(-x.toDouble()).toFloat())

    private fun iouXywh(
        acx: Float, acy: Float, aw: Float, ah: Float,
        bcx: Float, bcy: Float, bw: Float, bh: Float
    ): Float {
        val ax1 = acx - aw / 2f; val ay1 = acy - ah / 2f
        val ax2 = acx + aw / 2f; val ay2 = acy + ah / 2f
        val bx1 = bcx - bw / 2f; val by1 = bcy - bh / 2f
        val bx2 = bcx + bw / 2f; val by2 = bcy + bh / 2f
        val ix = max(0f, min(ax2, bx2) - max(ax1, bx1))
        val iy = max(0f, min(ay2, by2) - max(ay1, by1))
        val inter = ix * iy
        val union = aw * ah + bw * bh - inter
        return if (union <= 0f) 0f else inter / union
    }

    override fun close() {
        interpreter.close()
        when (val d = ownedDelegate) {
            is GpuDelegate -> d.close()
            is NnApiDelegate -> d.close()
            else -> { /* no-op */ }
        }
    }

    private fun tryBuildInterpreter(model: MappedByteBuffer): Pair<Interpreter, Delegate?> {
        // 1. NNAPI. If the Interpreter constructor rejects the model under the
        //    delegate, close the delegate before falling through — otherwise its
        //    native handle leaks per session.
        try {
            val d = NnApiDelegate()
            try {
                val ip = Interpreter(model, Interpreter.Options().addDelegate(d))
                Log.i(TAG, "Using NNAPI delegate")
                return ip to d
            } catch (t: Throwable) {
                d.close()
                Log.w(TAG, "NNAPI failed: ${t.message}; trying GPU")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "NNAPI delegate unavailable: ${t.message}; trying GPU")
        }

        // 2. GPU
        try {
            val d = GpuDelegate()
            try {
                val ip = Interpreter(model, Interpreter.Options().addDelegate(d))
                Log.i(TAG, "Using GPU delegate")
                return ip to d
            } catch (t: Throwable) {
                d.close()
                Log.w(TAG, "GPU failed: ${t.message}; falling back to CPU")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate unavailable: ${t.message}; falling back to CPU")
        }

        // 3. CPU
        val ip = Interpreter(model, Interpreter.Options().setNumThreads(4))
        Log.i(TAG, "Using CPU (4 threads)")
        return ip to null
    }

    private fun loadModelOrThrow(assets: AssetManager, name: String): MappedByteBuffer {
        val fd = try {
            assets.openFd(name)
        } catch (e: java.io.IOException) {
            throw IllegalStateException(
                "YOLOv8-seg model missing at assets/$name.\n" +
                        "Export it with:\n" +
                        "  pip install ultralytics\n" +
                        "  yolo export model=yolov8n-seg.pt format=tflite half=True imgsz=640\n" +
                        "then copy the .tflite into app/src/main/assets/$name", e
            )
        }
        return fd.createInputStream().channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}
