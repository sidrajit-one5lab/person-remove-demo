# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PersonRemoval is an Android app that removes people from photos in real-time. The user points the camera, taps on detected people to mark them for removal, and captures a photo with those people erased. The app builds a temporal buffer of frames, aligns them, stitches clean background from multiple viewpoints, and uses AI inpainting (LaMa) or classical inpainting (OpenCV Telea) to fill any remaining gaps.

## Build & Run

```bash
./gradlew assembleDebug                    # Build debug APK
./gradlew installDebug                     # Build + install on connected device
./gradlew :app:testDebugUnitTest           # Unit tests
./gradlew :app:connectedDebugAndroidTest   # Instrumented tests (needs device/emulator)
```

The project targets **arm64-v8a only** (NDK ABI filter). It will not build native code for x86/x86_64 emulators — use a physical arm64 device or an arm64 emulator image.

The app requires `android:largeHeap="true"` — the ring buffer holds ~30 frames at ~8 MB each (~240 MB total).

## Project Structure

Two Gradle modules: `:app` (the application) and `:opencv` (local OpenCV SDK, consumed as a prefab library by the native code).

### Kotlin Layer (`app/src/main/java/com/one5/personremoval/`)

- **`PersonRemovalApp`** — Application subclass. Owns the app-scoped `MlRepository` and eagerly preloads LaMa at process start.
- **`MainActivity`** — Single activity, sets up Compose with `CaptureScreen`.
- **`ui/CaptureScreen`** — Main composable. Owns the camera preview, detection loop, tap-to-toggle interaction, and the capture pipeline (`runPipeline()`). This is where the stitch → inpaint → finalize → composite → save flow lives.
- **`ui/MaskOverlay`** — Draws bbox overlays (green=KEEP, red=REMOVE) on the camera preview.
- **`camera/CameraManager`** — CameraX wrapper. Exposes analyzer frames as a `Flow<ImageProxy>` (mailbox pattern, not buffered flow). Also binds `ImageCapture` for full-sensor-resolution photos used in the hi-res composite path.
- **`ml/YoloSegmenter`** — YOLOv8n-seg inference via LiteRT (TensorFlow Lite). Produces per-person bounding boxes + binary instance masks. Tries NNAPI → GPU → CPU delegates in order.
- **`ml/LamaInpainter`** — LaMa inpainting via ONNX Runtime. Fixed 512x512 input. Has two strategies: `inpaint()` (whole-frame downsample) and `inpaintGaps()` (per-gap crops with context padding — higher quality, used by default). Extracts the ~100 MB `.onnx` asset to internal storage on first launch.
- **`core/NativeSession`** — JNI facade for the C++ engine. Manages a native ring buffer handle with a `ReentrantLock` to serialize JNI calls against `close()`. Key operations: `pushFrame`, `stitchForInpaint`, `finalize`, `compositeHighRes`.
- **`core/Tracker`** — Greedy IoU tracker that assigns stable IDs across frames and persists KEEP/REMOVE state per track.
- **`core/Models`** — Data classes: `Person`, `PersonState` (KEEP/REMOVE), `DetectionResult`.
- **`core/MlRepository`** — App-scoped singleton holder for expensive ML models (currently just LaMa). Mutex-guarded lazy init with sticky failure flag.
- **`core/GallerySaver`** — Saves JPEG to device gallery via MediaStore (API 29+) or legacy file path (API 26-28).

### Native C++ Layer (`app/src/main/cpp/`)

Built with CMake 3.22.1, C++17, linked against the local OpenCV module and `jnigraphics`.

- **`jni_bridge.cpp`** — JNI entry points. Manages `Session` structs (ring buffer + static-scene skip logic with thumbnail diffing and heartbeat timer).
- **`buffer.cpp/h`** — `RingBuffer` — fixed-capacity circular buffer storing RGBA frames + per-person masks.
- **`aligner.cpp/h`** — ORB feature detection + homography estimation to align buffered frames to a reference viewpoint. Auto-downsamples to ~270px wide for feature detection.
- **`stitcher.cpp/h`** — Temporal median stitching: for each pixel in the REMOVE mask, picks the median value across aligned frames where that pixel was not masked as a person.
- **`blender.cpp/h`** — Poisson seamless cloning (via `cv::seamlessClone`) + CLAHE + Gaussian feathering for final polish.

### ML Model Assets (`app/src/main/assets/`)

- `yolov8n-seg.tflite` — YOLOv8-nano segmentation model (FP16). Export: `yolo export model=yolov8n-seg.pt format=tflite half=True imgsz=640`
- `lama.onnx` — LaMa inpainting model (FP32, fixed 512x512 input). Both `.tflite` and `.onnx` are listed in `noCompress` in build.gradle.kts.

## Capture Pipeline Flow

1. **Detection loop** (continuous): CameraX analyzer → RGBA extraction → YoloSegmenter → Tracker → push frame+masks into NativeSession ring buffer
2. **User taps** a person → Tracker toggles that trackId to REMOVE
3. **Capture button** → 1.5s delay (post-tap buffer continuation) → parallel hi-res ImageCapture + `runPipeline()`:
   - Build REMOVE mask (per-pixel mask + optional bbox floor for large persons >25% frame area)
   - `stitchForInpaint()` — native alignment + temporal median stitch
   - Route by fill ratio: full=skip, high=OpenCV Telea, low=LaMa then Telea fallback
   - `finalize()` — Poisson seamless clone + CLAHE + feather on unfilled region only
   - `compositeHighRes()` — bicubic-upscale low-res patch into full-sensor photo
   - JPEG encode → save to gallery

## Key Constraints

- LaMa is currently disabled (threshold set to 0.95 effectively routes everything to OpenCV Telea) because fp32 inference takes 22-32s and produces worse results than Telea on large holes. Re-enable only with a faster/better model.
- Analyzer runs at ~1080x1920 resolution. The hi-res composite upscales the patched region into the full sensor capture (~2460x3280).
- The ring buffer uses a static-scene skip (thumbnail diff) with a 1s heartbeat to avoid filling with redundant frames while still capturing clean-background samples during "user steps out of frame" periods.
- `NativeSession.opLock` uses `tryLock()` in `pushFrame` to drop frames rather than block the analyzer when a capture is in progress.