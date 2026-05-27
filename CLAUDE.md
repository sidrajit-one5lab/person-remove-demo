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

The app requires `android:largeHeap="true"` — the ring buffer holds up to 45 frames at ~4 MB each (~180 MB peak).

## Project Structure

Two Gradle modules: `:app` (the application) and `:opencv` (local OpenCV SDK, consumed as a prefab library by the native code).

### Kotlin Layer (`app/src/main/java/com/one5/personremoval/`)

- **`PersonRemovalApp`** — Application subclass. Owns the app-scoped `MlRepository` and eagerly preloads + warms LaMa at process start.
- **`MainActivity`** — Single activity, sets up Compose with `CaptureScreen`.
- **`ui/CaptureScreen`** — Compose UI. Renders camera preview, bbox overlay, capture button. Reads state from `CaptureViewModel`; user input (taps, capture, flip) flows back to the VM.
- **`ui/CaptureViewModel`** — Owns the analyzer coroutine (collects CameraX frames, runs YOLO + tracker, pushes into native ring buffer) and `onCapture()` (stillness gate → freeze → capture pipeline → save). Holds gyro integrator and exposure-relock logic.
- **`ui/MaskOverlay`** — Draws bbox overlays (green=KEEP, red=REMOVE) on the camera preview.
- **`camera/CameraManager`** — CameraX wrapper. Exposes analyzer frames as a `Flow<ImageProxy>` (mailbox pattern, not buffered flow). Also binds `ImageCapture` for full-sensor-resolution photos (currently bound but unused — output ships at analyzer resolution).
- **`ml/YoloSegmenter`** — YOLOv8n-seg inference via LiteRT (TensorFlow Lite). Produces per-person bounding boxes + binary instance masks. Tries NNAPI → GPU → CPU delegates in order. Default `confidenceThreshold = 0.25` (recall-biased; false positives are cheap because the user just doesn't tap them).
- **`ml/LamaInpainter`** — LaMa inpainting via ONNX Runtime. Fixed 512×512 working size. Crops are letterboxed (grey-padded to square) before resize so non-square gaps keep aspect ratio. Two strategies: `inpaint()` (whole-crop, single LaMa call) and `inpaintGaps()` (per-component crops with adaptive context padding, sorted by area, capped at 3 LaMa calls; crops > 768 px route to `inpaintLargeCrop` for 2×2 tiled inference). Final paste uses a feathered alpha (`computeFeatherAlpha`) so the LaMa↔stitcher seam fades smoothly. `warmUp()` runs a 64×64 dummy inference at construction to amortize ORT JIT cost. Extracts the ~100 MB `.onnx` asset to internal storage on first launch.
- **`core/NativeSession`** — JNI facade for the C++ engine. Manages a native ring buffer handle with a `ReentrantLock` to serialize JNI calls against `close()`. Key operations: `pushFrame`, `stitchForInpaint`, `opencvInpaint`, `harmonizeLamaRegion`, `neutralizePreLama`, `finalize`, `encodeJpeg`.
- **`core/CaptureUseCase`** — Routes the per-capture pipeline: build REMOVE mask → `stitchForInpaint` → choose inpainter by `actualFill` → optional pre-LaMa neutralize → `harmonizeLamaRegion` → `finalize`. Returns `PipelineResult` (RGB + hole mask) + status string.
- **`core/Tracker`** — Greedy IoU tracker that assigns stable IDs across frames and persists KEEP/REMOVE state per track.
- **`core/Models`** — Data classes: `Person`, `PersonState` (KEEP/REMOVE), `DetectionResult`.
- **`core/MlRepository`** — App-scoped singleton holder for expensive ML models (currently just LaMa). Mutex-guarded lazy init with sticky failure flag. Triggers `LamaInpainter.warmUp()` after construction.
- **`core/GallerySaver`** — Saves JPEG to device gallery via MediaStore (API 29+) or legacy file path (API 26–28).
- **`sensors/GyroIntegrator`** — Integrates gyro samples into a rotation matrix. Provides per-frame rotation for the native aligner's gyro-fallback path and powers the stillness gate + AE-relock detection in `CaptureViewModel`.

### Native C++ Layer (`app/src/main/cpp/`)

Built with CMake 3.22.1, C++17, linked against the local OpenCV module and `jnigraphics`.

- **`jni_bridge.cpp`** — JNI entry points. Manages `Session` structs (ring buffer + static-scene skip logic with thumbnail diffing and heartbeat timer). Implements `stitchForInpaint`, `opencvInpaint`, `finalize`, `harmonizeLamaRegion`, `neutralizePreLama`, `encodeJpeg`, plus bitmap pack/unpack helpers.
- **`buffer.cpp/h`** — `RingBuffer` — fixed-capacity circular buffer storing RGBA frames + per-person masks + per-frame rotation matrices.
- **`aligner.cpp/h`** — Two-tier feature alignment. Tier 1: ORB at ~270 px detect scale. Tier 2 (need-based fallback): AKAZE at ~120 px on any frame where ORB failed, bounded by a per-stitch atomic budget (`kMaxAkazeCalls = 8`). Tier 0 (last resort): gyro-derived homography when both feature tiers produce no inliers. Phase-1 (parallel): per-frame feature detect/match/RANSAC/ECC. Phase-2 (serial): full-res `warpPerspective`.
- **`stitcher.cpp/h`** — Temporal median stitching: for each pixel in the REMOVE mask, picks the median value across aligned frames where that pixel was not masked as a person. Dual-bucket quality weighting (`kHiQualityCutoff = 0.4f`) prefers high-quality samples when available.
- **`blender.cpp/h`** — Poisson seamless cloning (via `cv::seamlessClone`) + CLAHE + Gaussian feathering for final polish.

### ML Model Assets (`app/src/main/assets/`)

- `yolov8n-seg.tflite` — YOLOv8-nano segmentation model (FP16). Export: `yolo export model=yolov8n-seg.pt format=tflite half=True imgsz=640`
- `lama.onnx` — LaMa inpainting model (FP32, fixed 512×512 input). Both `.tflite` and `.onnx` are listed in `noCompress` in build.gradle.kts.

## Capture Pipeline Flow

1. **Detection loop** (continuous, on `Dispatchers.Default`): CameraX analyzer → RGBA extraction (reused single buffer) → `YoloSegmenter.detect` → `Tracker.update` → `nativeSession.pushFrame(rgba, masks, timestamp, gyroRotation, trackIds)`.
2. **User taps** a person → `Tracker.toggle(trackId)` flips KEEP ↔ REMOVE.
3. **Capture button** (`CaptureViewModel.onCapture`):
   1. Stillness gate (poll gyro until `ω < 0.15 rad/s`, capped at 500 ms).
   2. `nativeSession.frozen = true`.
   3. `CaptureUseCase.execute(persons, states, w, h)`:
      - Build REMOVE mask = binary YOLO mask OR'd across REMOVE persons, with bbox-floor for any person > 40% of frame area.
      - `stitchForInpaint` — native align + temporal median stitch. Returns `rgb`, `unfilledMask`, `fullHoleMask`, `fillRatio`, `noSampleRatio`.
      - Recompute `actualFill` from `unfilledMask` (ghost detector may have flagged extra pixels post-`fillRatio`).
      - Route by `actualFill`:
        - `actualFill ≥ 0.85` (or zero unfilled): OpenCV Telea / skip.
        - `actualFill < 0.85`: LaMa via `inpaintGaps`. If `actualFill < 0.40` (`NEUTRALIZE_FILL_THRESHOLD`), the unfilled hole is first painted with the ring-mean color (`nativeSession.neutralizePreLama`) so LaMa doesn't echo the still-visible subject silhouette as a ghost.
      - `harmonizeLamaRegion` shifts the LaMa-filled region's mean RGB toward the surrounding ring (±15 clamp).
      - `finalize` — Poisson seamless clone of `filledRgb` into `stitch.rgb` across the full hole + CLAHE + feather.
   4. `encodeJpeg(result.rgb, ...)` → save single JPEG via `GallerySaver`. Status string drives a toast hint when fill is low or noSampleRatio is high.

## Key Constraints

- **LaMa routing**: `actualFill < 0.85` → LaMa, else Telea (`CaptureUseCase.classicalThreshold`). Higher threshold biases toward Telea, lower toward LaMa.
- **Pre-LaMa neutralization**: `actualFill < 0.40` (`NEUTRALIZE_FILL_THRESHOLD`) routes through `nativeNeutralizePreLama` before LaMa, replacing person-tinted hole pixels with the ring mean so the model produces a smooth color blob instead of a recognizable silhouette.
- **Output resolution**: ships at analyzer resolution (~1080×1920). `ImageCapture` is bound but its bitmap is currently unused — the `compositeHighRes` JNI exists but isn't wired into the save path.
- Ring buffer uses a static-scene skip (thumbnail diff) with a 1 s heartbeat to avoid filling with redundant frames while still capturing clean-background samples during "user steps out of frame" periods.
- `NativeSession.opLock` uses `tryLock()` in `pushFrame` to drop frames rather than block the analyzer when a capture is in progress.
- AE/AWB lock is applied after first frames stabilize; `CaptureViewModel` flushes the ring buffer on lock so only exposure-consistent frames feed the stitcher. Large gyro rotation (> 30°) triggers a re-lock + buffer flush.
- AKAZE alignment fires only when ORB fails on a given frame and the per-stitch budget (`kMaxAkazeCalls = 8`) hasn't been exhausted — replaces the older "newest 5 frames only" recency window.
