package com.one5.personremoval.ui

import android.Manifest
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.Dp
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material3.Icon
import androidx.compose.runtime.remember
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush

// Hide the Capture button until the ring buffer has at least this many frames.
// On fresh open (and right after a camera flip / exposure-relock clear) the
// buffer is empty, so a capture would stitch from nothing and dead-end on the
// generic "hold steady / ask subject to step aside" message. Gating the button
// on buffer fill removes that case entirely — the user can only capture once
// frames actually exist.
private const val CAPTURE_READY_MIN_FRAMES = 1

@Composable
fun CaptureScreen(viewModel: CaptureViewModel = viewModel()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val hasPermission by viewModel.hasPermission.collectAsState()
    val detection by viewModel.detection.collectAsState()
    val stitchPreview by viewModel.stitchPreview.collectAsState()
    val lastResultText by viewModel.lastResultText.collectAsState()
    val cameraSelector by viewModel.cameraSelector.collectAsState()
    val bufSize by viewModel.bufSize.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val personStates by viewModel.personStates.collectAsState()
    val parallaxState by viewModel.parallaxState.collectAsState()

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> viewModel.onPermissionResult(granted) }

    LaunchedEffect(Unit) {
        viewModel.checkPermission(context)
        if (!viewModel.hasPermission.value) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    LaunchedEffect(Unit) {
        viewModel.toastEvents.collect { event ->
            Toast.makeText(
                context, event.message,
                if (event.long) Toast.LENGTH_LONG else Toast.LENGTH_SHORT
            ).show()
        }
    }

    if (!hasPermission) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission required")
        }
        return
    }

    Box(
        Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    viewModel.onTap(offset.x, offset.y, size.width, size.height)
                }
            }
    ) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).also { previewView ->
                    viewModel.bindCamera(lifecycleOwner, previewView)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        detection?.let { d ->
            MaskOverlay(
                persons = d.persons,
                states = personStates,
                frameW = d.sourceWidth,
                frameH = d.sourceHeight,
                isFrontCamera = cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA,
                modifier = Modifier.fillMaxSize()
            )
        }

        detection?.let { d ->
//            val removeCount = personStates.count { it.value == com.one5.personremoval.core.PersonState.REMOVE }
//            val cam = if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) "front" else "back"
            Text(
                text = "${d.inferenceMs} ms   buf: $bufSize/45",
                color = Color.White,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(12.dp)
                    .background(Color.Black.copy(alpha = 0.4f))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            )
        }

        stitchPreview?.let { bmp ->
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "Stitch preview",
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(12.dp)
                    .size(300.dp)
                    .background(Color.Black)
            )
        }

        lastResultText?.let { txt ->
            Text(
                text = txt,
                color = Color.White,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 180.dp, end = 16.dp)
                    .background(Color.Black.copy(alpha = 0.5f))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            )
        }

        // Live parallax coach: nudges the user to pan (reveal real background),
        // confirms when ready, or asks a frame-filling subject to step aside.
        // Replaces the old static "hold steady" guidance, which defeated the
        // multi-frame stitch. Driven by CaptureViewModel.parallaxState.
        if (!isProcessing) {
            ParallaxCoachOverlay(
                state = parallaxState,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 96.dp)
            )
        }

        // Camera-style control bar: a gradient scrim with the steady hint
        // stacked ABOVE a row holding the circular shutter (center) and the
        // flip-camera icon (right) — so the hint never overlaps the shutter.
        // The shutter only appears once the ring buffer has frames (an empty
        // buffer can't produce a usable capture) and fades/scales in.
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        listOf(Color.Transparent, Color.Black.copy(alpha = 0.55f))
                    )
                )
                .padding(top = 20.dp, bottom = 44.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Move slowly to capture the background behind",
                color = Color.White.copy(alpha = 0.7f),
                fontSize = 12.sp
            )

            Spacer(Modifier.height(20.dp))

            Box(modifier = Modifier.fillMaxWidth()) {
                if (bufSize >= CAPTURE_READY_MIN_FRAMES && !isProcessing) {
                    ShutterButton(
                        onClick = { viewModel.onCapture() },
                        modifier = Modifier.align(Alignment.Center)
                    )
                }

                FlipButton(
                    enabled = !isProcessing,
                    onClick = { viewModel.onFlipCamera() },
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .padding(end = 36.dp)
                )
            }
        }

        if (isProcessing) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.45f))
                    .pointerInput(Unit) {},
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    GridLoader(color = Color.White, size = 64.dp)
                }
            }
        }
    }
}

/**
 * 3×3 pulsing-dot loader, ported 1:1 from `assets/grid_anim.svg` (the SVG uses
 * SMIL animation, which Android can't render at runtime — Coil would show only
 * a static frame). Each dot's opacity cycles 1 → 0.2 → 1 over 1s, staggered by
 * the same per-dot begin offsets as the SVG.
 */
@Composable
private fun GridLoader(
    modifier: Modifier = Modifier,
    color: Color = Color.White,
    size: Dp = 64.dp
) {
    // (cx, cy, beginMs) — coordinates in the SVG's 105×105 viewBox.
    val dots = remember {
        listOf(
            Triple(12.5f, 12.5f, 0f),
            Triple(12.5f, 52.5f, 100f),
            Triple(52.5f, 12.5f, 300f),
            Triple(52.5f, 52.5f, 600f),
            Triple(92.5f, 12.5f, 800f),
            Triple(92.5f, 52.5f, 400f),
            Triple(12.5f, 92.5f, 700f),
            Triple(52.5f, 92.5f, 500f),
            Triple(92.5f, 92.5f, 200f),
        )
    }
    val transition = rememberInfiniteTransition(label = "gridLoader")
    val t by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "gridT"
    )
    Canvas(modifier.size(size)) {
        val scale = this.size.minDimension / 105f
        val r = 12.5f * scale
        for ((cx, cy, beginMs) in dots) {
            // Loop-phase for this dot; .mod keeps it in [0,1) for the begin offset.
            val phase = (t - beginMs / 1000f).mod(1f)
            // values="1;.2;1" linear ⇒ triangle wave between 1.0 and 0.2.
            val alpha = 0.2f + 0.8f * kotlin.math.abs(2f * phase - 1f)
            drawCircle(
                color = color.copy(alpha = alpha),
                radius = r,
                center = Offset(cx * scale, cy * scale)
            )
        }
    }
}

/**
 * iOS-style shutter: a thin white outer ring with a solid white inner disc.
 * The inner disc scales down on press for tactile feedback (no ripple — the
 * shape itself is the affordance).
 */
@Composable
private fun ShutterButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val innerScale by animateFloatAsState(
        targetValue = if (pressed) 0.82f else 1f,
        label = "shutterInner"
    )
    Box(
        modifier = modifier
            .size(78.dp)
            .clip(CircleShape)
            .clickable(
                interactionSource = interaction,
                indication = null,
                onClick = onClick
            ),
        contentAlignment = Alignment.Center
    ) {
        Box(
            Modifier
                .size(78.dp)
                .border(width = 4.dp, color = Color.White, shape = CircleShape)
        )
        Box(
            Modifier
                .size(62.dp)
                .scale(innerScale)
                .clip(CircleShape)
                .background(Color.White)
        )
    }
}

/** Circular translucent flip-camera button with a cameraswitch glyph. */
@Composable
private fun FlipButton(
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .size(54.dp)
            .clip(CircleShape)
            .background(Color.White.copy(alpha = 0.18f))
            .clickable(enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = Icons.Filled.Cameraswitch,
            contentDescription = "Flip camera",
            tint = Color.White,
            modifier = Modifier.size(28.dp)
        )
    }
}
