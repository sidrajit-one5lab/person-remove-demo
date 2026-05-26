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
import androidx.compose.material3.CircularProgressIndicator
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
            val removeCount = personStates.count { it.value == com.one5.personremoval.core.PersonState.REMOVE }
            val cam = if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) "front" else "back"
            Text(
                text = "$cam   persons: ${d.persons.size}   remove: $removeCount   " +
                        "${d.inferenceMs} ms   buf: $bufSize/45",
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

        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 32.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                enabled = !isProcessing,
                onClick = { viewModel.onCapture() }
            ) {
                Text("Capture")
            }
            Button(
                enabled = !isProcessing,
                onClick = { viewModel.onFlipCamera() }
            ) {
                Text("Flip")
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
                    CircularProgressIndicator(
                        color = Color.White,
                        strokeWidth = 3.dp,
                        modifier = Modifier.size(44.dp)
                    )
                    Spacer(Modifier.height(16.dp))
                    Text(
                        text = "Processing…",
                        color = Color.White,
                        fontSize = 14.sp
                    )
                }
            }
        }
    }
}
