package com.one5.personremoval.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import com.one5.personremoval.core.Person
import com.one5.personremoval.core.PersonState

/**
 * Draws per-person bbox overlays over the camera preview.
 *
 *   KEEP   → thin green outline
 *   REMOVE → translucent red fill + thicker red outline
 *
 * Per-pixel mask rendering was tried and reverted because the speckled
 * silhouette didn't look good in the live preview. The mask data still lives
 * on each [Person] and is used by the tap hit-test in CaptureScreen — only
 * the visual layer here is bbox-only.
 *
 * @param frameW source frame width  (analyzer image width, NOT preview width)
 * @param frameH source frame height
 */
@Composable
fun MaskOverlay(
    modifier: Modifier = Modifier,
    persons: List<Person>,
    states: Map<Int, PersonState>,
    frameW: Int,
    frameH: Int,
    isFrontCamera: Boolean = false,
) {
    Canvas(modifier) {
        if (frameW == 0 || frameH == 0) return@Canvas
        // Match PreviewView's default FILL_CENTER: scale the frame uniformly to
        // fill the view, center-cropping the overflow axis. A FIT_XY stretch
        // (independent sx/sy over the full frame) misaligns the boxes because
        // the preview crops the overflow instead of squishing it — boxes drift
        // sideways from the person they outline.
        val scale = maxOf(size.width / frameW.toFloat(), size.height / frameH.toFloat())
        val offsetX = (size.width - frameW * scale) / 2f
        val offsetY = (size.height - frameH * scale) / 2f

        for (p in persons) {
            val frameLeft = if (isFrontCamera) (frameW - p.bBox.right) else p.bBox.left
            val left = frameLeft * scale + offsetX
            val top = p.bBox.top * scale + offsetY
            val w = (p.bBox.right - p.bBox.left) * scale
            val h = (p.bBox.bottom - p.bBox.top) * scale

            // Boxes are hidden during normal runtime. Only a person the user
            // has tapped (REMOVE) gets a translucent grey fill, so selection
            // is the single piece of visual feedback on the preview.
            if ((states[p.trackId] ?: PersonState.KEEP) == PersonState.REMOVE) {
                drawRect(
                    color = Color.Red.copy(alpha = 0.2f),
                    topLeft = Offset(left, top),
                    size = Size(w, h)
                )
            }
        }
    }
}
