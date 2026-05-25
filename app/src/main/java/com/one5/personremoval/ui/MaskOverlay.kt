package com.one5.personremoval.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
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
    persons: List<Person>,
    states: Map<Int, PersonState>,
    frameW: Int,
    frameH: Int,
    modifier: Modifier = Modifier
) {
    Canvas(modifier) {
        if (frameW == 0 || frameH == 0) return@Canvas
        val sx = size.width / frameW.toFloat()
        val sy = size.height / frameH.toFloat()

        for (p in persons) {
            val left = p.bBox.left * sx
            val top = p.bBox.top * sy
            val w = (p.bBox.right - p.bBox.left) * sx
            val h = (p.bBox.bottom - p.bBox.top) * sy

            when (states[p.trackId] ?: PersonState.KEEP) {
                PersonState.KEEP -> {
                    drawRect(
                        color = Color.Green.copy(alpha = 0.7f),
                        topLeft = Offset(left, top),
                        size = Size(w, h),
                        style = Stroke(width = 3f)
                    )
                }
                PersonState.REMOVE -> {
                    drawRect(
                        color = Color.Red.copy(alpha = 0.35f),
                        topLeft = Offset(left, top),
                        size = Size(w, h)
                    )
                    drawRect(
                        color = Color.Red,
                        topLeft = Offset(left, top),
                        size = Size(w, h),
                        style = Stroke(width = 5f)
                    )
                }
            }
        }
    }
}
