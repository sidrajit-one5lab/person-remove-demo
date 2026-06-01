package com.one5.personremoval.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Live capture-guidance pill driven by [ParallaxCoach.State]. Sits at the top
 * of the preview and tells the user the one thing that improves the result:
 * pan to reveal background, almost there, ready, or (frame-filling subject)
 * step aside. Fades in/out with the coach phase; colour signals urgency.
 *
 * Stateless and VM-agnostic — pass it the collected coach state.
 */
@Composable
fun ParallaxCoachOverlay(
    state: ParallaxCoach.State,
    modifier: Modifier = Modifier
) {
    AnimatedVisibility(
        visible = state.visible,
        enter = fadeIn(),
        exit = fadeOut(),
        modifier = modifier
    ) {
        val accent = accentFor(state.phase)
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        ) {
            androidx.compose.foundation.layout.Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.62f), CircleShape)
                        .padding(horizontal = 16.dp, vertical = 10.dp)
                ) {
                    Icon(
                        imageVector = iconFor(state.phase),
                        contentDescription = null,
                        tint = accent,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        text = state.message,
                        color = Color.White,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium
                    )
                }

                // Thin sweep-progress bar for the active panning phases. Hidden
                // at READY / STEP_ASIDE where progress isn't the point.
                if (state.phase == ParallaxCoach.Phase.PAN ||
                    state.phase == ParallaxCoach.Phase.ALMOST
                ) {
                    Spacer(Modifier.height(8.dp))
                    SweepBar(progress = state.progress, accent = accent)
                }
            }
        }
    }
}

@Composable
private fun SweepBar(progress: Float, accent: Color) {
    val animated by animateFloatAsState(
        targetValue = progress.coerceIn(0f, 1f),
        label = "sweepProgress"
    )
    Box(
        modifier = Modifier
            .width(140.dp)
            .height(4.dp)
            .background(Color.White.copy(alpha = 0.25f), RoundedCornerShape(2.dp))
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth(animated)
                .height(4.dp)
                .background(accent, RoundedCornerShape(2.dp))
        )
    }
}

private fun iconFor(phase: ParallaxCoach.Phase): ImageVector = when (phase) {
    ParallaxCoach.Phase.READY -> Icons.Filled.CheckCircle
    ParallaxCoach.Phase.STEP_ASIDE -> Icons.Filled.WarningAmber
    else -> Icons.Filled.SwapHoriz
}

private fun accentFor(phase: ParallaxCoach.Phase): Color = when (phase) {
    ParallaxCoach.Phase.READY -> Color(0xFF4CAF50)
    ParallaxCoach.Phase.STEP_ASIDE -> Color(0xFFFF7043)
    else -> Color(0xFFFFC107)
}