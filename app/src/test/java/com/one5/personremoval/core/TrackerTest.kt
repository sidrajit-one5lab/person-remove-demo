package com.one5.personremoval.core

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class TrackerTest {

    private lateinit var tracker: Tracker

    @Before
    fun setUp() {
        tracker = Tracker(iouThreshold = 0.2f, maxGapMs = 500L)
    }

    private fun person(left: Float, top: Float, right: Float, bottom: Float): Person =
        Person(
            trackId = -1,
            bBox = RectF(left, top, right, bottom),
            mask = ByteArray(0),
            maskWidth = 0,
            maskHeight = 0,
            confidence = 0.9f
        )

    @Test
    fun `assigns stable IDs across frames`() {
        val p = person(10f, 10f, 100f, 200f)
        val (tracked1, _) = tracker.update(listOf(p), 0L)
        val id = tracked1[0].trackId

        val p2 = person(12f, 11f, 102f, 201f)
        val (tracked2, _) = tracker.update(listOf(p2), 100L)
        assertEquals(id, tracked2[0].trackId)
    }

    @Test
    fun `assigns new ID for non-overlapping detection`() {
        val p1 = person(0f, 0f, 50f, 50f)
        val p2 = person(200f, 200f, 300f, 300f)
        val (tracked, _) = tracker.update(listOf(p1, p2), 0L)
        assertTrue(tracked[0].trackId != tracked[1].trackId)
    }

    @Test
    fun `toggle flips KEEP to REMOVE`() {
        val p = person(10f, 10f, 100f, 200f)
        val (tracked, states1) = tracker.update(listOf(p), 0L)
        val id = tracked[0].trackId
        assertEquals(PersonState.KEEP, states1[id])

        tracker.toggle(id)
        val states2 = tracker.getStates()
        assertEquals(PersonState.REMOVE, states2[id])
    }

    @Test
    fun `toggle twice returns to KEEP`() {
        val p = person(10f, 10f, 100f, 200f)
        val (tracked, _) = tracker.update(listOf(p), 0L)
        val id = tracked[0].trackId

        tracker.toggle(id)
        tracker.toggle(id)
        assertEquals(PersonState.KEEP, tracker.getStates()[id])
    }

    @Test
    fun `state persists across frames`() {
        val p = person(10f, 10f, 100f, 200f)
        val (tracked1, _) = tracker.update(listOf(p), 0L)
        val id = tracked1[0].trackId
        tracker.toggle(id)

        val p2 = person(12f, 11f, 102f, 201f)
        val (_, states) = tracker.update(listOf(p2), 100L)
        assertEquals(PersonState.REMOVE, states[id])
    }

    @Test
    fun `stale track expires after gap`() {
        val p = person(10f, 10f, 100f, 200f)
        tracker.update(listOf(p), 0L)

        val (tracked, states) = tracker.update(emptyList(), 600L)
        assertTrue(tracked.isEmpty())
        assertTrue(states.isEmpty())
    }

    @Test
    fun `track survives within gap threshold`() {
        val p = person(10f, 10f, 100f, 200f)
        val (t1, _) = tracker.update(listOf(p), 0L)
        val id = t1[0].trackId
        tracker.toggle(id)

        tracker.update(emptyList(), 400L)

        val p2 = person(12f, 11f, 102f, 201f)
        val (t2, states) = tracker.update(listOf(p2), 450L)
        assertEquals(id, t2[0].trackId)
        assertEquals(PersonState.REMOVE, states[id])
    }

    @Test
    fun `reset clears all tracks`() {
        val p = person(10f, 10f, 100f, 200f)
        tracker.update(listOf(p), 0L)
        tracker.reset()

        val states = tracker.getStates()
        assertTrue(states.isEmpty())
    }

    @Test
    fun `getStates returns snapshot of current states`() {
        val p1 = person(0f, 0f, 50f, 50f)
        val p2 = person(200f, 200f, 300f, 300f)
        val (tracked, _) = tracker.update(listOf(p1, p2), 0L)

        tracker.toggle(tracked[1].trackId)

        val states = tracker.getStates()
        assertEquals(PersonState.KEEP, states[tracked[0].trackId])
        assertEquals(PersonState.REMOVE, states[tracked[1].trackId])
    }
}