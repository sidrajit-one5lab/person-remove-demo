package com.one5.personremoval.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Integrates device gyroscope samples into a running rotation matrix
 * relative to the moment [start] was called. Used by the alignment pipeline
 * to construct a rough homography when ORB / AKAZE feature matching fails
 * on texture-poor scenes (sky, white walls, ceiling).
 *
 * Math: TYPE_GYROSCOPE returns angular velocity (rad/s) about device-local
 * X, Y, Z axes. We integrate each sample into a small-angle rotation
 * matrix and compose with the running rotation. Drift accumulates over
 * tens of seconds; for our ~3 s ring buffer window the error stays within
 * ~1° which projects to a few pixels at typical analyzer resolutions.
 *
 * Threading: register/unregister on the main thread (SensorManager
 * requirement). Sensor callbacks arrive on a sensor-dedicated thread; the
 * rotation matrix is updated under a synchronized block. [snapshot] is
 * safe to call from any thread.
 *
 * If the device has no gyroscope (rare, but possible on cheap or emulator
 * targets), [isAvailable] stays false and [snapshot] returns identity —
 * the caller's gyro-fallback path simply does nothing.
 */
class GyroIntegrator(context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "GyroIntegrator"
        // NS_PER_SECOND for the SensorEvent.timestamp (nanoseconds).
        private const val NS_PER_SECOND = 1_000_000_000.0
    }

    private val sensorManager: SensorManager =
            context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    val isAvailable: Boolean get() = gyro != null

    private val lock = Any()
    // 3x3 row-major rotation matrix. Initialized to identity.
    private val R = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )
    private var lastTimestampNs: Long = 0L
    private var running: Boolean = false

    // Most recent sample's angular speed magnitude (rad/s) and the wall-clock
    // ms at which it was recorded. Read by the capture coroutine via
    // [recentAngularVelocityRadPerSec] to gate the shutter on phone
    // stillness. Volatile because the writer is the sensor thread and the
    // reader is the capture coroutine — no atomic read-modify-write needed.
    @Volatile private var lastOmegaMagRadPerSec: Float = Float.POSITIVE_INFINITY
    @Volatile private var lastOmegaWallMs: Long = 0L

    init {
        if (gyro == null) {
            Log.w(TAG, "Gyroscope not available on this device; rotation snapshots will be identity")
        }
    }

    /** Begin accumulating rotation samples. Resets to identity. */
    fun start() {
        val g = gyro ?: return
        synchronized(lock) {
            resetRMatrix()
            lastTimestampNs = 0L
            if (!running) {
                sensorManager.registerListener(this, g, SensorManager.SENSOR_DELAY_GAME)
                running = true
            }
        }
    }

    /** Stop accumulating and detach the listener. */
    fun stop() {
        synchronized(lock) {
            if (running) {
                sensorManager.unregisterListener(this)
                running = false
            }
        }
    }

    /**
     * Snapshot the current cumulative rotation. Returns a new 9-element
     * float array (row-major 3x3). Safe to call from any thread.
     */
    fun snapshot(): FloatArray {
        synchronized(lock) {
            return R.copyOf()
        }
    }

    /**
     * Most recent angular speed magnitude in rad/s, or +∞ if no sample has
     * arrived within [staleMs]. Used by the capture coroutine to wait for
     * phone stillness before freezing the ring buffer — a fresh "tap →
     * shutter" delay that adapts to actual motion instead of a fixed
     * timeout. Returns +∞ when the device has no gyroscope so callers'
     * threshold comparisons fall through to the max-wait fallback.
     */
    fun recentAngularVelocityRadPerSec(staleMs: Long = 200L): Float {
        if (gyro == null) return Float.POSITIVE_INFINITY
        val now = System.currentTimeMillis()
        if (now - lastOmegaWallMs > staleMs) return Float.POSITIVE_INFINITY
        return lastOmegaMagRadPerSec
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* no-op */ }

    override fun onSensorChanged(event: SensorEvent) {
        // event.values = [omegaX, omegaY, omegaZ] in rad/s.
        // event.timestamp in ns (monotonic).
        val tNs = event.timestamp
        if (lastTimestampNs == 0L) {
            lastTimestampNs = tNs
            return
        }
        val dt = (tNs - lastTimestampNs) / NS_PER_SECOND
        lastTimestampNs = tNs
        if (dt <= 0.0 || dt > 0.5) {
            // Outliers: monotonic anomaly or huge gap (app paused). Skip
            // sample but reset baseline so the next dt is sane.
            return
        }

        val wx = event.values[0]
        val wy = event.values[1]
        val wz = event.values[2]
        val omegaMag = sqrt((wx * wx + wy * wy + wz * wz).toDouble())
        // Publish the instantaneous angular-speed magnitude before the
        // sub-precision early-return below. Steady phone produces samples
        // at gyro bias level (~0.005 rad/s); intentional motion is orders
        // of magnitude larger, so the noise floor is fine for our use.
        lastOmegaMagRadPerSec = omegaMag.toFloat()
        lastOmegaWallMs = System.currentTimeMillis()
        val angle = omegaMag * dt
        if (angle < 1e-9) return  // sub-precision; ignore

        // Rodrigues' rotation formula. Build incremental dR = I + sinθ K + (1-cosθ) K²
        // where K is the skew-symmetric of the unit axis. For small dt this
        // is essentially the matrix exponential.
        val invMag = 1.0 / (angle / dt)  // 1 / |omega|
        val kx = wx * invMag
        val ky = wy * invMag
        val kz = wz * invMag
        val s = sin(angle)
        val c = 1.0 - cos(angle)

        val dR00 = (1.0 - c * (ky * ky + kz * kz)).toFloat()
        val dR01 = (c * (kx * ky) - s * kz).toFloat()
        val dR02 = (c * (kx * kz) + s * ky).toFloat()
        val dR10 = (c * (kx * ky) + s * kz).toFloat()
        val dR11 = (1.0 - c * (kx * kx + kz * kz)).toFloat()
        val dR12 = (c * (ky * kz) - s * kx).toFloat()
        val dR20 = (c * (kx * kz) - s * ky).toFloat()
        val dR21 = (c * (ky * kz) + s * kx).toFloat()
        val dR22 = (1.0 - c * (kx * kx + ky * ky)).toFloat()

        synchronized(lock) {
            // R := R * dR
            val r00 = R[0]; val r01 = R[1]; val r02 = R[2]
            val r10 = R[3]; val r11 = R[4]; val r12 = R[5]
            val r20 = R[6]; val r21 = R[7]; val r22 = R[8]

            R[0] = r00 * dR00 + r01 * dR10 + r02 * dR20
            R[1] = r00 * dR01 + r01 * dR11 + r02 * dR21
            R[2] = r00 * dR02 + r01 * dR12 + r02 * dR22
            R[3] = r10 * dR00 + r11 * dR10 + r12 * dR20
            R[4] = r10 * dR01 + r11 * dR11 + r12 * dR21
            R[5] = r10 * dR02 + r11 * dR12 + r12 * dR22
            R[6] = r20 * dR00 + r21 * dR10 + r22 * dR20
            R[7] = r20 * dR01 + r21 * dR11 + r22 * dR21
            R[8] = r20 * dR02 + r21 * dR12 + r22 * dR22
        }
    }

    private fun resetRMatrix() {
        R[0] = 1f; R[1] = 0f; R[2] = 0f
        R[3] = 0f; R[4] = 1f; R[5] = 0f
        R[6] = 0f; R[7] = 0f; R[8] = 1f
    }
}
