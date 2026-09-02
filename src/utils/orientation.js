/**
 * Device orientation parsing and robust tilt-compensated compass heading calculation.
 * Extracted from SIH (ultra-GPS)
 */

export function normalizeDegrees(deg) {
  return ((deg % 360) + 360) % 360;
}

export function angularDifference(targetDeg, sourceDeg) {
  return ((targetDeg - sourceDeg + 540) % 360) - 180;
}

/**
 * Computes robust 3D tilt-compensated compass heading from W3C Euler angles (alpha, beta, gamma).
 */
export function computeRobustCompassHeading(alpha, beta, gamma) {
  const deg2rad = Math.PI / 180.0;
  const rad2deg = 180.0 / Math.PI;

  const a = (alpha || 0) * deg2rad;
  const b = (beta || 0) * deg2rad;
  const g = (gamma || 0) * deg2rad;

  const sA = Math.sin(a);
  const cA = Math.cos(a);
  const sB = Math.sin(b);
  const sG = Math.sin(g);
  const cG = Math.cos(g);

  // W3C Standard horizontal projection of the phone's forward vector:
  const x = -sA * cG - cA * sB * sG;
  const y = cA * cG - sA * sB * sG;

  let heading = Math.atan2(x, y) * rad2deg;
  if (heading < 0) {
    heading += 360;
  }

  return normalizeDegrees(heading);
}

/**
 * Fuses high-frequency Gyroscope yaw rate with low-frequency Compass absolute heading
 * to eliminate gyro drift while preventing compass jitter and over-rotation.
 */
export function fuseGyroCompass(prevHeadingDeg, gyroYawDegPerSec, dt, compassHeadingDeg, alpha = 0.96) {
  // 1. Predict heading from Gyroscope
  const gyroPredicted = normalizeDegrees(prevHeadingDeg + gyroYawDegPerSec * dt);

  if (compassHeadingDeg === null || compassHeadingDeg === undefined || isNaN(compassHeadingDeg)) {
    return gyroPredicted;
  }

  // 2. Shortest angular difference between compass and gyro prediction
  const diff = angularDifference(compassHeadingDeg, gyroPredicted);

  // 3. Fused heading
  const fused = normalizeDegrees(gyroPredicted + (1.0 - alpha) * diff);
  return fused;
}
