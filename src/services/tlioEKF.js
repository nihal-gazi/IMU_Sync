/**
 * TLIO Extended Kalman Filter (EKF) & Strapdown Inertial Kinematics Engine
 * 1. High-Frequency Continuous Propagation: Integrates IMU acceleration & gyro kinematics every frame.
 * 2. 1-Second Transformer Measurement Update: Corrects kinematic drift using the IMU-Transformer's predicted displacement.
 */

export class TLIOEKFEngine {
  constructor() {
    this.reset();
  }

  reset() {
    // Kinematic States
    this.posX = 0.0;
    this.posY = 0.0;
    this.vx = 0.0;
    this.vy = 0.0;
    this.headingRad = 0.0; // 0 = North (+Y)

    // 1-Second Window Tracking
    this.lastSecPosX = 0.0;
    this.lastSecPosY = 0.0;

    // State Covariance Matrix (Position & Velocity)
    this.covPos = 0.5;
    this.covVel = 0.2;

    // Noise Parameters
    this.processNoisePos = 0.05;
    this.processNoiseVel = 0.1;
    this.measNoise = 0.25; // Transformer measurement uncertainty R

    // Metrics for HUD & ML Telemetry
    this.lastCorrectionDelta = { dx: 0, dy: 0, magnitude: 0 };
    this.correctionCount = 0;
    this.lastCorrectionTime = performance.now();
  }

  /**
   * High-Frequency Kinematic Propagation Step (Executed at 60 FPS / per IMU sample)
   * Integrates acceleration specific force and gyro yaw rate.
   */
  predictKinematicStep(imu, dt = 0.1) {
    const [ax, ay, az, gx, gy, gz] = imu;

    // 1. Update Heading (Yaw) from Gyroscope Gz
    this.headingRad += gz * dt;

    // 2. Transform Body Accelerations (Ax, Ay) to World Coordinate Frame
    // In our coordinate system: North is +Y, East is +X
    const cosH = Math.cos(this.headingRad);
    const sinH = Math.sin(this.headingRad);

    const ax_world = ax * cosH - ay * sinH;
    const ay_world = ax * sinH + ay * cosH;

    // 3. Integrate World Accelerations to Velocity with light velocity damping
    const damping = 0.98; // Realistic tire/damping bleed to prevent unconstrained integral explosion
    this.vx = (this.vx + ax_world * dt) * damping;
    this.vy = (this.vy + ay_world * dt) * damping;

    // 4. Integrate Velocity to Position: P = P + V * dt
    this.posX += this.vx * dt;
    this.posY += this.vy * dt;

    // 5. Propagate Covariance
    this.covPos += this.processNoisePos * dt;
    this.covVel += this.processNoiseVel * dt;

    return {
      posX: this.posX,
      posY: this.posY,
      vx: this.vx,
      vy: this.vy,
      speed: Math.hypot(this.vx, this.vy),
      headingRad: this.headingRad,
      headingDeg: (this.headingRad * 180) / Math.PI
    };
  }

  /**
   * 1-Second Transformer Measurement Correction Step
   * Fuses the IMU-Transformer's predicted 1-second displacement [predDx, predDy]
   * with the Kinematic accumulated displacement over the last second.
   */
  applyTransformerCorrection(predDx, predDy) {
    // 1. Calculate Kinematic displacement over the past 1.0 second
    const kinematicDx = this.posX - this.lastSecPosX;
    const kinematicDy = this.posY - this.lastSecPosY;

    // 2. Measurement Innovation / Residual: y = predDisplacement - kinematicDisplacement
    const innovX = predDx - kinematicDx;
    const innovY = predDy - kinematicDy;

    // 3. Kalman Gain Calculation: K = P / (P + R)
    const kalmanGainPos = this.covPos / (this.covPos + this.measNoise);
    const kalmanGainVel = this.covVel / (this.covVel + this.measNoise);

    // 4. State Correction: Update Position and Velocity
    const deltaPosX = kalmanGainPos * innovX;
    const deltaPosY = kalmanGainPos * innovY;

    this.posX += deltaPosX;
    this.posY += deltaPosY;
    this.vx += kalmanGainVel * (innovX / 1.0); // 1.0s window
    this.vy += kalmanGainVel * (innovY / 1.0);

    // 5. Update Covariance: P = (1 - K) * P
    this.covPos = (1.0 - kalmanGainPos) * this.covPos;
    this.covVel = (1.0 - kalmanGainVel) * this.covVel;

    // 6. Reset 1-second anchor baseline for next interval
    this.lastSecPosX = this.posX;
    this.lastSecPosY = this.posY;

    // Record Telemetry
    const correctionMag = Math.hypot(deltaPosX, deltaPosY);
    this.lastCorrectionDelta = {
      dx: deltaPosX,
      dy: deltaPosY,
      magnitude: correctionMag
    };
    this.correctionCount++;
    this.lastCorrectionTime = performance.now();

    return {
      correctedPosX: this.posX,
      correctedPosY: this.posY,
      deltaPosX,
      deltaPosY,
      correctionMag,
      kalmanGainPos,
      correctionCount: this.correctionCount
    };
  }
}

export const tlioEKFEngine = new TLIOEKFEngine();
