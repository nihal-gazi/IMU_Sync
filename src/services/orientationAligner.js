/**
 * OrientationAligner: 3D Gravity Orientation Estimator & Coordinate Frame Transformation
 * Uses accelerometer gravity readings to estimate phone tilt (Pitch & Roll)
 * and applies a closed-form 3D Rodrigues rotation matrix R_align to project
 * measured [gx, gy, gz] and [ax, ay, az] into the exact dataset dashboard mount frame.
 */

export class OrientationAligner {
  constructor() {
    this.enabled = true;

    // Filtered estimated gravity vector [gx, gy, gz] in phone body frame
    this.gravityEst = [0.0, 0.0, 9.81];

    // Low-pass filter smoothing factor (α ≈ 0.92 for smooth gravity estimation)
    this.alpha = 0.92;

    // 3x3 Alignment Matrix R_align (initialized to Identity)
    this.R = [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ];

    this.pitchDeg = 0.0;
    this.rollDeg = 0.0;
  }

  reset() {
    this.gravityEst = [0.0, 0.0, 9.81];
    this.R = [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ];
    this.pitchDeg = 0.0;
    this.rollDeg = 0.0;
  }

  /**
   * Updates estimated gravity vector and computes Rodrigues 3D Alignment Matrix R_align
   * @param {Array<number>} rawAcc - [ax, ay, az]
   * @param {number} dt - time delta in seconds
   */
  updateGravity(rawAcc, dt = 0.1) {
    const [ax, ay, az] = rawAcc;

    // 1. Low-Pass Filter on Accelerometer to isolate Earth's static gravity vector
    this.gravityEst[0] = this.alpha * this.gravityEst[0] + (1 - this.alpha) * ax;
    this.gravityEst[1] = this.alpha * this.gravityEst[1] + (1 - this.alpha) * ay;
    this.gravityEst[2] = this.alpha * this.gravityEst[2] + (1 - this.alpha) * az;

    const gx = this.gravityEst[0];
    const gy = this.gravityEst[1];
    const gz = this.gravityEst[2];

    const norm = Math.hypot(gx, gy, gz) || 9.81;

    // Unit gravity vector u = [ux, uy, uz] in phone body frame
    const ux = gx / norm;
    const uy = gy / norm;
    const uz = gz / norm;

    // Calculate Pitch and Roll angles in degrees for HUD telemetry
    this.pitchDeg = (Math.asin(Math.max(-1.0, Math.min(1.0, -uy))) * 180.0) / Math.PI;
    this.rollDeg = (Math.atan2(ux, uz) * 180.0) / Math.PI;

    // 2. Compute 3D Rotation Matrix R_align mapping u -> [0, 0, 1]^T
    // Special edge-case: Phone held upside down (uz ≈ -1.0)
    if (uz < -0.999) {
      this.R = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
      ];
      return;
    }

    const denom = 1.0 + uz;
    this.R = [
      [1.0 - (ux * ux) / denom, -(ux * uy) / denom, -ux],
      [-(ux * uy) / denom, 1.0 - (uy * uy) / denom, -uy],
      [ux, uy, uz]
    ];
  }

  /**
   * Applies R_align transformation to a 3D vector [vx, vy, vz]
   */
  rotateVector(v) {
    const R = this.R;
    return [
      R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
      R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
      R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]
    ];
  }

  /**
   * Transforms raw 6-axis IMU [ax, ay, az, gx, gy, gz] into the dataset reference frame
   * @param {Array<number>} rawImu - [ax, ay, az, gx, gy, gz]
   * @param {number} dt - timestep in seconds
   */
  alignIMU(rawImu, dt = 0.1) {
    if (!this.enabled) {
      return {
        alignedImu: rawImu,
        pitchDeg: this.pitchDeg,
        rollDeg: this.rollDeg,
        R: this.R,
        isAligned: false
      };
    }

    const rawAcc = [rawImu[0], rawImu[1], rawImu[2]];
    const rawGyr = [rawImu[3], rawImu[4], rawImu[5]];

    this.updateGravity(rawAcc, dt);

    const alignedAcc = this.rotateVector(rawAcc);
    const alignedGyr = this.rotateVector(rawGyr);

    return {
      alignedImu: [
        alignedAcc[0],
        alignedAcc[1],
        alignedAcc[2],
        alignedGyr[0],
        alignedGyr[1],
        alignedGyr[2]
      ],
      rawAcc,
      rawGyr,
      alignedAcc,
      alignedGyr,
      pitchDeg: this.pitchDeg,
      rollDeg: this.rollDeg,
      gravityEst: this.gravityEst,
      R: this.R,
      isAligned: true
    };
  }
}

export const orientationAligner = new OrientationAligner();
