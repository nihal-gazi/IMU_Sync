/**
 * 3D Gravity Orientation Aligner Service
 * Rotates raw phone IMU signals into the canonical SCREEN-FACING-UP Reference Frame:
 * - Gravity is strictly aligned along +Z (+9.81 m/s²)
 * - Longitudinal vehicle acceleration/braking is along +Y
 * - Lateral cornering acceleration is along +X
 * - Vehicle turning yaw rate is directly measured by Gz
 */

class OrientationAligner {
  constructor() {
    this.gravityEstimate = [0.0, 0.0, 9.81];
    this.alpha = 0.05; // Low-pass filter smoothing factor (cut-off ~0.5Hz)
    this.enabled = true;
  }

  reset() {
    this.gravityEstimate = [0.0, 0.0, 9.81];
  }

  /**
   * Estimates 3D Gravity Vector using a low-pass filter
   * @param {Array<number>} rawAcc - [ax, ay, az] in m/s²
   */
  updateGravity(rawAcc) {
    const [ax, ay, az] = rawAcc;
    this.gravityEstimate[0] = this.alpha * ax + (1.0 - this.alpha) * this.gravityEstimate[0];
    this.gravityEstimate[1] = this.alpha * ay + (1.0 - this.alpha) * this.gravityEstimate[1];
    this.gravityEstimate[2] = this.alpha * az + (1.0 - this.alpha) * this.gravityEstimate[2];
  }

  /**
   * Computes 3D Rodrigues rotation matrix R_up that aligns measured gravity with [0, 0, 1]^T
   * @returns {Array<Array<number>>} 3x3 rotation matrix
   */
  computeAlignmentMatrix() {
    const gx = this.gravityEstimate[0];
    const gy = this.gravityEstimate[1];
    const gz = this.gravityEstimate[2];
    const norm = Math.hypot(gx, gy, gz);

    if (norm < 1e-4) {
      return [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];
    }

    const v_src = [gx / norm, gy / norm, gz / norm];
    const v_dst = [0.0, 0.0, 1.0]; // Canonical Screen-Facing-Up Z-axis

    // Cross product: v = v_src x v_dst
    const v = [
      v_src[1] * v_dst[2] - v_src[2] * v_dst[1],
      v_src[2] * v_dst[0] - v_src[0] * v_dst[2],
      v_src[0] * v_dst[1] - v_src[1] * v_dst[0]
    ];

    const s = Math.hypot(v[0], v[1], v[2]);
    const c = v_src[0] * v_dst[0] + v_src[1] * v_dst[1] + v_src[2] * v_dst[2];

    if (s < 1e-6) {
      if (c > 0) {
        return [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ];
      } else {
        return [
          [1, 0, 0],
          [0, -1, 0],
          [0, 0, -1]
        ];
      }
    }

    const vx = [
      [0, -v[2], v[1]],
      [v[2], 0, -v[0]],
      [-v[1], v[0], 0]
    ];

    const factor = (1.0 - c) / (s * s);

    // R = I + vx + vx^2 * factor
    const R = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ];

    for (let r = 0; r < 3; r++) {
      for (let col = 0; col < 3; col++) {
        let vx2_rc = 0.0;
        for (let k = 0; k < 3; k++) {
          vx2_rc += vx[r][k] * vx[k][col];
        }
        R[r][col] += vx[r][col] + vx2_rc * factor;
      }
    }

    return R;
  }

  /**
   * Aligns 6-axis IMU readings [ax, ay, az, gx, gy, gz] into the Screen-Facing-Up canonical frame
   * @param {Array<number>} rawImu - [ax, ay, az, gx, gy, gz]
   * @returns {{ alignedImu: Array<number>, pitchDeg: number, rollDeg: number, rawGyr: Array<number> }}
   */
  alignIMU(rawImu) {
    if (!this.enabled) {
      return {
        alignedImu: rawImu,
        pitchDeg: 0,
        rollDeg: 0,
        rawGyr: [rawImu[3], rawImu[4], rawImu[5]]
      };
    }

    const rawAcc = [rawImu[0], rawImu[1], rawImu[2]];
    const rawGyr = [rawImu[3], rawImu[4], rawImu[5]];

    this.updateGravity(rawAcc);
    const R = this.computeAlignmentMatrix();

    // Rotate Accel & Gyro
    const alignedAcc = [
      R[0][0] * rawAcc[0] + R[0][1] * rawAcc[1] + R[0][2] * rawAcc[2],
      R[1][0] * rawAcc[0] + R[1][1] * rawAcc[1] + R[1][2] * rawAcc[2],
      R[2][0] * rawAcc[0] + R[2][1] * rawAcc[1] + R[2][2] * rawAcc[2]
    ];

    const alignedGyr = [
      R[0][0] * rawGyr[0] + R[0][1] * rawGyr[1] + R[0][2] * rawGyr[2],
      R[1][0] * rawGyr[0] + R[1][1] * rawGyr[1] + R[1][2] * rawGyr[2],
      R[2][0] * rawGyr[0] + R[2][1] * rawGyr[1] + R[2][2] * rawGyr[2]
    ];

    const gx = this.gravityEstimate[0];
    const gy = this.gravityEstimate[1];
    const gz = this.gravityEstimate[2];
    const pitchDeg = (Math.atan2(-gy, Math.hypot(gx, gz)) * 180.0) / Math.PI;
    const rollDeg = (Math.atan2(gx, gz) * 180.0) / Math.PI;

    return {
      alignedImu: [
        alignedAcc[0],
        alignedAcc[1],
        alignedAcc[2],
        alignedGyr[0],
        alignedGyr[1],
        alignedGyr[2]
      ],
      pitchDeg,
      rollDeg,
      rawGyr
    };
  }
}

export const orientationAligner = new OrientationAligner();
