/**
 * Mathematical Sensor Fusion & Extended Kalman Filter (EKF) Service
 * Implements 6-DOF Extended Kalman Filter and Pure Classical Kinematic Dead-Reckoning
 * 
 * Features:
 * - State Vector: x = [pos_x, pos_y, vel_x, vel_y, theta (heading), gyro_bias_z]^T
 * - 3D Gravity Tilt Compensation (Roll / Pitch from Accelerometer)
 * - Zero-Velocity Update (ZUPT) Detection & Kalman Measurement Update
 * - Non-Holonomic Vehicle Kinematic Constraint (Zero Lateral Velocity Update)
 * - Pure Mathematical Dead-Reckoning Baseline
 */

export class ExtendedKalmanFilter {
  constructor() {
    this.reset();
  }

  reset(initialHeadingRad = 0.0) {
    // State Vector: [px, py, vx, vy, theta, bg_z]
    this.x = new Float64Array([0.0, 0.0, 0.0, 0.0, initialHeadingRad, 0.0]);

    // Error Covariance Matrix P (6x6)
    this.P = this.createIdentity6(0.1);
    this.P[0][0] = 1.0; // pos x variance
    this.P[1][1] = 1.0; // pos y variance
    this.P[2][2] = 0.5; // vel x variance
    this.P[3][3] = 0.5; // vel y variance
    this.P[4][4] = 0.05; // heading variance (rad^2)
    this.P[5][5] = 0.001; // gyro bias variance

    // Process Noise Covariance Q
    this.qPos = 0.01;
    this.qVel = 0.15;
    this.qTheta = 0.002;
    this.qBias = 0.00001;

    // Measurement Noise Variances
    this.rZupt = 0.05;      // ZUPT velocity variance
    this.rNhc = 0.20;       // Non-holonomic lateral velocity variance
    this.rTiltHeading = 0.1;// Tilt heading variance

    // Stationary ZUPT detector window
    this.accHistory = [];
    this.gyrHistory = [];
    this.historySize = 10;
    this.isStationary = false;
  }

  createIdentity6(diagVal = 1.0) {
    const mat = [];
    for (let i = 0; i < 6; i++) {
      const row = new Float64Array(6);
      row[i] = diagVal;
      mat.push(row);
    }
    return mat;
  }

  /**
   * Evaluates stationary condition based on acceleration variance and gyro magnitude
   */
  checkStationary(ax, ay, az, gx, gy, gz) {
    const aMag = Math.sqrt(ax * ax + ay * ay + az * az);
    const gMag = Math.sqrt(gx * gx + gy * gy + gz * gz);

    this.accHistory.push(aMag);
    this.gyrHistory.push(gMag);
    if (this.accHistory.length > this.historySize) {
      this.accHistory.shift();
      this.gyrHistory.shift();
    }

    if (this.accHistory.length < this.historySize) {
      return false;
    }

    const aMean = this.accHistory.reduce((a, b) => a + b, 0) / this.historySize;
    const aVar = this.accHistory.reduce((s, v) => s + (v - aMean) * (v - aMean), 0) / this.historySize;
    const gMean = this.gyrHistory.reduce((a, b) => a + b, 0) / this.historySize;

    // Stationary thresholds for vehicle console/smartphone
    this.isStationary = (aVar < 0.08 && Math.abs(aMean - 9.81) < 0.6 && gMean < 0.08);
    return this.isStationary;
  }

  /**
   * Main EKF Step: Predict + Measurement Updates (ZUPT + NHC)
   * @param {Array<number>} imu - [ax, ay, az, gx, gy, gz] in Screen-Up Frame
   * @param {number} dt - Time step in seconds (e.g. 0.1s for 10Hz)
   */
  step(imu, dt = 0.1) {
    const t0 = performance.now();
    const [ax, ay, az, gx, gy, gz] = imu;

    // 1. Compute 3D Gravity Tilt Angles (Pitch and Roll)
    const pitch = Math.atan2(ay, Math.sqrt(ax * ax + az * az));
    const roll = Math.atan2(-ax, az);

    // Dynamic linear acceleration with gravity projection removed
    const aFwd = ay * Math.cos(pitch) - (az - 9.81) * Math.sin(pitch);
    const aLat = ax * Math.cos(roll) + (az - 9.81) * Math.sin(roll);

    const isRest = this.checkStationary(ax, ay, az, gx, gy, gz);

    // Current states
    let px = this.x[0];
    let py = this.x[1];
    let vx = this.x[2];
    let vy = this.x[3];
    let th = this.x[4];
    let bg = this.x[5];

    // ==========================================
    // 2. STATE PREDICTION (Physics Propagation)
    // ==========================================
    const effGz = gz - bg;
    const thNew = th + effGz * dt;

    // World-frame dynamic acceleration
    const sinTh = Math.sin(thNew);
    const cosTh = Math.cos(thNew);
    const axWorld = isRest ? 0.0 : (aFwd * sinTh + aLat * cosTh);
    const ayWorld = isRest ? 0.0 : (aFwd * cosTh - aLat * sinTh);

    const vxNew = isRest ? 0.0 : (vx + axWorld * dt);
    const vyNew = isRest ? 0.0 : (vy + ayWorld * dt);
    const pxNew = px + ((vx + vxNew) / 2.0) * dt;
    const pyNew = py + ((vy + vyNew) / 2.0) * dt;

    // Assign predicted state
    this.x[0] = pxNew;
    this.x[1] = pyNew;
    this.x[2] = vxNew;
    this.x[3] = vyNew;
    this.x[4] = thNew;
    this.x[5] = bg; // bias random walk

    // ==========================================
    // 3. COVARIANCE PROPAGATION (P = F * P * F^T + Q)
    // ==========================================
    const F = this.createIdentity6(1.0);
    F[0][2] = dt; // d(px)/d(vx)
    F[1][3] = dt; // d(py)/d(vy)
    F[2][4] = (aFwd * cosTh - aLat * sinTh) * dt; // d(vx)/d(theta)
    F[3][4] = (-aFwd * sinTh - aLat * cosTh) * dt; // d(vy)/d(theta)
    F[4][5] = -dt; // d(theta)/d(bg)

    // P_new = F * P * F^T + Q
    const FP = this.matMul6(F, this.P);
    const FT = this.transpose6(F);
    const FPFT = this.matMul6(FP, FT);

    // Add process noise Q
    FPFT[0][0] += this.qPos * dt;
    FPFT[1][1] += this.qPos * dt;
    FPFT[2][2] += this.qVel * dt;
    FPFT[3][3] += this.qVel * dt;
    FPFT[4][4] += this.qTheta * dt;
    FPFT[5][5] += this.qBias * dt;

    this.P = FPFT;

    // ==========================================
    // 4. MEASUREMENT UPDATES (ZUPT + NHC)
    // ==========================================
    if (isRest) {
      // Zero-Velocity Update (ZUPT): vx = 0, vy = 0, update gyro bias
      this.applyZuptUpdate(gz);
    } else {
      // Non-Holonomic Constraint (NHC): Lateral velocity in body frame = 0
      // v_lateral = -vx * sin(th) + vy * cos(th) = 0
      this.applyNhcUpdate(thNew);
    }

    const latencyMs = Math.max(0.01, performance.now() - t0);
    const speedMps = Math.sqrt(this.x[2] * this.x[2] + this.x[3] * this.x[3]);
    const speedKmh = speedMps * 3.6;

    // Calculate Covariance Trace
    let covTrace = 0.0;
    for (let i = 0; i < 6; i++) covTrace += this.P[i][i];

    return {
      posX: this.x[0],
      posY: this.x[1],
      vx: this.x[2],
      vy: this.x[3],
      speedMps: speedMps,
      speedKmh: speedKmh,
      headingRad: this.x[4],
      headingDeg: ((this.x[4] * 180.0) / Math.PI) % 360,
      gyroBias: this.x[5],
      pitchDeg: (pitch * 180.0) / Math.PI,
      rollDeg: (roll * 180.0) / Math.PI,
      aFwd: aFwd,
      aLat: aLat,
      isMoving: !isRest,
      covTrace: covTrace,
      latencyMs: latencyMs
    };
  }

  /**
   * ZUPT Measurement Update: Directly forces velocities to 0 and estimates gyro bias
   */
  applyZuptUpdate(measuredGz) {
    // 1. Clamp velocities
    this.x[2] = 0.0;
    this.x[3] = 0.0;
    this.P[2][2] = Math.min(this.P[2][2], 0.01);
    this.P[3][3] = Math.min(this.P[3][3], 0.01);

    // 2. Gyro Bias update from measured stationary Gz
    const biasInnov = measuredGz - this.x[5];
    const kGain = this.P[5][5] / (this.P[5][5] + 0.05);
    this.x[5] += kGain * biasInnov;
    this.P[5][5] *= (1.0 - kGain);
  }

  /**
   * Non-Holonomic Constraint (NHC) Measurement Update
   * Measurement: z = -vx * sin(th) + vy * cos(th) = 0
   */
  applyNhcUpdate(th) {
    const sinTh = Math.sin(th);
    const cosTh = Math.cos(th);

    // Measurement function value
    const vLat = -this.x[2] * sinTh + this.x[3] * cosTh;
    const yInnov = 0.0 - vLat; // Innovation

    // Measurement Jacobian H (1x6): [0, 0, -sin(th), cos(th), -vx*cos(th) - vy*sin(th), 0]
    const H = [0.0, 0.0, -sinTh, cosTh, -this.x[2] * cosTh - this.x[3] * sinTh, 0.0];

    // S = H * P * H^T + R
    let S = this.rNhc;
    const PHt = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        PHt[i] += this.P[i][j] * H[j];
      }
      S += H[i] * PHt[i];
    }

    if (Math.abs(S) < 1e-6) return;

    // Kalman Gain K = P * H^T / S (6x1)
    const K = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      K[i] = PHt[i] / S;
    }

    // State update: x = x + K * yInnov
    for (let i = 0; i < 6; i++) {
      this.x[i] += K[i] * yInnov;
    }

    // Covariance update: P = (I - K * H) * P
    const I_KH = this.createIdentity6(1.0);
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        I_KH[i][j] -= K[i] * H[j];
      }
    }
    this.P = this.matMul6(I_KH, this.P);
  }

  // 6x6 Matrix Math Helpers
  matMul6(A, B) {
    const C = [];
    for (let i = 0; i < 6; i++) {
      const row = new Float64Array(6);
      for (let j = 0; j < 6; j++) {
        let sum = 0.0;
        for (let k = 0; k < 6; k++) {
          sum += A[i][k] * B[k][j];
        }
        row[j] = sum;
      }
      C.push(row);
    }
    return C;
  }

  transpose6(A) {
    const AT = [];
    for (let i = 0; i < 6; i++) {
      const row = new Float64Array(6);
      for (let j = 0; j < 6; j++) {
        row[j] = A[j][i];
      }
      AT.push(row);
    }
    return AT;
  }
}

/**
 * Pure Mathematical Classical Kinematic Dead-Reckoning
 * Double-integration with High-Pass Accelerometer Filtering & Trapezoidal Rule
 */
export class ClassicalMathDeadReckoning {
  constructor() {
    this.reset();
  }

  reset(initialHeadingRad = 0.0) {
    this.posX = 0.0;
    this.posY = 0.0;
    this.speedMps = 0.0;
    this.headingRad = initialHeadingRad;
    this.aFwdPrev = 0.0;
    this.accHistory = [];
  }

  step(imu, dt = 0.1) {
    const t0 = performance.now();
    const [ax, ay, az, gx, gy, gz] = imu;

    // 1. Tilt angles
    const pitch = Math.atan2(ay, Math.sqrt(ax * ax + az * az));
    const roll = Math.atan2(-ax, az);

    // 2. Remove gravity
    const aFwdRaw = ay * Math.cos(pitch) - (az - 9.81) * Math.sin(pitch);
    const aLatRaw = ax * Math.cos(roll) + (az - 9.81) * Math.sin(roll);

    // 3. Stationary check
    const aMag = Math.sqrt(ax * ax + ay * ay + az * az);
    const gMag = Math.sqrt(gx * gx + gy * gy + gz * gz);
    const isRest = Math.abs(aMag - 9.81) < 0.35 && gMag < 0.05;

    // 4. Heading integration
    this.headingRad += gz * dt;

    // 5. Trapezoidal Speed Integration with Dead-Band
    let aFwdFiltered = isRest ? 0.0 : aFwdRaw;
    if (Math.abs(aFwdFiltered) < 0.15) aFwdFiltered = 0.0; // Dead-band filter

    const vPrev = this.speedMps;
    if (isRest) {
      this.speedMps = 0.0;
    } else {
      this.speedMps = Math.max(0.0, this.speedMps + ((this.aFwdPrev + aFwdFiltered) / 2.0) * dt);
    }
    this.aFwdPrev = aFwdFiltered;

    // 6. 2D Position update
    const fwdDisp = ((vPrev + this.speedMps) / 2.0) * dt;
    const dx = fwdDisp * Math.sin(this.headingRad);
    const dy = fwdDisp * Math.cos(this.headingRad);

    this.posX += dx;
    this.posY += dy;

    const latencyMs = Math.max(0.01, performance.now() - t0);

    return {
      posX: this.posX,
      posY: this.posY,
      vx: dx / dt,
      vy: dy / dt,
      speedMps: this.speedMps,
      speedKmh: this.speedMps * 3.6,
      headingRad: this.headingRad,
      headingDeg: ((this.headingRad * 180.0) / Math.PI) % 360,
      pitchDeg: (pitch * 180.0) / Math.PI,
      rollDeg: (roll * 180.0) / Math.PI,
      aFwd: aFwdFiltered,
      aLat: aLatRaw,
      isMoving: !isRest,
      covTrace: 0.0,
      latencyMs: latencyMs
    };
  }
}

export const ekfFilterService = new ExtendedKalmanFilter();
export const classicalMathService = new ClassicalMathDeadReckoning();
