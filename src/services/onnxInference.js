/**
 * ONNX Runtime Web Inference Service for IMU_Sync
 * Supports:
 * 1. 100Hz 2-Stage Neural Kinematic System (Exp 1 Transformer + Classifier)
 * 2. SIH Multi-Head Inertial MLP (Dense 120 -> 256 -> 128 -> 64 -> [dx, dy, v, delta_theta]) with Gaussian Filter
 */

import * as ort from 'onnxruntime-web';

// 1D Gaussian Smoothing Filter
export class GaussianFilter1D {
  constructor(kernelSize = 7, sigma = 1.2) {
    this.kernelSize = Math.max(3, kernelSize % 2 === 0 ? kernelSize + 1 : kernelSize);
    this.sigma = Math.max(0.1, sigma);
    this.kernel = [];
    this.buffer = [];
    this.generateKernel();
  }

  generateKernel() {
    this.kernel = [];
    const radius = Math.floor(this.kernelSize / 2);
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      const weight = (1 / (this.sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-(i * i) / (2 * this.sigma * this.sigma));
      this.kernel.push(weight);
      sum += weight;
    }
    for (let i = 0; i < this.kernel.length; i++) {
      this.kernel[i] /= sum;
    }
  }

  process(val) {
    this.buffer.push(val);
    if (this.buffer.length > this.kernelSize) this.buffer.shift();
    if (this.buffer.length < this.kernelSize) return val;
    let smoothed = 0;
    for (let i = 0; i < this.kernelSize; i++) {
      smoothed += this.buffer[i] * this.kernel[i];
    }
    return smoothed;
  }

  reset() {
    this.buffer = [];
  }
}

// 6-DOF Gaussian IMU Smoother
export class GaussianIMUFilter6D {
  constructor(kernelSize = 7, sigma = 1.2) {
    this.fAx = new GaussianFilter1D(kernelSize, sigma);
    this.fAy = new GaussianFilter1D(kernelSize, sigma);
    this.fAz = new GaussianFilter1D(kernelSize, sigma);
    this.fGx = new GaussianFilter1D(kernelSize, sigma);
    this.fGy = new GaussianFilter1D(kernelSize, sigma);
    this.fGz = new GaussianFilter1D(kernelSize, sigma);
  }

  process(ax, ay, az, gx, gy, gz) {
    return [
      this.fAx.process(ax),
      this.fAy.process(ay),
      this.fAz.process(az),
      this.fGx.process(gx),
      this.fGy.process(gy),
      this.fGz.process(gz)
    ];
  }

  reset() {
    this.fAx.reset();
    this.fAy.reset();
    this.fAz.reset();
    this.fGx.reset();
    this.fGy.reset();
    this.fGz.reset();
  }
}

class ONNXInferenceService {
  constructor() {
    this.sessionClassifier = null;
    this.sessionTransformer = null;
    this.sessionSihMlp = null;
    this.isReady = false;
    this.mode = 'tlio'; // 'tlio' | 'sih_mlp' | 'ekf' | 'math'

    this.gaussianFilter = new GaussianIMUFilter6D(7, 1.2);
    this.sihBuffer = [];

    this.scalers = {
      features: {
        names: ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
        mean: [0.0, 0.0, 9.81, 0.0, 0.0, 0.0],
        std: [1.0, 1.0, 0.50, 0.05, 0.05, 0.05]
      },
      targets: {
        names: ['dv_lateral', 'dv_forward'],
        mean: [0.0, 0.0],
        std: [1.0, 0.85]
      },
      calibration: {
        k_accel: 1.1379,
        k_gyro: 0.9794
      }
    };
  }

  async init(modelsBasePath = '/models') {
    try {
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ort.env.wasm.simd = true;

      // 1. Fetch Scaler & Calibration Parameters
      try {
        const scalerRes = await fetch(`${modelsBasePath}/scaler_params.json`);
        if (scalerRes.ok) {
          const scalerJson = await scalerRes.json();
          if (scalerJson.features && scalerJson.targets) {
            this.scalers = scalerJson;
            console.log('[ONNX Service] Scaler parameters loaded:', this.scalers.calibration);
          }
        }
      } catch (e) {
        console.warn('[ONNX Service] Using fallback scalers:', e);
      }

      // 2. Load Exp 1 Stage 1 Motion Classifier (MLP)
      try {
        const clsRes = await fetch(`${modelsBasePath}/motion_classifier.onnx`);
        if (clsRes.ok) {
          const clsBuffer = await clsRes.arrayBuffer();
          this.sessionClassifier = await ort.InferenceSession.create(new Uint8Array(clsBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Exp 1 Motion Classifier ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Classifier loading error:', err);
      }

      // 3. Load Exp 1 Stage 2 Acceleration Transformer (TLIO)
      try {
        const transformerRes = await fetch(`${modelsBasePath}/tlio_transformer.onnx`);
        if (transformerRes.ok) {
          const transBuffer = await transformerRes.arrayBuffer();
          this.sessionTransformer = await ort.InferenceSession.create(new Uint8Array(transBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Exp 1 Acceleration Transformer ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Transformer loading error:', err);
      }

      // 4. Load SIH Multi-Head Inertial MLP (Dense 120 -> 4)
      try {
        const sihRes = await fetch(`${modelsBasePath}/sih_inertial_mlp.onnx`);
        if (sihRes.ok) {
          const sihBuffer = await sihRes.arrayBuffer();
          this.sessionSihMlp = await ort.InferenceSession.create(new Uint8Array(sihBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] SIH Multi-Head Inertial MLP ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] SIH MLP loading error:', err);
      }

      this.isReady = !!(this.sessionClassifier && this.sessionTransformer);
      return this.isReady;
    } catch (err) {
      console.error('[ONNX Service] Failed to initialize ONNX sessions:', err);
      this.isReady = false;
      return false;
    }
  }

  setMode(mode) {
    this.mode = mode;
  }

  normalizeSample(raw) {
    const mean = this.scalers.features.mean;
    const std = this.scalers.features.std;
    const norm = new Float32Array(6);
    for (let i = 0; i < 6; i++) {
      norm[i] = (raw[i] - mean[i]) / (std[i] || 1.0);
    }
    return norm;
  }

  denormalizeAcceleration(rawOutput) {
    const mean = this.scalers.targets.mean;
    const std = this.scalers.targets.std;
    const out = new Float32Array(2);
    for (let i = 0; i < 2; i++) {
      out[i] = rawOutput[i] * (std[i] || 1.0) + mean[i];
    }
    return out;
  }

  /**
   * Runs Experiment 1 2-Stage Gated Acceleration Inference (10 samples @ 10Hz)
   * @param {Array<Array<number>>} rawWindow - 10 consecutive 6-axis IMU samples [10, 6]
   * @returns {Promise<{isMoving: boolean, aFwd: number, aLat: number, latencyMs: number}>}
   */
  async predict1sAcceleration(rawWindow) {
    const t0 = performance.now();
    const windowSize = 10;
    const flatNormWindow = new Float32Array(windowSize * 6);

    for (let t = 0; t < windowSize; t++) {
      const sample = rawWindow[t] || [0, 0, 9.81, 0, 0, 0];
      const norm = this.normalizeSample(sample);
      for (let c = 0; c < 6; c++) {
        flatNormWindow[t * 6 + c] = norm[c];
      }
    }

    let isMoving = true;
    let aFwd = 0.0;
    let aLat = 0.0;
    const kAccel = (this.scalers.calibration && this.scalers.calibration.k_accel) ? this.scalers.calibration.k_accel : 1.1379;

    try {
      const inputTensor = new ort.Tensor('float32', flatNormWindow, [1, windowSize, 6]);

      // STAGE 1: Classify REST vs MOVING
      if (this.sessionClassifier) {
        const clsFeeds = { input_window: inputTensor };
        const clsResults = await this.sessionClassifier.run(clsFeeds);
        const logits = clsResults.motion_logits.data;
        isMoving = logits[1] > logits[0];
      }

      // STAGE 2: If MOVING -> Predict Forward Acceleration; If REST -> a = 0.0
      if (isMoving && this.sessionTransformer) {
        const feeds = { input_window: inputTensor };
        const results = await this.sessionTransformer.run(feeds);
        const outData = results.displacement_1s.data;
        const denorm = this.denormalizeAcceleration([outData[0], outData[1]]);
        aLat = denorm[0];
        aFwd = denorm[1] * kAccel;
      } else {
        aLat = 0.0;
        aFwd = 0.0;
      }
    } catch (err) {
      console.warn('[ONNX Service] Inference error:', err);
    }

    const latencyMs = Math.max(0.01, performance.now() - t0);
    return {
      isMoving,
      aFwd,
      aLat,
      latencyMs
    };
  }

  /**
   * Runs SIH Multi-Head Inertial MLP Inference (20 samples @ 10Hz/50Hz)
   * Pipeline: Gaussian Smoothing -> ZUPT Gating -> Dense 120 -> [dx, dy, v, delta_theta]
   * @param {Array<Array<number>>} rawWindow20 - 20 consecutive 6-axis IMU samples [20, 6]
   * @returns {Promise<{isMoving: boolean, dx: number, dy: number, speedMps: number, speedKmh: number, deltaThetaDeg: number, latencyMs: number}>}
   */
  async predictSihMlp(rawWindow20) {
    const t0 = performance.now();
    const seqLen = 20;

    if (!this.sessionSihMlp || rawWindow20.length < seqLen) {
      return { isMoving: true, dx: 0, dy: 0, speedMps: 0, speedKmh: 0, deltaThetaDeg: 0, latencyMs: 0.1 };
    }

    // 1. Check Stationary Condition (ZUPT Gating)
    let sumNorm = 0, sumSqNorm = 0, sumGyro = 0;
    for (let i = 0; i < seqLen; i++) {
      const [ax, ay, az, gx, gy, gz] = rawWindow20[i];
      const norm = Math.sqrt(ax * ax + ay * ay + az * az);
      const gyroNormDeg = Math.sqrt(gx * gx + gy * gy + gz * gz) * (180 / Math.PI);
      sumNorm += norm;
      sumSqNorm += norm * norm;
      sumGyro += gyroNormDeg;
    }
    const meanNorm = sumNorm / seqLen;
    const accelVariance = Math.max(0, (sumSqNorm / seqLen) - (meanNorm * meanNorm));
    const avgGyroDeg = sumGyro / seqLen;
    const isStationary = accelVariance < 0.05 && avgGyroDeg < 1.8;

    if (isStationary) {
      return { isMoving: false, dx: 0, dy: 0, speedMps: 0, speedKmh: 0, deltaThetaDeg: 0, latencyMs: 0.05 };
    }

    // 2. Format Gaussian-smoothed sequence [1, 20, 6]
    const flatData = new Float32Array(seqLen * 6);
    for (let i = 0; i < seqLen; i++) {
      const [ax, ay, az, gx, gy, gz] = rawWindow20[i];
      const sm = this.gaussianFilter.process(ax, ay, az, gx, gy, gz);
      flatData[i * 6 + 0] = sm[0]; // ax
      flatData[i * 6 + 1] = sm[1]; // ay
      flatData[i * 6 + 2] = sm[2]; // az
      flatData[i * 6 + 3] = sm[5]; // gz (rad/s)
      flatData[i * 6 + 4] = sm[3]; // gx (rad/s)
      flatData[i * 6 + 5] = sm[4]; // gy (rad/s)
    }

    let dx = 0, dy = 0, speedMps = 0, deltaThetaDeg = 0;

    try {
      const inputTensor = new ort.Tensor('float32', flatData, [1, seqLen, 6]);
      const feeds = { imu_sequence: inputTensor };
      const results = await this.sessionSihMlp.run(feeds);
      const outputTensor = results.odometry_output || Object.values(results)[0];
      const outData = outputTensor.data;

      dx = outData[0] || 0;
      dy = outData[1] || 0;
      speedMps = Math.max(0, outData[2] || 0);
      deltaThetaDeg = (outData[3] || 0) * (180 / Math.PI);
    } catch (err) {
      console.warn('[SIH MLP] Inference notice:', err);
    }

    const latencyMs = Math.max(0.01, performance.now() - t0);
    return {
      isMoving: true,
      dx,
      dy,
      speedMps,
      speedKmh: speedMps * 3.6,
      deltaThetaDeg,
      latencyMs
    };
  }
}

export const onnxInferenceService = new ONNXInferenceService();
