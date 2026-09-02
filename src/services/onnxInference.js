/**
 * ONNX Runtime Web Inference Service for Experiment 1 (2-Stage Kinematic System)
 * Stage 1: RestMovingClassifierMLP (Zero-Velocity Stationary Detector)
 * Stage 2: IMUTransformerTLIO (Forward Acceleration Δv Estimator)
 * Calibrated with k_accel = 1.1379, k_gyro = 0.9794
 */

import * as ort from 'onnxruntime-web';

class ONNXInferenceService {
  constructor() {
    this.sessionClassifier = null;
    this.sessionTransformer = null;
    this.isReady = false;
    this.mode = 'tlio'; // Experiment 1 2-Stage Kinematic Model

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
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      // 1. Fetch Scaler & Calibration Parameters
      try {
        const scalerRes = await fetch(`${modelsBasePath}/scaler_params.json`);
        if (scalerRes.ok) {
          const scalerJson = await scalerRes.json();
          if (scalerJson.features && scalerJson.targets) {
            this.scalers = scalerJson;
            console.log('[ONNX Service] Exp 1 Scaler parameters loaded with calibration factors:', this.scalers.calibration);
          }
        }
      } catch (e) {
        console.warn('[ONNX Service] Using fallback scalers:', e);
      }

      // 2. Load Stage 1 Motion Classifier (Exp 1 MLP)
      try {
        const clsRes = await fetch(`${modelsBasePath}/motion_classifier.onnx`);
        if (clsRes.ok) {
          const clsBuffer = await clsRes.arrayBuffer();
          this.sessionClassifier = await ort.InferenceSession.create(new Uint8Array(clsBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Exp 1 Stage 1 Motion Classifier ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Classifier loading error:', err);
      }

      // 3. Load Stage 2 Acceleration Transformer (Exp 1 TLIO)
      try {
        const transformerRes = await fetch(`${modelsBasePath}/tlio_transformer.onnx`);
        if (transformerRes.ok) {
          const transBuffer = await transformerRes.arrayBuffer();
          this.sessionTransformer = await ort.InferenceSession.create(new Uint8Array(transBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Exp 1 Stage 2 Acceleration Transformer ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Transformer loading error:', err);
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
   * Runs Experiment 1 2-Stage Gated Acceleration Inference
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
}

export const onnxInferenceService = new ONNXInferenceService();
