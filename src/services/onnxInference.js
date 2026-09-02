/**
 * ONNX Runtime Web Inference Service for 2-Stage Motion System
 * Stage 1: RestMovingClassifierMLP (Zero-Velocity Stationary Detector)
 * Stage 2: IMUTransformerTLIO (Body-Frame Displacement Regressor)
 */

import * as ort from 'onnxruntime-web';

class ONNXInferenceService {
  constructor() {
    this.sessionClassifier = null;
    this.sessionTransformer = null;
    this.sessionRNN = null;
    this.sessionMLP = null;
    this.isReady = false;
    this.mode = 'tlio';

    this.hiddenDim = 32;
    this.hiddenState = new Float32Array(this.hiddenDim);

    this.scalers = {
      features: {
        names: ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
        mean: [0.042, 0.062, 9.847, 0.002, -0.007, 0.002],
        std: [0.954, 0.887, 1.214, 0.124, 0.098, 0.089]
      },
      targets: {
        names: ['dx_lateral', 'dy_forward'],
        mean: [0.0, 2.182],
        std: [1.0, 1.477]
      }
    };
  }

  async init(modelsBasePath = '/models') {
    try {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      // 1. Fetch Scaler Parameters
      try {
        const scalerRes = await fetch(`${modelsBasePath}/scaler_params.json`);
        if (scalerRes.ok) {
          const scalerJson = await scalerRes.json();
          if (scalerJson.features && scalerJson.targets) {
            this.scalers = scalerJson;
            console.log('[ONNX Service] Scaler parameters loaded.');
          }
        }
      } catch (e) {
        console.warn('[ONNX Service] Using fallback scalers:', e);
      }

      // 2. Load Stage 1 Motion Classifier (MLP)
      try {
        const clsRes = await fetch(`${modelsBasePath}/motion_classifier.onnx`);
        if (clsRes.ok) {
          const clsBuffer = await clsRes.arrayBuffer();
          this.sessionClassifier = await ort.InferenceSession.create(new Uint8Array(clsBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Stage 1 Motion Classifier ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Classifier loading error:', err);
      }

      // 3. Load Stage 2 IMU-Transformer
      try {
        const transformerRes = await fetch(`${modelsBasePath}/tlio_transformer.onnx`);
        if (transformerRes.ok) {
          const transBuffer = await transformerRes.arrayBuffer();
          this.sessionTransformer = await ort.InferenceSession.create(new Uint8Array(transBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Stage 2 IMU-Transformer ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Transformer loading error:', err);
      }

      // 4. Load baseline models
      try {
        const rnnRes = await fetch(`${modelsBasePath}/rnn_model.onnx`);
        if (rnnRes.ok) {
          const rnnBuffer = await rnnRes.arrayBuffer();
          this.sessionRNN = await ort.InferenceSession.create(new Uint8Array(rnnBuffer), {
            executionProviders: ['wasm']
          });
        }
      } catch (e) {}

      this.isReady = true;
      return true;
    } catch (err) {
      console.error('[ONNX Service] Failed to initialize ONNX sessions:', err);
      this.isReady = false;
      return false;
    }
  }

  setMode(mode) {
    if (mode === 'tlio' || mode === 'rnn' || mode === 'mlp') {
      this.mode = mode;
    }
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

  denormalizeDisplacement(rawDisp) {
    const mean = this.scalers.targets.mean;
    const std = this.scalers.targets.std;
    const out = new Float32Array(2);
    for (let i = 0; i < 2; i++) {
      out[i] = rawDisp[i] * (std[i] || 1.0) + mean[i];
    }
    return out;
  }

  /**
   * Runs 2-Stage Gated Motion Inference
   * @param {Array<Array<number>>} rawWindow - 10 consecutive 6-axis IMU samples [10, 6]
   * @returns {Promise<{isMoving: boolean, dxLat: number, dyFwd: number, speedKmh: number, latencyMs: number}>}
   */
  async predict1sDisplacement(rawWindow) {
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
    let dxLat = 0.0;
    let dyFwd = 0.0;

    try {
      const inputTensor = new ort.Tensor('float32', flatNormWindow, [1, windowSize, 6]);

      // STAGE 1: Classify REST vs MOVING
      if (this.sessionClassifier) {
        const clsFeeds = { input_window: inputTensor };
        const clsResults = await this.sessionClassifier.run(clsFeeds);
        const logits = clsResults.motion_logits.data; // [logit_rest, logit_moving]
        isMoving = logits[1] > logits[0];
      }

      // STAGE 2: If MOVING -> Run Transformer Regression; If REST -> Return 0.0
      if (isMoving && this.sessionTransformer) {
        const feeds = { input_window: inputTensor };
        const results = await this.sessionTransformer.run(feeds);
        const outData = results.displacement_1s.data;
        const denorm = this.denormalizeDisplacement([outData[0], outData[1]]);
        dxLat = denorm[0];
        dyFwd = Math.max(0.0, denorm[1]);
      } else {
        dxLat = 0.0;
        dyFwd = 0.0;
      }
    } catch (err) {
      console.warn('[ONNX Service] Inference error:', err);
    }

    const latencyMs = Math.max(0.01, performance.now() - t0);
    return {
      isMoving,
      dxLat,
      dyFwd,
      speedKmh: dyFwd * 3.6,
      latencyMs
    };
  }
}

export const onnxInferenceService = new ONNXInferenceService();
