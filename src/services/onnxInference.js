/**
 * ONNX Runtime Web Inference Service for IMU-Sync & TLIO Transformer
 * Includes:
 * 1. IMUTransformerTLIO: 1-Second Sliding Window -> [dx_1s, dy_1s] Displacement Vector
 * 2. SimpleRNN / SimpleMLP baseline models
 */

import * as ort from 'onnxruntime-web';

class ONNXInferenceService {
  constructor() {
    this.sessionTransformer = null;
    this.sessionRNN = null;
    this.sessionMLP = null;
    this.isReady = false;
    this.mode = 'tlio'; // 'tlio' | 'rnn' | 'mlp'

    // RNN Hidden state tensor (1 x 32)
    this.hiddenDim = 32;
    this.hiddenState = new Float32Array(this.hiddenDim);

    // Normalization Scalers
    this.scalers = {
      features: {
        names: ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
        mean: [0.042, 0.062, 9.847, 0.002, -0.007, 0.002],
        std: [0.954, 0.887, 1.214, 0.124, 0.098, 0.089]
      },
      targets: {
        names: ['dx_1s', 'dy_1s'],
        mean: [0.0002, -0.215],
        std: [1.483, 1.852]
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
            console.log('[ONNX Service] TLIO Scaler parameters loaded.');
          }
        }
      } catch (e) {
        console.warn('[ONNX Service] Using fallback scalers:', e);
      }

      // 2. Fetch and Load TLIO IMU-Transformer ONNX Model
      try {
        const transformerRes = await fetch(`${modelsBasePath}/tlio_transformer.onnx`);
        if (transformerRes.ok) {
          const transBuffer = await transformerRes.arrayBuffer();
          this.sessionTransformer = await ort.InferenceSession.create(new Uint8Array(transBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] TLIO IMU-Transformer ONNX session initialized.');
        }
      } catch (err) {
        console.warn('[ONNX Service] TLIO Transformer loading:', err);
      }

      // 3. Fetch and Load RNN Model
      try {
        const rnnRes = await fetch(`${modelsBasePath}/rnn_model.onnx`);
        if (rnnRes.ok) {
          const rnnBuffer = await rnnRes.arrayBuffer();
          this.sessionRNN = await ort.InferenceSession.create(new Uint8Array(rnnBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] SimpleRNN ONNX session initialized.');
        }
      } catch (e) {
        console.warn('[ONNX Service] RNN model loading:', e);
      }

      // 4. Fetch and Load MLP Model
      try {
        const mlpRes = await fetch(`${modelsBasePath}/mlp_model.onnx`);
        if (mlpRes.ok) {
          const mlpBuffer = await mlpRes.arrayBuffer();
          this.sessionMLP = await ort.InferenceSession.create(new Uint8Array(mlpBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] SimpleMLP ONNX session initialized.');
        }
      } catch (e) {
        console.warn('[ONNX Service] MLP model loading:', e);
      }

      this.isReady = true;
      return true;
    } catch (err) {
      console.error('[ONNX Service] Failed to initialize ONNX sessions:', err);
      this.isReady = false;
      return false;
    }
  }

  resetHiddenState() {
    this.hiddenState.fill(0);
  }

  setMode(mode) {
    if (mode === 'tlio' || mode === 'rnn' || mode === 'mlp') {
      this.mode = mode;
      if (mode === 'rnn') this.resetHiddenState();
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
   * Runs 1-Second Window TLIO Transformer Inference
   * @param {Array<Array<number>>} rawWindow - 10 consecutive 6-axis IMU samples [10, 6]
   * @returns {Promise<{dx: number, dy: number, latencyMs: number}>}
   */
  async predict1sDisplacement(rawWindow) {
    const t0 = performance.now();
    const windowSize = 10;
    const flatNormWindow = new Float32Array(windowSize * 6);

    // Normalize each sample in the 1-second window
    for (let t = 0; t < windowSize; t++) {
      const sample = rawWindow[t] || [0, 0, 9.81, 0, 0, 0];
      const norm = this.normalizeSample(sample);
      for (let c = 0; c < 6; c++) {
        flatNormWindow[t * 6 + c] = norm[c];
      }
    }

    let predDx = 0.0;
    let predDy = 0.0;

    try {
      if (this.sessionTransformer) {
        const inputTensor = new ort.Tensor('float32', flatNormWindow, [1, windowSize, 6]);
        const feeds = { input_window: inputTensor };
        const results = await this.sessionTransformer.run(feeds);

        const outData = results.displacement_1s.data;
        const denorm = this.denormalizeDisplacement([outData[0], outData[1]]);
        predDx = denorm[0];
        predDy = denorm[1];
      }
    } catch (err) {
      console.warn('[ONNX Service] Transformer inference error:', err);
    }

    const latencyMs = Math.max(0.01, performance.now() - t0);
    return {
      dx: predDx,
      dy: predDy,
      magnitude: Math.hypot(predDx, predDy),
      latencyMs
    };
  }
}

export const onnxInferenceService = new ONNXInferenceService();
