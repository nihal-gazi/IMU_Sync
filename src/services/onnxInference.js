/**
 * ONNX Runtime Web Inference Service for IMU-Sync
 * Model Target Outputs: [vx, vy] (2 outputs)
 * Direct Velocity Prediction: Predicts 2D velocity vector [vx, vy] directly from 6-axis IMU.
 */

import * as ort from 'onnxruntime-web';

class ONNXInferenceService {
  constructor() {
    this.sessionRNN = null;
    this.sessionMLP = null;
    this.isReady = false;
    this.mode = 'rnn'; // 'rnn' | 'mlp'

    // RNN Hidden state tensor (1 x 32)
    this.hiddenDim = 32;
    this.hiddenState = new Float32Array(this.hiddenDim);

    // Normalization Scalers for 6 inputs and 2 targets [vx, vy]
    this.scalers = {
      features: {
        names: ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
        mean: [0.042, 0.062, 9.847, 0.002, -0.007, 0.002],
        std: [0.954, 0.887, 1.214, 0.124, 0.098, 0.089]
      },
      targets: {
        names: ['vx', 'vy'],
        mean: [0.12, 1.84],
        std: [1.25, 2.10]
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
            console.log('[ONNX Service] Scaler parameters loaded for [vx, vy].');
          }
        }
      } catch (e) {
        console.warn('[ONNX Service] Using fallback scalers:', e);
      }

      // 2. Fetch and Load RNN ONNX Model as Uint8Array Memory Buffer
      const rnnRes = await fetch(`${modelsBasePath}/rnn_model.onnx`);
      if (!rnnRes.ok) throw new Error(`HTTP ${rnnRes.status} fetching rnn_model.onnx`);
      const rnnBuffer = await rnnRes.arrayBuffer();
      this.sessionRNN = await ort.InferenceSession.create(new Uint8Array(rnnBuffer), {
        executionProviders: ['wasm']
      });
      console.log('[ONNX Service] SimpleRNN ONNX session initialized ([vx, vy]).');

      // 3. Fetch and Load MLP ONNX Model as Uint8Array Memory Buffer
      const mlpRes = await fetch(`${modelsBasePath}/mlp_model.onnx`);
      if (!mlpRes.ok) throw new Error(`HTTP ${mlpRes.status} fetching mlp_model.onnx`);
      const mlpBuffer = await mlpRes.arrayBuffer();
      this.sessionMLP = await ort.InferenceSession.create(new Uint8Array(mlpBuffer), {
        executionProviders: ['wasm']
      });
      console.log('[ONNX Service] SimpleMLP ONNX session initialized ([vx, vy]).');

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
    if (mode === 'rnn' || mode === 'mlp') {
      this.mode = mode;
      if (mode === 'rnn') this.resetHiddenState();
    }
  }

  normalizeInput(raw) {
    const mean = this.scalers.features.mean;
    const std = this.scalers.features.std;
    const norm = new Float32Array(6);
    for (let i = 0; i < 6; i++) {
      norm[i] = (raw[i] - mean[i]) / (std[i] || 1.0);
    }
    return norm;
  }

  denormalizeOutput(rawY) {
    const mean = this.scalers.targets.mean;
    const std = this.scalers.targets.std;
    const out = new Float32Array(2);
    for (let i = 0; i < 2; i++) {
      out[i] = rawY[i] * (std[i] || 1.0) + mean[i];
    }
    return out;
  }

  /**
   * Runs single step inference
   * Predicts [vx, vy] directly from sensor data.
   * @param {Array<number>} rawImu - [ax, ay, az, gx, gy, gz]
   */
  async predictStep(rawImu) {
    const t0 = performance.now();
    const xNorm = this.normalizeInput(rawImu);
    let rawPred = new Float32Array(2);

    try {
      if (this.isReady && this.mode === 'rnn' && this.sessionRNN) {
        const tensorX = new ort.Tensor('float32', xNorm, [1, 6]);
        const tensorH = new ort.Tensor('float32', this.hiddenState, [1, 32]);

        const feeds = { input_imu: tensorX, h_prev: tensorH };
        const results = await this.sessionRNN.run(feeds);

        const outVector = results.vector_output.data;
        const outHNext = results.h_next.data;

        this.hiddenState.set(outHNext);
        rawPred[0] = outVector[0];
        rawPred[1] = outVector[1];
      } else if (this.isReady && this.mode === 'mlp' && this.sessionMLP) {
        const tensorX = new ort.Tensor('float32', xNorm, [1, 6]);
        const feeds = { input_imu: tensorX };
        const results = await this.sessionMLP.run(feeds);

        const outVector = results.vector_output.data;
        rawPred[0] = outVector[0];
        rawPred[1] = outVector[1];
      } else {
        const [ax, ay, az, gx, gy, gz] = rawImu;
        const spd = Math.max(0, Math.sqrt(ax * ax + ay * ay) * 0.5);
        rawPred[0] = ax * 0.3;
        rawPred[1] = ay * 0.3;
      }
    } catch (e) {
      console.warn('[ONNX Service] Inference step error:', e);
    }

    const denorm = this.denormalizeOutput(rawPred);
    const vx = denorm[0];
    const vy = denorm[1];
    const speed = Math.hypot(vx, vy);
    const latencyMs = Math.max(0.01, performance.now() - t0);

    return {
      vx,
      vy,
      speed,
      speedKmh: speed * 3.6,
      hiddenState: this.hiddenState,
      scalers: this.scalers,
      latencyMs
    };
  }
}

export const onnxInferenceService = new ONNXInferenceService();
