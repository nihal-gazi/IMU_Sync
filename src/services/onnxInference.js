/**
 * ONNX Runtime Web Inference Service for IMU-Sync
 * Manages ONNX sessions for SimpleRNN (stateful) and SimpleMLP (stateless),
 * handles input normalization, and computes dead-reckoning vectors.
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

    // Default normalization scalers (from IO-VNBD dataset)
    this.scalers = {
      features: {
        mean: [0.042, 0.062, 9.847, 0.002, -0.007, 0.002],
        std: [0.954, 0.887, 1.214, 0.124, 0.098, 0.089]
      },
      targets: {
        mean: [0.12, 1.84, 2.06, 0.05, 0.89],
        std: [1.25, 2.10, 2.15, 0.70, 0.70]
      }
    };
  }

  async init(modelsBasePath = '/models') {
    try {
      // Configure ONNX WebAssembly environment
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

      // 2. Load RNN ONNX Model
      const rnnPath = `${modelsBasePath}/rnn_model.onnx`;
      this.sessionRNN = await ort.InferenceSession.create(rnnPath, {
        executionProviders: ['wasm']
      });
      console.log('[ONNX Service] SimpleRNN ONNX session initialized.');

      // 3. Load MLP ONNX Model
      const mlpPath = `${modelsBasePath}/mlp_model.onnx`;
      this.sessionMLP = await ort.InferenceSession.create(mlpPath, {
        executionProviders: ['wasm']
      });
      console.log('[ONNX Service] SimpleMLP ONNX session initialized.');

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
    const out = new Float32Array(5);
    for (let i = 0; i < 5; i++) {
      out[i] = rawY[i] * (std[i] || 1.0) + mean[i];
    }
    return out;
  }

  /**
   * Runs single millisecond/step inference
   * @param {Array<number>} rawImu - [ax, ay, az, gx, gy, gz]
   * @param {number} dt - sample time delta in seconds (nominal 0.1s)
   */
  async predictStep(rawImu, dt = 0.1) {
    const t0 = performance.now();
    const xNorm = this.normalizeInput(rawImu);
    let rawPred = new Float32Array(5);

    try {
      if (this.isReady && this.mode === 'rnn' && this.sessionRNN) {
        // Create ONNX Tensors
        const tensorX = new ort.Tensor('float32', xNorm, [1, 6]);
        const tensorH = new ort.Tensor('float32', this.hiddenState, [1, 32]);

        const feeds = { input_imu: tensorX, h_prev: tensorH };
        const results = await this.sessionRNN.run(feeds);

        const outVector = results.vector_output.data;
        const outHNext = results.h_next.data;

        // Copy back hidden state
        this.hiddenState.set(outHNext);
        for (let i = 0; i < 5; i++) rawPred[i] = outVector[i];
      } else if (this.isReady && this.mode === 'mlp' && this.sessionMLP) {
        const tensorX = new ort.Tensor('float32', xNorm, [1, 6]);
        const feeds = { input_imu: tensorX };
        const results = await this.sessionMLP.run(feeds);

        const outVector = results.vector_output.data;
        for (let i = 0; i < 5; i++) rawPred[i] = outVector[i];
      } else {
        // Fast analytical fallback if session loading
        const [ax, ay, az, gx, gy, gz] = rawImu;
        const speed = Math.max(0, Math.sqrt(ax * ax + ay * ay) * 0.8);
        const heading = Math.atan2(ay, ax);
        rawPred[0] = speed * Math.sin(heading);
        rawPred[1] = speed * Math.cos(heading);
        rawPred[2] = speed;
        rawPred[3] = Math.sin(heading);
        rawPred[4] = Math.cos(heading);
      }
    } catch (e) {
      console.warn('[ONNX Service] Inference step error:', e);
    }

    const denorm = this.denormalizeOutput(rawPred);
    const vx = denorm[0];
    const vy = denorm[1];
    const speed = Math.max(0, denorm[2]);

    let headingRad = Math.atan2(vx, vy);
    if (headingRad < 0) headingRad += 2 * Math.PI;
    const headingDeg = (headingRad * 180) / Math.PI;

    const dx = vx * dt;
    const dy = vy * dt;
    const latencyMs = Math.max(0.01, performance.now() - t0);

    return {
      vx,
      vy,
      speed,
      speedKmh: speed * 3.6,
      headingDeg,
      headingRad,
      dx,
      dy,
      hiddenState: this.hiddenState,
      latencyMs
    };
  }
}

export const onnxInferenceService = new ONNXInferenceService();
