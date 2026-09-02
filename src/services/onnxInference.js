/**
 * ONNX Runtime Web Inference Service for 100Hz Unified Multi-Task Transformer
 * Single Forward Pass -> [is_rest, is_moving] logits + [a_x, a_y] dual accelerations
 * With Fallback to 2-Stage [motion_classifier.onnx + tlio_transformer.onnx]
 */

import * as ort from 'onnxruntime-web';

class ONNXInferenceService {
  constructor() {
    this.sessionUnified = null;
    this.sessionClassifier = null;
    this.sessionTransformer = null;
    this.isReady = false;
    this.mode = 'unified'; // 'unified' | 'tlio'

    this.scalers = {
      features: {
        names: ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
        mean: [0.0, 0.0, 9.81, 0.0, 0.0, 0.0],
        std: [1.0, 1.0, 0.50, 0.05, 0.05, 0.05]
      },
      targets: {
        names: ['a_lateral', 'a_forward'],
        mean: [0.0, 0.0],
        std: [1.0, 0.85]
      },
      calibration: {
        k_accel: 0.9000,
        k_gyro: 0.9850
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
            console.log('[ONNX Service] 100Hz Scaler parameters loaded with calibration factors:', this.scalers.calibration);
          }
        }
      } catch (e) {
        console.warn('[ONNX Service] Using fallback scalers:', e);
      }

      // 2. Load Unified Multi-Task Transformer (Primary Engine)
      try {
        const unifiedRes = await fetch(`${modelsBasePath}/unified_transformer.onnx`);
        if (unifiedRes.ok) {
          const buffer = await unifiedRes.arrayBuffer();
          this.sessionUnified = await ort.InferenceSession.create(new Uint8Array(buffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Unified Multi-Task Transformer ONNX loaded (Primary Engine).');
        }
      } catch (err) {
        console.warn('[ONNX Service] Unified Transformer loading fallback:', err);
      }

      // 3. Fallback: Stage 1 Motion Classifier (MLP)
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

      // 4. Fallback: Stage 2 Acceleration Transformer
      try {
        const transformerRes = await fetch(`${modelsBasePath}/tlio_transformer.onnx`);
        if (transformerRes.ok) {
          const transBuffer = await transformerRes.arrayBuffer();
          this.sessionTransformer = await ort.InferenceSession.create(new Uint8Array(transBuffer), {
            executionProviders: ['wasm']
          });
          console.log('[ONNX Service] Stage 2 Acceleration Transformer ONNX loaded.');
        }
      } catch (err) {
        console.warn('[ONNX Service] Transformer loading error:', err);
      }

      this.isReady = !!(this.sessionUnified || (this.sessionClassifier && this.sessionTransformer));
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
   * Runs 100Hz Unified Multi-Task Acceleration & Motion Inference
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
    const kAccel = (this.scalers.calibration && this.scalers.calibration.k_accel) ? this.scalers.calibration.k_accel : 1.0;

    try {
      const inputTensor = new ort.Tensor('float32', flatNormWindow, [1, windowSize, 6]);

      // 1. PRIMARY: Single Unified Multi-Task Transformer
      if (this.sessionUnified) {
        const feeds = { input_window: inputTensor };
        const results = await this.sessionUnified.run(feeds);
        
        // Output 1: Motion Classification Logits [is_rest, is_moving]
        const logits = results.motion_logits.data;
        isMoving = logits[1] > logits[0];

        // Output 2: 2D Physical Accelerations [a_x (lateral), a_y (forward)]
        if (isMoving) {
          const accData = results.acceleration_2d.data;
          const denorm = this.denormalizeAcceleration([accData[0], accData[1]]);
          aLat = denorm[0];
          aFwd = denorm[1] * kAccel;
        } else {
          aLat = 0.0;
          aFwd = 0.0;
        }
      } 
      // 2. FALLBACK: 2-Stage Sequential Models
      else {
        if (this.sessionClassifier) {
          const clsFeeds = { input_window: inputTensor };
          const clsResults = await this.sessionClassifier.run(clsFeeds);
          const logits = clsResults.motion_logits.data;
          isMoving = logits[1] > logits[0];
        }

        if (isMoving && this.sessionTransformer) {
          const feeds = { input_window: inputTensor };
          const results = await this.sessionTransformer.run(feeds);
          const outData = results.displacement_1s.data;
          const denorm = this.denormalizeAcceleration([outData[0], outData[1]]);
          aFwd = denorm[1] * kAccel;
        } else {
          aFwd = 0.0;
        }
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
