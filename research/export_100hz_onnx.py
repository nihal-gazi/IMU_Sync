"""
Export 100Hz High-Quality IMU Models to ONNX for Browser / Mobile WebAssembly Deployment
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "public", "models"))
RESEARCH_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "models"))
EXP1_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "experiment", "exp_1", "models"))
EXP2_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "experiment", "exp_2", "models"))

os.makedirs(PUBLIC_MODELS_DIR, exist_ok=True)
os.makedirs(RESEARCH_MODELS_DIR, exist_ok=True)

sys.path.append(os.path.join(ROOT_DIR, "experiment", "exp_1"))
sys.path.append(os.path.join(ROOT_DIR, "experiment", "exp_2"))

from run_exp1 import RestMovingClassifierMLP, IMUTransformerTLIO
from run_exp2 import UnifiedIMUTransformer


def export_models_to_onnx():
    print("[ONNX Export] Deploying 100Hz models to ONNX WebAssembly...")
    dummy_input = torch.randn(1, 10, 6, dtype=torch.float32)

    # 1. Export Unified Multi-Task Transformer (Experiment 2)
    exp2_pt = os.path.join(EXP2_MODELS_DIR, "unified_transformer.pt")
    unified_onnx_pub = os.path.join(PUBLIC_MODELS_DIR, "unified_transformer.onnx")
    unified_onnx_res = os.path.join(RESEARCH_MODELS_DIR, "unified_transformer.onnx")

    unified_model = UnifiedIMUTransformer(input_dim=6, window_size=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0)
    if os.path.exists(exp2_pt):
        unified_model.load_state_dict(torch.load(exp2_pt, map_location="cpu"))
        print(f"[ONNX Export] Loaded {exp2_pt}")
    unified_model.eval()

    torch.onnx.export(
        unified_model,
        dummy_input,
        unified_onnx_pub,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['motion_logits', 'acceleration_2d'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'motion_logits': {0: 'batch_size'}, 'acceleration_2d': {0: 'batch_size'}}
    )
    torch.onnx.export(
        unified_model,
        dummy_input,
        unified_onnx_res,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['motion_logits', 'acceleration_2d'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'motion_logits': {0: 'batch_size'}, 'acceleration_2d': {0: 'batch_size'}}
    )
    print(f"[ONNX Export] Exported unified_transformer.onnx ({os.path.getsize(unified_onnx_pub):,} bytes)")

    # 2. Export 100Hz Motion Classifier (Stage 1)
    cls_pt = os.path.join(EXP1_MODELS_DIR, "motion_classifier.pt")
    cls_onnx_pub = os.path.join(PUBLIC_MODELS_DIR, "motion_classifier.onnx")
    cls_onnx_res = os.path.join(RESEARCH_MODELS_DIR, "motion_classifier.onnx")

    cls_model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64)
    if os.path.exists(cls_pt):
        cls_model.load_state_dict(torch.load(cls_pt, map_location="cpu"))
    cls_model.eval()

    torch.onnx.export(
        cls_model,
        dummy_input,
        cls_onnx_pub,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['motion_logits'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'motion_logits': {0: 'batch_size'}}
    )
    torch.onnx.export(
        cls_model,
        dummy_input,
        cls_onnx_res,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['motion_logits'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'motion_logits': {0: 'batch_size'}}
    )
    print(f"[ONNX Export] Exported motion_classifier.onnx ({os.path.getsize(cls_onnx_pub):,} bytes)")

    # 3. Export 100Hz IMU Transformer (Stage 2)
    trans_pt = os.path.join(EXP1_MODELS_DIR, "tlio_transformer.pt")
    trans_onnx_pub = os.path.join(PUBLIC_MODELS_DIR, "tlio_transformer.onnx")
    trans_onnx_res = os.path.join(RESEARCH_MODELS_DIR, "tlio_transformer.onnx")

    trans_model = IMUTransformerTLIO(input_dim=6, window_size=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0, output_dim=2)
    if os.path.exists(trans_pt):
        trans_model.load_state_dict(torch.load(trans_pt, map_location="cpu"))
    trans_model.eval()

    torch.onnx.export(
        trans_model,
        dummy_input,
        trans_onnx_pub,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['displacement_1s'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'displacement_1s': {0: 'batch_size'}}
    )
    torch.onnx.export(
        trans_model,
        dummy_input,
        trans_onnx_res,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['displacement_1s'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'displacement_1s': {0: 'batch_size'}}
    )
    print(f"[ONNX Export] Exported tlio_transformer.onnx ({os.path.getsize(trans_onnx_pub):,} bytes)")

    # 4. Scaler Parameters
    scaler_data = {
        "features": {
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            "mean": [0.0, 0.0, 9.81, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 0.50, 0.05, 0.05, 0.05]
        },
        "targets": {
            "names": ["a_lateral", "a_forward"],
            "mean": [0.0, 0.0],
            "std": [1.0, 0.85]
        },
        "calibration": {
            "k_accel": 0.9000,
            "k_gyro": 0.9850
        }
    }

    scaler_pub = os.path.join(PUBLIC_MODELS_DIR, "scaler_params.json")
    scaler_res = os.path.join(RESEARCH_MODELS_DIR, "scaler_params.json")
    with open(scaler_pub, "w") as f:
        json.dump(scaler_data, f, indent=2)
    with open(scaler_res, "w") as f:
        json.dump(scaler_data, f, indent=2)
    print(f"[ONNX Export] Exported scaler_params.json with calibration factors.")

    # 5. Verify ONNX Runtime Execution
    import onnxruntime as ort
    ort_session = ort.InferenceSession(unified_onnx_pub, providers=['CPUExecutionProvider'])
    test_out = ort_session.run(None, {'input_window': dummy_input.numpy()})
    print(f"[ONNX Verification] Unified Transformer forward pass succeeded! Output shapes: {[o.shape for o in test_out]}")


if __name__ == "__main__":
    export_models_to_onnx()
