"""
Deploy Experiment 1 Models to public/models as Self-Contained ONNX Files
Exports:
  1. motion_classifier.onnx (Exp 1 Stage 1 ZUPT Classifier)
  2. tlio_transformer.onnx (Exp 1 Stage 2 Acceleration Estimator)
  3. scaler_params.json (Exp 1 Scaler & Calibration Parameters: k_accel=1.1379, k_gyro=0.9794)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx.external_data_helper import load_external_data_for_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "public", "models"))
EXP1_MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "experiment", "exp_1", "models"))
EXP1_DIR = os.path.abspath(os.path.join(ROOT_DIR, "experiment", "exp_1"))

sys.path.append(EXP1_DIR)
from run_exp1 import RestMovingClassifierMLP, IMUTransformerTLIO, load_and_preprocess_100hz_data


def export_and_deploy_exp1():
    print(f"[Deploy Exp1] Exporting Experiment 1 models from {EXP1_MODELS_DIR} to {PUBLIC_MODELS_DIR}...")
    dummy_input = torch.randn(1, 10, 6, dtype=torch.float32)

    # 1. Export Stage 1 Motion Classifier (Exp 1)
    cls_pt = os.path.join(EXP1_MODELS_DIR, "motion_classifier.pt")
    cls_onnx = os.path.join(PUBLIC_MODELS_DIR, "motion_classifier.onnx")
    cls_data = cls_onnx + ".data"

    cls_model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64)
    cls_model.load_state_dict(torch.load(cls_pt, map_location="cpu"))
    cls_model.eval()

    torch.onnx.export(
        cls_model,
        dummy_input,
        cls_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['motion_logits'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'motion_logits': {0: 'batch_size'}}
    )

    # Ensure 100% self-contained
    cls_onnx_proto = onnx.load(cls_onnx)
    load_external_data_for_model(cls_onnx_proto, PUBLIC_MODELS_DIR)
    onnx.save_model(cls_onnx_proto, cls_onnx, save_as_external_data=False)
    if os.path.exists(cls_data): os.remove(cls_data)
    print(f"[Deploy Exp1] Saved self-contained motion_classifier.onnx ({os.path.getsize(cls_onnx):,} bytes)")

    # 2. Export Stage 2 Acceleration Transformer (Exp 1)
    trans_pt = os.path.join(EXP1_MODELS_DIR, "tlio_transformer.pt")
    trans_onnx = os.path.join(PUBLIC_MODELS_DIR, "tlio_transformer.onnx")
    trans_data = trans_onnx + ".data"

    trans_model = IMUTransformerTLIO(input_dim=6, window_size=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0, output_dim=2)
    trans_model.load_state_dict(torch.load(trans_pt, map_location="cpu"))
    trans_model.eval()

    torch.onnx.export(
        trans_model,
        dummy_input,
        trans_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_window'],
        output_names=['displacement_1s'],
        dynamic_axes={'input_window': {0: 'batch_size'}, 'displacement_1s': {0: 'batch_size'}}
    )

    # Ensure 100% self-contained
    trans_onnx_proto = onnx.load(trans_onnx)
    load_external_data_for_model(trans_onnx_proto, PUBLIC_MODELS_DIR)
    onnx.save_model(trans_onnx_proto, trans_onnx, save_as_external_data=False)
    if os.path.exists(trans_data): os.remove(trans_data)
    print(f"[Deploy Exp1] Saved self-contained tlio_transformer.onnx ({os.path.getsize(trans_onnx):,} bytes)")

    # 3. Load dataset to get exact feature & target normalizer scalers
    data = load_and_preprocess_100hz_data()
    scaler_dict = {
        "features": {
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            "mean": data['feat_norm'].mean.tolist(),
            "std": data['feat_norm'].std.tolist()
        },
        "targets": {
            "names": ["dv_lateral", "dv_forward"],
            "mean": data['target_norm'].mean.tolist(),
            "std": data['target_norm'].std.tolist()
        },
        "calibration": {
            "k_accel": 1.1379,
            "k_gyro": 0.9794
        }
    }

    scaler_path = os.path.join(PUBLIC_MODELS_DIR, "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"[Deploy Exp1] Saved scaler_params.json with Exp 1 calibration factors (k_accel=1.1379, k_gyro=0.9794).")

    # 4. Verify with ONNX Runtime
    import onnxruntime as ort
    sess_cls = ort.InferenceSession(cls_onnx, providers=['CPUExecutionProvider'])
    sess_trans = ort.InferenceSession(trans_onnx, providers=['CPUExecutionProvider'])
    
    out_c = sess_cls.run(None, {'input_window': dummy_input.numpy()})
    out_t = sess_trans.run(None, {'input_window': dummy_input.numpy()})
    print(f"[Deploy Exp1 Verification] Stage 1 output shape: {out_c[0].shape}, Stage 2 output shape: {out_t[0].shape}")


if __name__ == "__main__":
    export_and_deploy_exp1()
