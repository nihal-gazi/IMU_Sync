"""
ONNX Model Exporter for 2-Stage Motion System
Exports:
1. RestMovingClassifierMLP -> public/models/motion_classifier.onnx
2. IMUTransformerTLIO -> public/models/tlio_transformer.onnx
3. SimpleRNN & SimpleMLP baseline models
"""

import os
import sys
import json
import shutil
import torch
import torch.nn as nn
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models import RestMovingClassifierMLP, IMUTransformerTLIO, SimpleRNN, SimpleMLP, SimpleRNNCell

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_MODELS_DIR = os.path.join(RESEARCH_DIR, "..", "public", "models")


class StatefulRNNStepWrapper(nn.Module):
    def __init__(self, cell: SimpleRNNCell):
        super().__init__()
        self.cell = cell

    def forward(self, input_imu: torch.Tensor, h_prev: torch.Tensor):
        return self.cell(input_imu, h_prev)


def export_models():
    os.makedirs(PUBLIC_MODELS_DIR, exist_ok=True)

    # 1. Export Rest vs Moving Classifier MLP
    print("\n--- Exporting RestMovingClassifierMLP to Self-Contained ONNX ---")
    cls_model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64)
    cls_pt = os.path.join(RESEARCH_DIR, "motion_classifier.pt")
    if os.path.exists(cls_pt):
        cls_model.load_state_dict(torch.load(cls_pt, map_location="cpu"))
        print(f"Loaded trained Motion Classifier weights from {cls_pt}")
    cls_model.eval()

    dummy_window = torch.randn(1, 10, 6, dtype=torch.float32)
    cls_onnx_path = os.path.join(PUBLIC_MODELS_DIR, "motion_classifier.onnx")

    torch.onnx.export(
        cls_model,
        dummy_window,
        cls_onnx_path,
        input_names=["input_window"],
        output_names=["motion_logits"],
        dynamic_axes={
            "input_window": {0: "batch_size"},
            "motion_logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        dynamo=False
    )
    print(f"Exported Motion Classifier model to {cls_onnx_path} ({os.path.getsize(cls_onnx_path):,} bytes)")

    # 2. Export IMUTransformerTLIO
    print("\n--- Exporting IMUTransformerTLIO to Self-Contained ONNX ---")
    transformer_model = IMUTransformerTLIO(
        input_dim=6,
        window_size=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        output_dim=2
    )
    transformer_pt = os.path.join(RESEARCH_DIR, "tlio_transformer.pt")
    if os.path.exists(transformer_pt):
        transformer_model.load_state_dict(torch.load(transformer_pt, map_location="cpu"))
        print(f"Loaded trained Transformer weights from {transformer_pt}")
    transformer_model.eval()

    transformer_onnx_path = os.path.join(PUBLIC_MODELS_DIR, "tlio_transformer.onnx")
    torch.onnx.export(
        transformer_model,
        dummy_window,
        transformer_onnx_path,
        input_names=["input_window"],
        output_names=["displacement_1s"],
        dynamic_axes={
            "input_window": {0: "batch_size"},
            "displacement_1s": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        dynamo=False
    )
    print(f"Exported TLIO Transformer model to {transformer_onnx_path} ({os.path.getsize(transformer_onnx_path):,} bytes)")

    # 3. Export SimpleRNN
    rnn_model = SimpleRNN(input_size=6, hidden_size=32, output_size=2)
    step_wrapper = StatefulRNNStepWrapper(rnn_model.cell)
    step_wrapper.eval()
    dummy_x = torch.randn(1, 6, dtype=torch.float32)
    dummy_h = torch.zeros(1, 32, dtype=torch.float32)
    rnn_onnx_path = os.path.join(PUBLIC_MODELS_DIR, "rnn_model.onnx")
    torch.onnx.export(
        step_wrapper,
        (dummy_x, dummy_h),
        rnn_onnx_path,
        input_names=["input_imu", "h_prev"],
        output_names=["vector_output", "h_next"],
        dynamic_axes={
            "input_imu": {0: "batch_size"},
            "h_prev": {0: "batch_size"},
            "vector_output": {0: "batch_size"},
            "h_next": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        dynamo=False
    )

    # 4. Copy scaler parameters
    scaler_src = os.path.join(RESEARCH_DIR, "scaler_params.json")
    scaler_dst = os.path.join(PUBLIC_MODELS_DIR, "scaler_params.json")
    if os.path.exists(scaler_src):
        shutil.copy(scaler_src, scaler_dst)
        print(f"Copied scaler params to {scaler_dst}")


if __name__ == "__main__":
    export_models()
