"""
Export trained ResidualDriftTransformer PyTorch model to ONNX.
Uses dynamo=False and UTF-8 stdout to avoid Windows charmap encoding errors.
"""

import os
import sys
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class ResidualDriftTransformer(nn.Module):
    def __init__(self, in_features=6, seq_len=60, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.05,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embedding
        h = self.transformer_encoder(h)
        h = self.norm(h)
        pool = torch.mean(h, dim=1)
        out = self.head(pool)
        return out

def export():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(exp_dir, "models", "best_residual_transformer.pt")
    onnx_exp = os.path.join(exp_dir, "models", "sih_rect_transformer.onnx")
    onnx_public = os.path.abspath(os.path.join(exp_dir, "..", "..", "..", "public", "models", "sih_rect_transformer.onnx"))

    print(f"Loading checkpoint: {model_path}")
    model = ResidualDriftTransformer()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 60, 6, dtype=torch.float32)

    # Export using standard TorchScript-based ONNX exporter
    torch.onnx.export(
        model,
        dummy_input,
        onnx_exp,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['imu_window_60hz'],
        output_names=['residual_corrections'],
        dynamic_axes={
            'imu_window_60hz': {0: 'batch_size'},
            'residual_corrections': {0: 'batch_size'}
        },
        dynamo=False
    )
    
    import shutil
    shutil.copyfile(onnx_exp, onnx_public)
    print(f"ONNX Model successfully saved to:")
    print(f"  Exp:    {onnx_exp} ({os.path.getsize(onnx_exp)} bytes)")
    print(f"  Public: {onnx_public} ({os.path.getsize(onnx_public)} bytes)")

    # Verify ONNX Runtime session
    sess = ort.InferenceSession(onnx_public, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: dummy_input.numpy()})
    print(f"Verification run successful! Output shape: {out[0].shape}, values: {out[0][0]}")

if __name__ == '__main__':
    export()
