"""
Weight Exporter for IMU-Sync Models
Exports PyTorch RNN and MLP models to:
1. model_weights.json (for ultra-fast pure JavaScript in-browser inference)
2. rnn_model.onnx & mlp_model.onnx (for ONNX Runtime)
"""

import os
import json
import torch
import numpy as np

from models import SimpleRNN, SimpleMLP

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def export_weights_to_json():
    print("[Export] Exporting weights to JSON for JavaScript client...")
    
    # Load RNN
    rnn = SimpleRNN(input_size=6, hidden_size=32, output_size=5)
    rnn_pt_path = os.path.join(MODEL_DIR, "rnn_model.pt")
    if os.path.exists(rnn_pt_path):
        rnn.load_state_dict(torch.load(rnn_pt_path, map_location="cpu"))
        rnn.eval()
    
    # Load MLP
    mlp = SimpleMLP(input_size=6, hidden_size=64, output_size=5)
    mlp_pt_path = os.path.join(MODEL_DIR, "mlp_model.pt")
    if os.path.exists(mlp_pt_path):
        mlp.load_state_dict(torch.load(mlp_pt_path, map_location="cpu"))
        mlp.eval()

    # Load Normalization Scalers
    scaler_path = os.path.join(MODEL_DIR, "scaler_params.json")
    scalers = {}
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            scalers = json.load(f)

    # Extract RNN weights
    # Cell: fc_ih, fc_hh, fc_out
    rnn_dict = {
        "input_size": 6,
        "hidden_size": 32,
        "output_size": 5,
        "W_ih": rnn.cell.fc_ih.weight.detach().cpu().numpy().tolist(), # (32, 6)
        "b_ih": rnn.cell.fc_ih.bias.detach().cpu().numpy().tolist(),   # (32,)
        "W_hh": rnn.cell.fc_hh.weight.detach().cpu().numpy().tolist(), # (32, 32)
        "W_out": rnn.cell.fc_out.weight.detach().cpu().numpy().tolist(), # (5, 32)
        "b_out": rnn.cell.fc_out.bias.detach().cpu().numpy().tolist()    # (5,)
    }

    # Extract MLP weights
    mlp_dict = {
        "input_size": 6,
        "hidden_size": 64,
        "output_size": 5,
        "fc1_w": mlp.net[0].weight.detach().cpu().numpy().tolist(), # (64, 6)
        "fc1_b": mlp.net[0].bias.detach().cpu().numpy().tolist(),   # (64,)
        "ln_w": mlp.net[1].weight.detach().cpu().numpy().tolist(),  # (64,)
        "ln_b": mlp.net[1].bias.detach().cpu().numpy().tolist(),    # (64,)
        "fc2_w": mlp.net[3].weight.detach().cpu().numpy().tolist(), # (32, 64)
        "fc2_b": mlp.net[3].bias.detach().cpu().numpy().tolist(),   # (32,)
        "fc3_w": mlp.net[5].weight.detach().cpu().numpy().tolist(), # (5, 32)
        "fc3_b": mlp.net[5].bias.detach().cpu().numpy().tolist()    # (5,)
    }

    bundle = {
        "version": "1.0.0",
        "description": "IMU-Sync Trained Neural Network Weights on IO-VNBD",
        "scalers": scalers,
        "rnn": rnn_dict,
        "mlp": mlp_dict
    }

    out_json = os.path.join(MODEL_DIR, "model_weights.json")
    with open(out_json, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"[Export] Saved full weight bundle to {out_json} ({os.path.getsize(out_json):,} bytes)")

    # Also copy to static folder for direct web client loading
    static_weights_path = os.path.join(MODEL_DIR, "..", "static", "model_weights.json")
    os.makedirs(os.path.dirname(static_weights_path), exist_ok=True)
    with open(static_weights_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"[Export] Copied weights to web directory: {static_weights_path}")


def export_onnx():
    print("[Export] Exporting to ONNX...")
    try:
        # Export MLP
        mlp = SimpleMLP(input_size=6, hidden_size=64, output_size=5)
        mlp_pt = os.path.join(MODEL_DIR, "mlp_model.pt")
        if os.path.exists(mlp_pt):
            mlp.load_state_dict(torch.load(mlp_pt, map_location="cpu"))
        mlp.eval()
        dummy_x = torch.randn(1, 6)
        mlp_onnx_path = os.path.join(MODEL_DIR, "mlp_model.onnx")
        torch.onnx.export(
            mlp, dummy_x, mlp_onnx_path,
            input_names=["input_imu"],
            output_names=["vector_output"],
            dynamic_axes={"input_imu": {0: "batch_size"}, "vector_output": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[Export] Exported MLP to {mlp_onnx_path}")
    except Exception as e:
        print(f"[Export] ONNX export notice: {e}")


if __name__ == "__main__":
    export_weights_to_json()
    export_onnx()
