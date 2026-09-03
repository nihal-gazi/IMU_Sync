"""
Experiment 1: SIH-Rect Residual Drift Transformer
Trains a 1.0-second (60 samples @ 60Hz) Transformer to predict the residual drift error
between SIH MLP predictions and ground truth, dynamically rectifying odometry.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import onnxruntime as ort
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data", "resampled_60hz")
MODELS_DIR = os.path.join(EXP_DIR, "models")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
SIH_MODEL_PATH = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "..", "public", "models", "inertial_mlp.onnx"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1D Discrete Gaussian Filter (matching SIH k=7, sigma=1.2)
def apply_gaussian_filter(imu_data, kernel_size=7, sigma=1.2):
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = k / k.sum()
    
    smoothed = np.zeros_like(imu_data)
    for c in range(imu_data.shape[1]):
        smoothed[:, c] = np.convolve(imu_data[:, c], kernel, mode='same')
    return smoothed

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
        
        # Head predicting residual drift corrections: [delta_displacement, delta_speed]
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x: [B, seq_len, in_features]
        h = self.input_proj(x) + self.pos_embedding
        h = self.transformer_encoder(h)
        h = self.norm(h)
        # Global Average Pooling across time
        pool = torch.mean(h, dim=1)
        out = self.head(pool) # [B, 2] -> [delta_d, delta_v]
        return out

def build_dataset(seq_len=60, step_interval=30):
    """
    seq_len = 60 (1.0s @ 60Hz)
    step_interval = 30 (0.5s evaluation interval)
    """
    sih_sess = ort.InferenceSession(SIH_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_60hz_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 60Hz files found in {DATA_DIR}")
        
    all_X = []
    all_Y_residual = []
    all_gt = []
    all_sih = []
    
    for f in csv_files:
        df = pd.read_csv(f)
        raw_imu = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
        
        # Smooth IMU using 6-DOF Gaussian filter
        smoothed_imu = apply_gaussian_filter(raw_imu, kernel_size=7, sigma=1.2)
        
        # SIH channel order: [ax, ay, az, gz, gx, gy]
        sih_channels = np.stack([
            smoothed_imu[:, 0], # ax
            smoothed_imu[:, 1], # ay
            smoothed_imu[:, 2], # az
            smoothed_imu[:, 5], # gz
            smoothed_imu[:, 3], # gx
            smoothed_imu[:, 4], # gy
        ], axis=1)
        
        pos_x = df['pos_x'].values
        pos_y = df['pos_y'].values
        speed_mps = df['speed_mps'].values
        
        n_samples = len(df)
        stride = 5  # sliding step
        
        for i in range(0, n_samples - seq_len - step_interval, stride):
            # 1.0s input window for Transformer
            x_win = raw_imu[i : i + seq_len]
            
            # Ground truth over the 0.5s step interval (at end of window)
            idx_start = i + seq_len
            idx_end = idx_start + step_interval
            
            dx_gt = pos_x[idx_end] - pos_x[idx_start]
            dy_gt = pos_y[idx_end] - pos_y[idx_start]
            disp_gt = np.sqrt(dx_gt**2 + dy_gt**2)
            speed_gt = disp_gt / (step_interval / 60.0)
            
            # SIH model prediction on the latest 20 samples of the window
            sih_input = sih_channels[idx_start - 20 : idx_start].reshape(1, 20, 6)
            sih_res = sih_sess.run(None, {'imu_sequence': sih_input})[0][0]
            dx_sih, dy_sih = sih_res[0], sih_res[1]
            disp_sih = np.sqrt(dx_sih**2 + dy_sih**2)
            speed_sih = max(0.0, float(sih_res[2]))
            
            # Residual error (Drift value)
            res_disp = disp_gt - disp_sih
            res_speed = speed_gt - speed_sih
            
            all_X.append(x_win)
            all_Y_residual.append([res_disp, res_speed])
            all_gt.append([disp_gt, speed_gt])
            all_sih.append([disp_sih, speed_sih])
            
    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y_residual, dtype=np.float32)
    GT = np.array(all_gt, dtype=np.float32)
    SIH = np.array(all_sih, dtype=np.float32)
    
    print(f"Dataset generated: {len(X)} sequences of shape {X[0].shape}")
    print(f"Mean residual disp: {np.mean(Y[:, 0]):.4f} m, std: {np.std(Y[:, 0]):.4f} m")
    print(f"Mean residual speed: {np.mean(Y[:, 1]):.4f} m/s, std: {np.std(Y[:, 1]):.4f} m/s")
    return X, Y, GT, SIH

def main():
    X, Y, GT, SIH = build_dataset()
    
    # Feature Normalization (zero-mean unit-variance on 6 IMU channels)
    feat_mean = np.mean(X, axis=(0, 1))
    feat_std = np.std(X, axis=(0, 1))
    feat_std[feat_std < 1e-4] = 1.0
    X_norm = (X - feat_mean) / feat_std
    
    # Train / Val Split (85% / 15%)
    N = len(X)
    indices = np.random.RandomState(42).permutation(N)
    split = int(0.85 * N)
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, Y_train = X_norm[train_idx], Y[train_idx]
    X_val, Y_val = X_norm[val_idx], Y[val_idx]
    GT_val, SIH_val = GT[val_idx], SIH[val_idx]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = ResidualDriftTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.SmoothL1Loss()
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, "best_residual_transformer.pt")
    
    epochs = 25
    print("\n--- Starting Residual Transformer Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * len(bx)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.5f}")
    
    # -------------------------------------------------------------
    # Verification & Benchmark: Raw SIH vs SIH-Rectified
    # -------------------------------------------------------------
    model.load_state_dict(torch.save if not os.path.exists(best_model_path) else torch.load(best_model_path))
    model.eval()
    
    with torch.no_grad():
        val_preds = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
        
    pred_res_disp = val_preds[:, 0]
    pred_res_speed = val_preds[:, 1]
    
    # Raw SIH errors
    raw_sih_disp_err = np.abs(GT_val[:, 0] - SIH_val[:, 0])
    raw_sih_speed_err = np.abs(GT_val[:, 1] - SIH_val[:, 1])
    
    # Rectified SIH predictions:
    rect_sih_disp = np.maximum(0.0, SIH_val[:, 0] + pred_res_disp)
    rect_sih_speed = np.maximum(0.0, SIH_val[:, 1] + pred_res_speed)
    
    rect_sih_disp_err = np.abs(GT_val[:, 0] - rect_sih_disp)
    rect_sih_speed_err = np.abs(GT_val[:, 1] - rect_sih_speed)
    
    raw_disp_mae = np.mean(raw_sih_disp_err)
    rect_disp_mae = np.mean(rect_sih_disp_err)
    disp_reduction = (raw_disp_mae - rect_disp_mae) / raw_disp_mae * 100
    
    raw_speed_mae = np.mean(raw_sih_speed_err)
    rect_speed_mae = np.mean(rect_sih_speed_err)
    speed_reduction = (raw_speed_mae - rect_speed_mae) / raw_speed_mae * 100
    
    print("\n=======================================================")
    print("        BENCHMARK VERIFICATION: SIH vs SIH-RECT        ")
    print("=======================================================")
    print(f"Step Displacement MAE:")
    print(f"  Raw SIH MLP:       {raw_disp_mae:.4f} meters")
    print(f"  SIH-Rectified:     {rect_disp_mae:.4f} meters (Improvement: {disp_reduction:+.2f}%)")
    print(f"Speed MAE:")
    print(f"  Raw SIH MLP:       {raw_speed_mae * 3.6:.2f} km/h")
    print(f"  SIH-Rectified:     {rect_speed_mae * 3.6:.2f} km/h (Improvement: {speed_reduction:+.2f}%)")
    print("=======================================================")
    
    # Export Scaler Parameters for Web Deployment
    scaler_dict = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "seq_len": 60,
        "step_interval_s": 0.5,
        "sampling_rate_hz": 60,
        "metrics": {
            "raw_disp_mae_m": float(raw_disp_mae),
            "rect_disp_mae_m": float(rect_disp_mae),
            "disp_reduction_pct": float(disp_reduction),
            "raw_speed_mae_kmh": float(raw_speed_mae * 3.6),
            "rect_speed_mae_kmh": float(rect_speed_mae * 3.6),
            "speed_reduction_pct": float(speed_reduction)
        }
    }
    
    scaler_out = os.path.join(MODELS_DIR, "rect_scaler.json")
    with open(scaler_out, "w") as f:
        json.dump(scaler_dict, f, indent=2)
        
    public_scaler_out = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "..", "public", "models", "rect_scaler.json"))
    with open(public_scaler_out, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"Saved scaler params to {scaler_out} and {public_scaler_out}")
    
    # -------------------------------------------------------------
    # ONNX Export
    # -------------------------------------------------------------
    model.eval()
    dummy_input = torch.randn(1, 60, 6, dtype=torch.float32).to(device)
    onnx_path_exp = os.path.join(MODELS_DIR, "sih_rect_transformer.onnx")
    onnx_path_public = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "..", "public", "models", "sih_rect_transformer.onnx"))
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path_exp,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['imu_window_60hz'],
        output_names=['residual_corrections'],
        dynamic_axes={
            'imu_window_60hz': {0: 'batch_size'},
            'residual_corrections': {0: 'batch_size'}
        }
    )
    
    # Copy to public/models for web app
    import shutil
    shutil.copyfile(onnx_path_exp, onnx_path_public)
    print(f"ONNX Model successfully exported to:")
    print(f"  1. {onnx_path_exp} ({os.path.getsize(onnx_path_exp)} bytes)")
    print(f"  2. {onnx_path_public} ({os.path.getsize(onnx_path_public)} bytes)")
    
    # Verify ONNX Runtime
    test_sess = ort.InferenceSession(onnx_path_public, providers=['CPUExecutionProvider'])
    test_out = test_sess.run(None, {'imu_window_60hz': dummy_input.cpu().numpy()})
    print(f"Verified ONNX Inference: output shape={test_out[0].shape}, values={test_out[0][0]}")

if __name__ == '__main__':
    main()
