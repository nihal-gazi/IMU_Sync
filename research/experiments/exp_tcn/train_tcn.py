"""
Train Dilated Temporal Convolutional Network (TCN) on IO-VNBD Dataset (S-S1 & S-S2).
Native 10Hz sampling (T=20 window over 2.0s).
Predicts:
  1. v_forward (Instantaneous Speed in m/s)
  2. ZUPT Flag (Stationary Detection Probability [0, 1])
Exports ONNX model and normalizer scaler to public/models/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class InertialTCN(nn.Module):
    def __init__(self, in_channels=6, num_channels=[32, 48, 64], kernel_size=3, dropout=0.05):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1,
                    dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Head 1: Forward Speed (m/s)
        self.speed_head = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.ReLU() # Speed is non-negative
        )

        # Head 2: ZUPT Stationary Flag [0, 1]
        self.zupt_head = nn.Sequential(
            nn.Linear(num_channels[-1], 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, SeqLen, InChannels] -> Transpose to [Batch, InChannels, SeqLen]
        x_trans = x.transpose(1, 2)
        feats = self.tcn(x_trans)
        pooled = self.pool(feats).squeeze(-1) # [Batch, 64]

        v_forward = self.speed_head(pooled) # [Batch, 1]
        zupt_flag = self.zupt_head(pooled)  # [Batch, 1]

        # Return concatenated [v_forward, zupt_flag] shape: [Batch, 2]
        return torch.cat([v_forward, zupt_flag], dim=-1)

class IOVNBDSeqDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

def load_iovnb_data(seq_len=20, stride=2):
    file_paths = [
        r'C:\Users\user\Desktop\IMU_Sync\research\data\S-S1.csv',
        r'C:\Users\user\Desktop\IMU_Sync\research\data\S-S2.csv'
    ]

    all_windows = []
    all_targets = []

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        print(f"Loading {os.path.basename(path)}...")
        df = pd.read_csv(path, encoding='latin1')

        # Clean NaNs
        ax = df.iloc[:, 9].ffill().bfill().values.astype(np.float32)
        ay = df.iloc[:, 10].ffill().bfill().values.astype(np.float32)
        az = df.iloc[:, 11].ffill().bfill().values.astype(np.float32)
        
        # Gyro in rad/s: Yaw, Pitch, Roll -> map to [Roll, Pitch, Yaw] = [gx, gy, gz]
        gz = df.iloc[:, 15].ffill().bfill().values.astype(np.float32)
        gy = df.iloc[:, 16].ffill().bfill().values.astype(np.float32)
        gx = df.iloc[:, 17].ffill().bfill().values.astype(np.float32)

        # GPS Speed in km/h -> convert to m/s
        speed_kmh = df.iloc[:, 3].ffill().bfill().values.astype(np.float32)
        speed_mps = speed_kmh / 3.6

        # ZUPT Flag: 1 if stationary (speed < 0.2 m/s or ~0.7 km/h), 0 if moving
        zupt = (speed_mps < 0.20).astype(np.float32)

        feats = np.stack([ax, ay, az, gx, gy, gz], axis=-1)
        targets = np.stack([speed_mps, zupt], axis=-1)

        N = len(feats)
        for s in range(0, N - seq_len + 1, stride):
            e = s + seq_len
            all_windows.append(feats[s:e])
            # Target is the instantaneous state at the end of the window
            all_targets.append(targets[e - 1])

    all_windows = np.array(all_windows, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    return all_windows, all_targets

def main():
    print("=== TCN Speed Filter & ZUPT Training Pipeline ===")
    seq_len = 20
    windows, targets = load_iovnb_data(seq_len=seq_len, stride=2)
    print(f"Dataset generated: {len(windows)} sequences of shape ({seq_len}, 6)")
    print(f"Mean Speed: {targets[:, 0].mean():.2f} m/s, Max Speed: {targets[:, 0].max():.2f} m/s")
    print(f"Stationary Ratio (ZUPT): {(targets[:, 1] > 0.5).mean() * 100:.1f}%")

    # Compute Feature Normalization (Scaler)
    mean_feats = np.mean(windows, axis=(0, 1))
    std_feats = np.std(windows, axis=(0, 1)) + 1e-6

    # Normalize
    windows = (windows - mean_feats) / std_feats

    # Split Train / Validation (85% / 15%)
    n_samples = len(windows)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split = int(0.85 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = IOVNBDSeqDataset(windows[train_idx], targets[train_idx])
    val_ds = IOVNBDSeqDataset(windows[val_idx], targets[val_idx])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = InertialTCN(in_channels=6, num_channels=[32, 48, 64], kernel_size=3).to(device)

    # Multi-task Loss: Smooth L1 for speed + BCE for ZUPT
    speed_criterion = nn.SmoothL1Loss()
    zupt_criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(exp_dir, "models", "best_tcn.pt")

    print("\n--- Starting TCN Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            pred = model(batch_x)
            loss_v = speed_criterion(pred[:, 0], batch_y[:, 0])
            loss_z = zupt_criterion(pred[:, 1], batch_y[:, 1])
            total_loss = loss_v + 0.5 * loss_z

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(batch_x)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_speed_errors = []
        val_zupt_acc = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)

                loss_v = speed_criterion(pred[:, 0], batch_y[:, 0])
                loss_z = zupt_criterion(pred[:, 1], batch_y[:, 1])
                total_loss = loss_v + 0.5 * loss_z
                val_loss += total_loss.item() * len(batch_x)

                speed_err = torch.abs(pred[:, 0] - batch_y[:, 0]).cpu().numpy()
                val_speed_errors.extend(speed_err)

                z_pred = (pred[:, 1] > 0.5).float()
                acc = (z_pred == batch_y[:, 1]).float().mean().item()
                val_zupt_acc.append(acc)

        val_loss /= len(val_ds)
        mean_speed_mae = np.mean(val_speed_errors)
        mean_zupt_acc = np.mean(val_zupt_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 4 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Speed MAE: {mean_speed_mae:.3f} m/s ({mean_speed_mae * 3.6:.2f} km/h) | ZUPT Acc: {mean_zupt_acc*100:.1f}%")

    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.4f}")

    # Save Scaler Params
    scaler_dict = {
        "mean": mean_feats.tolist(),
        "std": std_feats.tolist(),
        "seq_len": seq_len,
        "features": ["ax", "ay", "az", "gx", "gy", "gz"]
    }
    scaler_exp_path = os.path.join(exp_dir, "models", "tcn_scaler.json")
    scaler_pub_path = os.path.abspath(os.path.join(exp_dir, "..", "..", "..", "public", "models", "tcn_scaler.json"))
    with open(scaler_exp_path, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    with open(scaler_pub_path, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"Saved scaler to: {scaler_pub_path}")

    # Export to ONNX
    onnx_exp_path = os.path.join(exp_dir, "models", "tcn_speed_filter.onnx")
    onnx_pub_path = os.path.abspath(os.path.join(exp_dir, "..", "..", "..", "public", "models", "tcn_speed_filter.onnx"))

    print("\nExporting TCN Model to ONNX...")
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    model.eval()
    model.to('cpu')

    dummy_input = torch.randn(1, seq_len, 6, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_exp_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['imu_window_10hz'],
        output_names=['tcn_output'],
        dynamic_axes={
            'imu_window_10hz': {0: 'batch_size'},
            'tcn_output': {0: 'batch_size'}
        },
        dynamo=False
    )
    import shutil
    shutil.copyfile(onnx_exp_path, onnx_pub_path)
    file_size_kb = os.path.getsize(onnx_pub_path) / 1024.0
    print(f"ONNX Model saved to {onnx_pub_path} ({file_size_kb:.1f} KB)")

    # Verify ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_pub_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: dummy_input.numpy()})
    print(f"ONNX Session Verified! Output shape: {out[0].shape}, Sample [v_fwd, zupt]: {out[0][0]}")

if __name__ == '__main__':
    main()
