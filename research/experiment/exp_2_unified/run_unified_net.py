"""
Experiment 2: Unified Multi-Task IMU Transformer
A single neural network that outputs:
  - Motion Classification Logits: [is_not_moving, is_moving]
  - Kinematic Acceleration: [a_x (lateral), a_y (forward)]
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(EXP_DIR, "report")
MODELS_DIR = os.path.join(EXP_DIR, "models")
DATA_DIR = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "data", "highquality"))

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class UnifiedIMUTransformer(nn.Module):
    """
    Unified Multi-Task IMU Transformer
    Outputs both [is_not_moving, is_moving] classification logits and continuous [a_x, a_y] acceleration
    """
    def __init__(self, input_dim=6, window_size=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_shared = nn.LayerNorm(d_model)

        # Head 1: Motion Classification (REST:0, MOVING:1)
        self.motion_classifier_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

        # Head 2: Kinematic Acceleration (a_x lateral, a_y forward)
        self.accel_regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        B, T, _ = x.shape
        proj = self.input_proj(x) + self.pos_embedding[:, :T, :]
        trans_out = self.transformer_encoder(proj)
        pooled = self.ln_shared(trans_out.mean(dim=1))

        motion_logits = self.motion_classifier_head(pooled) # (B, 2) -> [is_not_moving, is_moving]
        accel_preds = self.accel_regression_head(pooled)     # (B, 2) -> [a_x, a_y]

        return motion_logits, accel_preds


def load_dataset():
    csv_file = os.path.join(DATA_DIR, "kitti_urban_100hz_drive.csv")
    df = pd.read_csv(csv_file)
    print(f"[UnifiedNet] Loaded {csv_file}: {len(df):,} samples @ 100Hz ({df['timestamp_s'].max():.1f}s)")

    # Downsample to 10Hz windows (10 samples/sec)
    df_10hz = df.iloc[::10].reset_index(drop=True)

    feature_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    raw_feats = df_10hz[feature_cols].values.astype(np.float32)
    speeds = df_10hz['speed_mps'].values.astype(np.float32)

    feat_norm = Normalizer()
    feat_norm.fit(raw_feats)
    norm_feats = feat_norm.transform(raw_feats)

    window_size = 10
    stride = 1
    windows_X = []
    labels_motion = []
    targets_accel = []

    for start in range(0, len(df_10hz) - window_size, stride):
        end = start + window_size
        w_x = norm_feats[start:end]

        dv_fwd = float(speeds[end - 1] - speeds[start]) # 1s acceleration
        avg_speed = float(np.mean(speeds[start:end]))
        is_moving = 1 if avg_speed >= 0.2 else 0

        # a_x (lateral turn force estimate) and a_y (forward acceleration)
        mean_ax = float(np.mean(raw_feats[start:end, 0]))
        
        windows_X.append(w_x)
        labels_motion.append(is_moving)
        targets_accel.append([mean_ax, dv_fwd])

    windows_X = np.array(windows_X, dtype=np.float32)
    labels_motion = np.array(labels_motion, dtype=np.int64)
    targets_accel = np.array(targets_accel, dtype=np.float32)

    target_norm = Normalizer()
    target_norm.fit(targets_accel)
    norm_accel = target_norm.transform(targets_accel)

    split = int(len(windows_X) * 0.80)
    print(f"[UnifiedNet] Extracted {len(windows_X):,} Windows (Train: {split}, Val: {len(windows_X)-split})")

    return {
        'df_10hz': df_10hz,
        'feat_norm': feat_norm,
        'target_norm': target_norm,
        'train_X': windows_X[:split],
        'train_cls_y': labels_motion[:split],
        'train_reg_y': norm_accel[:split],
        'val_X': windows_X[split:],
        'val_cls_y': labels_motion[split:],
        'val_reg_y': norm_accel[split:],
        'raw_val_accel': targets_accel[split:],
        'split': split
    }


def train_unified_transformer():
    data = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UnifiedNet] Training Unified Transformer on device: {device}")

    train_X = torch.tensor(data['train_X'], dtype=torch.float32)
    train_cls_y = torch.tensor(data['train_cls_y'], dtype=torch.long)
    train_reg_y = torch.tensor(data['train_reg_y'], dtype=torch.float32)

    val_X = torch.tensor(data['val_X'], dtype=torch.float32).to(device)
    val_cls_y = torch.tensor(data['val_cls_y'], dtype=torch.long).to(device)
    val_reg_y = torch.tensor(data['val_reg_y'], dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_X, train_cls_y, train_reg_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = UnifiedIMUTransformer(input_dim=6, window_size=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    best_acc = 0.0

    print("\n[UnifiedNet] Starting Multi-Task Training (30 Epochs)...")
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        for bx, b_cls, b_reg in train_loader:
            bx, b_cls, b_reg = bx.to(device), b_cls.to(device), b_reg.to(device)
            optimizer.zero_grad()
            
            pred_cls, pred_reg = model(bx)
            loss_cls = criterion_cls(pred_cls, b_cls)
            loss_reg = criterion_reg(pred_reg, b_reg)

            # Joint Multi-Task Loss: Classification + Acceleration Regression
            total_loss = loss_cls + (1.5 * loss_reg)
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(bx)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            v_cls, v_reg = model(val_X)
            v_loss_cls = criterion_cls(v_cls, val_cls_y).item()
            v_loss_reg = criterion_reg(v_reg, val_reg_y).item()
            v_total_loss = v_loss_cls + (1.5 * v_loss_reg)

            pred_labels = torch.argmax(v_cls, dim=1)
            acc = (torch.sum(pred_labels == val_cls_y).item() / len(val_cls_y)) * 100.0

            if v_total_loss < best_val_loss:
                best_val_loss = v_total_loss
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, "unified_imu_transformer.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/30 | Train Loss: {train_loss:.4f} | Val Loss: {v_total_loss:.4f} (Cls: {v_loss_cls:.4f}, Reg: {v_loss_reg:.4f}) | Cls Acc: {acc:.2f}%")

    print(f"\n[UnifiedNet] Training Complete! Best Val Total Loss: {best_val_loss:.4f}, Motion Accuracy: {best_acc:.2f}%")

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "unified_imu_transformer.pt")))
    model.eval()

    with torch.no_grad():
        v_cls, v_reg = model(val_X)
        preds_phys = data['target_norm'].inverse_transform(v_reg.cpu().numpy())
        true_phys = data['raw_val_accel']

        mae_ay = np.mean(np.abs(preds_phys[:, 1] - true_phys[:, 1]))
        mae_ax = np.mean(np.abs(preds_phys[:, 0] - true_phys[:, 0]))

    print(f"[UnifiedNet] Forward Acceleration (a_y) MAE: {mae_ay:.4f} m/s² ({mae_ay*3.6:.2f} km/h/s)")
    print(f"[UnifiedNet] Lateral Force (a_x) MAE:        {mae_ax:.4f} m/s²")

    # Run Trajectory Simulation with the Unified Net
    df_10hz = data['df_10hz']
    speeds_gt = df_10hz['speed_mps'].values
    gt_x = df_10hz['pos_x'].values
    gt_y = df_10hz['pos_y'].values
    raw_gz = df_10hz['gz'].values

    feat_norm = data['feat_norm']
    norm_all_feats = feat_norm.transform(df_10hz[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values)

    pred_x, pred_y = [0.0], [0.0]
    pred_speeds = [0.0]
    cur_px, cur_py, cur_v = 0.0, 0.0, 0.0
    cur_h = 0.0

    with torch.no_grad():
        for s in range(0, len(df_10hz) - 10, 10):
            w_x = norm_all_feats[s:s+10]
            tx = torch.tensor(w_x, dtype=torch.float32).unsqueeze(0).to(device)

            # Single forward pass yields BOTH classification and acceleration
            logits_motion, pred_accel = model(tx)
            
            is_moving = int(torch.argmax(logits_motion, dim=1).item())
            v_prev = cur_v

            if is_moving == 1:
                out_phys = data['target_norm'].inverse_transform(pred_accel.cpu().numpy()[0])
                a_x_lat = float(out_phys[0])
                a_y_fwd = float(out_phys[1])
                cur_v = max(0.0, cur_v + a_y_fwd)
            else:
                cur_v = 0.0

            fwd_disp = ((v_prev + cur_v) / 2.0) * 1.0
            gz_1s = float(np.sum(raw_gz[s:s+10]) * 0.1)
            cur_h += gz_1s

            cur_px += fwd_disp * np.sin(cur_h)
            cur_py += fwd_disp * np.cos(cur_h)
            pred_x.append(cur_px)
            pred_y.append(cur_py)
            pred_speeds.append(cur_v * 3.6)

    pred_x = np.array(pred_x)
    pred_y = np.array(pred_y)
    gt_x_step = gt_x[::10][:len(pred_x)]
    gt_y_step = gt_y[::10][:len(pred_y)]
    gt_speeds_step = (speeds_gt[::10][:len(pred_x)]) * 3.6

    total_dist = float(np.sum(np.hypot(np.diff(gt_x_step), np.diff(gt_y_step))))
    ate_errors = np.hypot(pred_x - gt_x_step, pred_y - gt_y_step)
    final_drift = float(ate_errors[-1])
    mean_ate = float(np.mean(ate_errors))

    print(f"\n--- Unified Transformer Trajectory Summary ---")
    print(f"Total Distance:   {total_dist:.2f} meters")
    print(f"Final Drift:      {final_drift:.2f} meters ({(final_drift/total_dist)*100:.2f}%)")
    print(f"Mean ATE:         {mean_ate:.2f} meters ({(mean_ate/total_dist)*100:.2f}%)")

    # Render Animated GIF
    print(f"[UnifiedNet] Rendering animation to {REPORT_DIR}...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8), dpi=100)
    fig.patch.set_facecolor('#0d1117')

    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], figure=fig, hspace=0.3, wspace=0.25)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor('#161b22')
    ax_map.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_map.set_title("Unified Single-Net Transformer Trajectory (a_x, a_y, REST, MOVING)", color='#00f0ff', fontsize=12, fontweight='bold')
    ax_map.set_xlabel("X Position (East / m)", color='#8b949e')
    ax_map.set_ylabel("Y Position (North / m)", color='#8b949e')

    margin = 30
    ax_map.set_xlim(min(np.min(gt_x_step), np.min(pred_x)) - margin, max(np.max(gt_x_step), np.max(pred_x)) + margin)
    ax_map.set_ylim(min(np.min(gt_y_step), np.min(pred_y)) - margin, max(np.max(gt_y_step), np.max(pred_y)) + margin)

    ax_map.plot(gt_x_step, gt_y_step, color='#2ea043', linestyle=':', alpha=0.35, label='GT Route (Full)')
    ax_map.plot(pred_x, pred_y, color='#00f0ff', linestyle=':', alpha=0.35, label='Unified AI Path (Full)')

    line_gt, = ax_map.plot([], [], color='#3fb950', linewidth=2.8, label='Ground Truth Path')
    line_pred, = ax_map.plot([], [], color='#00f0ff', linewidth=2.8, label='Unified Transformer Path')
    head_gt, = ax_map.plot([], [], marker='o', markersize=7, color='#2ea043', markeredgecolor='white')
    head_pred, = ax_map.plot([], [], marker='^', markersize=8, color='#ffb800', markeredgecolor='white')
    ax_map.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d')

    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.set_facecolor('#161b22')
    ax_speed.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_speed.set_title("Unified Kinematic Speed Tracking (km/h)", color='#f0883e', fontsize=11, fontweight='bold')
    ax_speed.set_xlim(0, len(pred_x))
    ax_speed.set_ylim(0, max(np.max(gt_speeds_step), np.max(pred_speeds)) + 12)

    line_speed_gt, = ax_speed.plot([], [], color='#3fb950', linewidth=1.8, label='GT Speed')
    line_speed_pred, = ax_speed.plot([], [], color='#00f0ff', linewidth=1.8, linestyle='--', label='AI Speed')
    ax_speed.legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d')

    ax_error = fig.add_subplot(gs[1, 1])
    ax_error.set_facecolor('#161b22')
    ax_error.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_error.set_title("Absolute Trajectory Drift Error (ATE in Meters)", color='#f85149', fontsize=11, fontweight='bold')
    ax_error.set_xlim(0, len(pred_x))
    ax_error.set_ylim(0, max(25, np.max(ate_errors) + 15))

    line_error, = ax_error.plot([], [], color='#f85149', linewidth=2.0, label='Drift ATE (m)')
    ax_error.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d')

    def init():
        line_gt.set_data([], [])
        line_pred.set_data([], [])
        head_gt.set_data([], [])
        head_pred.set_data([], [])
        line_speed_gt.set_data([], [])
        line_speed_pred.set_data([], [])
        line_error.set_data([], [])
        return line_gt, line_pred, head_gt, head_pred, line_speed_gt, line_speed_pred, line_error

    def update(frame):
        i = frame + 1
        t_arr = np.arange(i)
        line_gt.set_data(gt_x_step[:i], gt_y_step[:i])
        line_pred.set_data(pred_x[:i], pred_y[:i])
        head_gt.set_data([gt_x_step[i - 1]], [gt_y_step[i - 1]])
        head_pred.set_data([pred_x[i - 1]], [pred_y[i - 1]])
        line_speed_gt.set_data(t_arr, gt_speeds_step[:i])
        line_speed_pred.set_data(t_arr, pred_speeds[:i])
        line_error.set_data(t_arr, ate_errors[:i])
        return line_gt, line_pred, head_gt, head_pred, line_speed_gt, line_speed_pred, line_error

    anim = animation.FuncAnimation(fig, update, frames=len(pred_x), init_func=init, interval=80, blit=True)
    gif_out = os.path.join(REPORT_DIR, "trajectory_unified_net.gif")
    anim.save(gif_out, writer='pillow', fps=12)
    plt.close(fig)
    print(f"[UnifiedNet] Saved animation to {gif_out} ({os.path.getsize(gif_out):,} bytes)!")

    # Export Unified ONNX Model
    export_unified_onnx(model, data)


def export_unified_onnx(model, data):
    model.eval()
    dummy_input = torch.randn(1, 10, 6, dtype=torch.float32)

    onnx_path = os.path.join(MODELS_DIR, "unified_imu_transformer.onnx")
    pub_onnx_path = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "public", "models", "unified_imu_transformer.onnx"))

    torch.onnx.export(
        model.cpu(),
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['imu_window'],
        output_names=['motion_logits', 'accel_preds'],
        dynamic_axes={'imu_window': {0: 'batch_size'}}
    )

    import shutil
    shutil.copy(onnx_path, pub_onnx_path)
    print(f"[UnifiedNet] Exported ONNX to {onnx_path} and {pub_onnx_path}!")


if __name__ == "__main__":
    train_unified_transformer()
