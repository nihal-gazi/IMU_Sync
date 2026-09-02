"""
2-Stage Interactive & Animated Trajectory Evaluation Script
Stage 1: Rest vs Moving Classifier (Zero-Velocity Detector)
Stage 2: IMU-Transformer Body-Frame Displacement + 3D Orientation Tracking
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import parse_and_clean_imu_data, download_dataset, Normalizer
from models import RestMovingClassifierMLP, IMUTransformerTLIO

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))


def load_models_and_scalers():
    scaler_path = os.path.join(RESEARCH_DIR, "scaler_params.json")
    with open(scaler_path, "r") as f:
        scaler_dict = json.load(f)

    feat_norm = Normalizer()
    feat_norm.mean = np.array(scaler_dict['features']['mean'], dtype=np.float32)
    feat_norm.std = np.array(scaler_dict['features']['std'], dtype=np.float32)

    target_norm = Normalizer()
    target_norm.mean = np.array(scaler_dict['targets']['mean'], dtype=np.float32)
    target_norm.std = np.array(scaler_dict['targets']['std'], dtype=np.float32)

    # 1. Load Motion Classifier (Stage 1)
    cls_model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64)
    cls_pt = os.path.join(RESEARCH_DIR, "motion_classifier.pt")
    cls_model.load_state_dict(torch.load(cls_pt, map_location="cpu"))
    cls_model.eval()

    # 2. Load IMU-Transformer (Stage 2)
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
    trans_pt = os.path.join(RESEARCH_DIR, "tlio_transformer.pt")
    transformer_model.load_state_dict(torch.load(trans_pt, map_location="cpu"))
    transformer_model.eval()

    print(f"[Animate] Loaded 2-Stage Models: Motion Classifier ({cls_pt}) & Transformer ({trans_pt})")
    return cls_model, transformer_model, feat_norm, target_norm


def run_evaluation_trajectory(dataset_key: str = "S-S1", max_seconds: int = 120):
    cls_model, trans_model, feat_norm, target_norm = load_models_and_scalers()
    csv_path = download_dataset(dataset_key)
    raw_df = pd.read_csv(csv_path, encoding='latin1')
    raw_df.columns = [c.strip() for c in raw_df.columns]
    
    # Locate Orientation & IMU columns
    for c in raw_df.columns:
        if 'gps orientation' in c.lower() or ('orientation' in c.lower() and 'gps' in c.lower()): gps_h_col = c
        if 'orientation (yaw)' in c.lower(): phone_yaw_col = c
        if 'gps speed' in c.lower(): gps_sp_col = c
        if 'accel' in c.lower() and 'x' in c.lower(): ax_col = c
        if 'accel' in c.lower() and 'y' in c.lower(): ay_col = c
        if 'accel' in c.lower() and 'z' in c.lower(): az_col = c
        if 'gyro' in c.lower() and 'roll' in c.lower(): gx_col = c
        if 'gyro' in c.lower() and 'pitch' in c.lower(): gy_col = c
        if 'gyro' in c.lower() and 'yaw' in c.lower(): gz_col = c

    window_size = 10
    total_samples = min(len(raw_df), max_seconds * 10)
    sample_df = raw_df.iloc[:total_samples].reset_index(drop=True)

    features = sample_df[[ax_col, ay_col, az_col, gx_col, gy_col, gz_col]].values.astype(np.float32)
    norm_features = feat_norm.transform(features)

    speeds_mps = (sample_df[gps_sp_col].values / 3.6).astype(np.float32)
    gps_headings = sample_df[gps_h_col].values.astype(np.float32)
    phone_yaws = sample_df[phone_yaw_col].values.astype(np.float32)

    # 1. Ground Truth GPS Trajectory
    gt_x_all = [0.0]
    gt_y_all = [0.0]
    for i in range(len(sample_df) - 1):
        step = speeds_mps[i] * 0.1
        h_rad = np.radians(gps_headings[i])
        gt_x_all.append(gt_x_all[-1] + step * np.sin(h_rad))
        gt_y_all.append(gt_y_all[-1] + step * np.cos(h_rad))

    start_bearing_deg = float(gps_headings[0])
    initial_phone_yaw = float(phone_yaws[0])
    print(f"[Animate] Initial Heading: {start_bearing_deg:.1f}°, Initial Phone Yaw: {initial_phone_yaw:.1f}°")

    pred_x = [0.0]
    pred_y = [0.0]
    pred_speeds = [0.0]
    motion_states = ["REST"]

    cur_px = 0.0
    cur_py = 0.0

    gt_x_1s = [0.0]
    gt_y_1s = [0.0]
    gt_speeds_1s = [0.0]

    with torch.no_grad():
        for start in range(0, total_samples - window_size, window_size):
            end = start + window_size
            w_x = norm_features[start:end]
            tensor_x = torch.tensor(w_x, dtype=torch.float32).unsqueeze(0)

            # STAGE 1: Rest vs Moving Classifier
            logits = cls_model(tensor_x)
            is_moving = int(torch.argmax(logits, dim=1).item())

            # STAGE 2: Transformer Regressor
            if is_moving == 1:
                pred_norm = trans_model(tensor_x).numpy()[0]
                pred_disp = target_norm.inverse_transform(pred_norm)
                fwd_disp = max(0.0, float(pred_disp[1]))
                m_state = "MOVING"
            else:
                fwd_disp = 0.0
                m_state = "REST"

            motion_states.append(m_state)

            # 3D Vehicle Heading Tracking from Phone Orientation
            cur_phone_yaw = float(phone_yaws[end - 1])
            delta_yaw = (cur_phone_yaw - initial_phone_yaw + 180) % 360 - 180
            cur_heading_deg = (start_bearing_deg - delta_yaw) % 360
            cur_heading_rad = np.radians(cur_heading_deg)

            # Rotate Body-Frame Displacement into Global World Coordinates
            dx_world = fwd_disp * np.sin(cur_heading_rad)
            dy_world = fwd_disp * np.cos(cur_heading_rad)

            cur_px += dx_world
            cur_py += dy_world

            pred_x.append(cur_px)
            pred_y.append(cur_py)
            pred_speeds.append(fwd_disp * 3.6)

            # Ground truth 1-second step
            gt_x_1s.append(gt_x_all[end - 1])
            gt_y_1s.append(gt_y_all[end - 1])
            gt_sp = float(np.mean(speeds_mps[start:end])) * 3.6
            gt_speeds_1s.append(gt_sp)

    pred_x = np.array(pred_x)
    pred_y = np.array(pred_y)
    gt_x_1s = np.array(gt_x_1s)
    gt_y_1s = np.array(gt_y_1s)
    pred_speeds = np.array(pred_speeds)
    gt_speeds_1s = np.array(gt_speeds_1s)

    ate_errors = np.hypot(pred_x - gt_x_1s, pred_y - gt_y_1s)
    mean_ate = np.mean(ate_errors)
    final_ate = ate_errors[-1]

    print(f"\n--- 2-Stage Trajectory Evaluation Summary ({len(pred_x)} seconds) ---")
    print(f"Total Distance Traveled (Ground Truth): {np.hypot(gt_x_1s[-1], gt_y_1s[-1]):.2f} meters")
    print(f"Total Distance Traveled (Predicted):    {np.hypot(pred_x[-1], pred_y[-1]):.2f} meters")
    print(f"Mean Absolute Trajectory Error (ATE):   {mean_ate:.2f} meters")
    print(f"Final Drift Error at End of Route:      {final_ate:.2f} meters\n")

    return {
        'gt_x': gt_x_1s,
        'gt_y': gt_y_1s,
        'gt_speeds': gt_speeds_1s,
        'pred_x': pred_x,
        'pred_y': pred_y,
        'pred_speeds': pred_speeds,
        'ate_errors': ate_errors,
        'motion_states': motion_states
    }


def create_animated_plot(data: dict, save_gif: bool = True, gif_path: str = "trajectory_comparison.gif"):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8), dpi=100)
    fig.patch.set_facecolor('#0d1117')

    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], figure=fig, hspace=0.3, wspace=0.25)

    # Subplot 1: 2D Spatial Map
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor('#161b22')
    ax_map.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_map.set_title("2D Ground Truth Path vs 2-Stage IMU AI Prediction", color='#00f0ff', fontsize=12, fontweight='bold', pad=10)
    ax_map.set_xlabel("X Position (East / m)", color='#8b949e', fontsize=10)
    ax_map.set_ylabel("Y Position (North / m)", color='#8b949e', fontsize=10)

    gt_x = data['gt_x']
    gt_y = data['gt_y']
    pred_x = data['pred_x']
    pred_y = data['pred_y']

    min_x = min(np.min(gt_x), np.min(pred_x)) - 15
    max_x = max(np.max(gt_x), np.max(pred_x)) + 15
    min_y = min(np.min(gt_y), np.min(pred_y)) - 15
    max_y = max(np.max(gt_y), np.max(pred_y)) + 15

    ax_map.set_xlim(min_x, max_x)
    ax_map.set_ylim(min_y, max_y)

    ax_map.plot(gt_x, gt_y, color='#2ea043', linestyle=':', alpha=0.35, label='GT Route (Complete)')
    ax_map.plot(pred_x, pred_y, color='#00f0ff', linestyle=':', alpha=0.35, label='Predicted (Complete)')

    line_gt, = ax_map.plot([], [], color='#3fb950', linewidth=2.8, label='Ground Truth Path')
    line_pred, = ax_map.plot([], [], color='#00f0ff', linewidth=2.8, label='2-Stage AI Path')
    head_gt, = ax_map.plot([], [], marker='o', markersize=7, color='#2ea043', markeredgecolor='white')
    head_pred, = ax_map.plot([], [], marker='^', markersize=8, color='#ffb800', markeredgecolor='white')
    ax_map.plot(0, 0, marker='s', markersize=7, color='#ffffff', label='Start (0,0)')
    ax_map.legend(loc='upper left', framealpha=0.6, facecolor='#0d1117', edgecolor='#30363d', fontsize=9)

    # Subplot 2: Speed Tracking & Rest Gating
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.set_facecolor('#161b22')
    ax_speed.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_speed.set_title("Speed Tracking & Rest Detection (km/h)", color='#f0883e', fontsize=11, fontweight='bold')
    ax_speed.set_xlabel("Time (s)", color='#8b949e', fontsize=9)
    ax_speed.set_ylabel("Speed (km/h)", color='#8b949e', fontsize=9)
    ax_speed.set_xlim(0, len(pred_x))
    ax_speed.set_ylim(0, max(np.max(data['gt_speeds']), np.max(data['pred_speeds'])) + 10)

    line_speed_gt, = ax_speed.plot([], [], color='#3fb950', linewidth=1.8, label='GT Speed')
    line_speed_pred, = ax_speed.plot([], [], color='#00f0ff', linewidth=1.8, linestyle='--', label='AI Speed')
    ax_speed.legend(loc='upper right', framealpha=0.6, facecolor='#0d1117', edgecolor='#30363d', fontsize=8)

    # Subplot 3: Absolute Trajectory Error (ATE) Tracking
    ax_error = fig.add_subplot(gs[1, 1])
    ax_error.set_facecolor('#161b22')
    ax_error.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_error.set_title("Absolute Trajectory Drift Error (ATE in Meters)", color='#f85149', fontsize=11, fontweight='bold')
    ax_error.set_xlabel("Time (s)", color='#8b949e', fontsize=9)
    ax_error.set_ylabel("Drift Error (m)", color='#8b949e', fontsize=9)
    ax_error.set_xlim(0, len(pred_x))
    ax_error.set_ylim(0, max(10, np.max(data['ate_errors']) + 5))

    line_error, = ax_error.plot([], [], color='#f85149', linewidth=2.0, label='Position Drift ATE (m)')
    ax_error.legend(loc='upper left', framealpha=0.6, facecolor='#0d1117', edgecolor='#30363d', fontsize=8)

    hud_text = ax_map.text(
        0.03, 0.05, "",
        transform=ax_map.transAxes,
        color='#ffffff',
        fontfamily='monospace',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117', edgecolor='#30363d', alpha=0.85)
    )

    n_frames = len(pred_x)

    def init():
        line_gt.set_data([], [])
        line_pred.set_data([], [])
        head_gt.set_data([], [])
        head_pred.set_data([], [])
        line_speed_gt.set_data([], [])
        line_speed_pred.set_data([], [])
        line_error.set_data([], [])
        hud_text.set_text("")
        return line_gt, line_pred, head_gt, head_pred, line_speed_gt, line_speed_pred, line_error, hud_text

    def update(frame):
        i = frame + 1
        t_arr = np.arange(i)

        line_gt.set_data(gt_x[:i], gt_y[:i])
        line_pred.set_data(pred_x[:i], pred_y[:i])
        head_gt.set_data([gt_x[i - 1]], [gt_y[i - 1]])
        head_pred.set_data([pred_x[i - 1]], [pred_y[i - 1]])

        line_speed_gt.set_data(t_arr, data['gt_speeds'][:i])
        line_speed_pred.set_data(t_arr, data['pred_speeds'][:i])

        line_error.set_data(t_arr, data['ate_errors'][:i])

        cur_ate = data['ate_errors'][i - 1]
        cur_spd_gt = data['gt_speeds'][i - 1]
        cur_spd_pred = data['pred_speeds'][i - 1]
        state = data['motion_states'][i - 1]
        hud_text.set_text(
            f"TIME: {i:03d}s / {n_frames}s\n"
            f"MOTION STATE: [{state}]\n"
            f"GT POS:   ({gt_x[i-1]:.1f}, {gt_y[i-1]:.1f}) m\n"
            f"PRED POS: ({pred_x[i-1]:.1f}, {pred_y[i-1]:.1f}) m\n"
            f"GT SPEED:   {cur_spd_gt:.1f} km/h\n"
            f"PRED SPEED: {cur_spd_pred:.1f} km/h\n"
            f"DRIFT ATE:  {cur_ate:.2f} m"
        )

        return line_gt, line_pred, head_gt, head_pred, line_speed_gt, line_speed_pred, line_error, hud_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=80, blit=True
    )

    if save_gif:
        out_file = os.path.join(RESEARCH_DIR, gif_path)
        print(f"[Animate] Rendering animated GIF to {out_file}...")
        try:
            anim.save(out_file, writer='pillow', fps=12)
            print(f"[Animate] Saved animation to {out_file} ({os.path.getsize(out_file):,} bytes)!")
        except Exception as e:
            print(f"[Animate] Could not save GIF: {e}")

    return anim, fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="2-Stage IMU AI Trajectory Evaluation")
    parser.add_argument("--dataset", type=str, default="S-S1", help="Dataset key (S-S1, S-S2, S-M)")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to evaluate")
    parser.add_argument("--no-save", action="store_true", help="Skip saving GIF")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    args = parser.parse_args()

    eval_data = run_evaluation_trajectory(dataset_key=args.dataset, max_seconds=args.seconds)
    anim, fig = create_animated_plot(eval_data, save_gif=not args.no_save, gif_path="trajectory_comparison.gif")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
