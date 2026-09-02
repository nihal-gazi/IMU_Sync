"""
Animated Trajectory Visualizer for the EXACT Same Route WITHOUT Calibration (Raw k_accel=1.0, k_gyro=1.0)
"""

import os
import sys
import torch
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

sys.path.append(EXP_DIR)
from run_exp1 import RestMovingClassifierMLP, IMUTransformerTLIO, load_and_preprocess_100hz_data
from benchmark_50_paths import generate_single_diverse_path

os.makedirs(REPORT_DIR, exist_ok=True)


def animate_uncalibrated_route():
    data = load_and_preprocess_100hz_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64).to(device)
    cls_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'motion_classifier.pt'), map_location=device))
    cls_model.eval()

    trans_model = IMUTransformerTLIO(input_dim=6, window_size=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0, output_dim=2).to(device)
    trans_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'tlio_transformer.pt'), map_location=device))
    trans_model.eval()

    feat_norm = data['feat_norm']
    target_norm = data['target_norm']

    # RAW UNCALIBRATED FACTORS (1.0x)
    k_accel = 1.0000
    k_gyro = 1.0000

    # EXACT SAME ROUTE (seed=2026, mixed_driving, 150 seconds)
    duration = 150
    df_path, route_name, max_sp = generate_single_diverse_path(seed=2026, route_type="mixed_driving", duration_s=duration, dt=0.1)

    norm_feats = feat_norm.transform(df_path[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values)
    raw_gz = df_path['gz'].values
    gt_x = df_path['pos_x'].values
    gt_y = df_path['pos_y'].values
    speeds_gt = df_path['speed_mps'].values

    pred_x, pred_y = [0.0], [0.0]
    pred_speeds = [0.0]
    cur_px, cur_py, cur_v = 0.0, 0.0, 0.0
    cur_h = float(df_path['heading_rad'].iloc[0])

    with torch.no_grad():
        for s in range(0, len(df_path) - 10, 10):
            w_x = norm_feats[s:s+10]
            tx = torch.tensor(w_x, dtype=torch.float32).unsqueeze(0).to(device)
            is_moving = int(torch.argmax(cls_model(tx), dim=1).item())
            v_prev = cur_v

            if is_moving == 1:
                out = trans_model(tx).cpu().numpy()[0]
                a_pred = float(target_norm.inverse_transform(out)[1])
                cur_v = max(0.0, cur_v + a_pred * k_accel)
            else:
                cur_v = 0.0

            fwd_disp = ((v_prev + cur_v) / 2.0) * 1.0
            gz_1s = float(np.sum(raw_gz[s:s+10]) * 0.1) * k_gyro
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

    total_gt_distance = float(np.sum(np.hypot(np.diff(gt_x_step), np.diff(gt_y_step))))
    ate_errors = np.hypot(pred_x - gt_x_step, pred_y - gt_y_step)
    final_drift = float(ate_errors[-1])
    mean_ate = float(np.mean(ate_errors))

    print(f"[Uncalibrated] Route: {route_name}, Total Distance: {total_gt_distance:.2f} m")
    print(f"[Uncalibrated] Final Drift: {final_drift:.2f} m ({(final_drift/total_gt_distance)*100:.2f}%)")
    print(f"[Uncalibrated] Mean ATE:    {mean_ate:.2f} m ({(mean_ate/total_gt_distance)*100:.2f}%)")

    # Render Animated GIF
    print("[Uncalibrated] Rendering 150s animation frames...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8), dpi=100)
    fig.patch.set_facecolor('#0d1117')

    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], figure=fig, hspace=0.3, wspace=0.25)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor('#161b22')
    ax_map.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_map.set_title("RAW Uncalibrated Neural Odometry: Same Route (1.8 km Drive)", color='#f85149', fontsize=12, fontweight='bold')
    ax_map.set_xlabel("X Position (East / m)", color='#8b949e')
    ax_map.set_ylabel("Y Position (North / m)", color='#8b949e')

    margin = 40
    min_x = min(np.min(gt_x_step), np.min(pred_x)) - margin
    max_x = max(np.max(gt_x_step), np.max(pred_x)) + margin
    min_y = min(np.min(gt_y_step), np.min(pred_y)) - margin
    max_y = max(np.max(gt_y_step), np.max(pred_y)) + margin

    ax_map.set_xlim(min_x, max_x)
    ax_map.set_ylim(min_y, max_y)

    ax_map.plot(gt_x_step, gt_y_step, color='#2ea043', linestyle=':', alpha=0.35, label='GT Route (Full)')
    ax_map.plot(pred_x, pred_y, color='#f85149', linestyle=':', alpha=0.35, label='Raw AI Path (Full)')

    line_gt, = ax_map.plot([], [], color='#3fb950', linewidth=2.8, label='Ground Truth Path')
    line_pred, = ax_map.plot([], [], color='#f85149', linewidth=2.8, label='Raw AI Path (Uncalibrated)')
    head_gt, = ax_map.plot([], [], marker='o', markersize=7, color='#2ea043', markeredgecolor='white')
    head_pred, = ax_map.plot([], [], marker='^', markersize=8, color='#ff7b72', markeredgecolor='white')
    ax_map.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d')

    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.set_facecolor('#161b22')
    ax_speed.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_speed.set_title("Raw Kinematic Speed Tracking (km/h)", color='#f0883e', fontsize=11, fontweight='bold')
    ax_speed.set_xlim(0, len(pred_x))
    ax_speed.set_ylim(0, max(np.max(gt_speeds_step), np.max(pred_speeds)) + 12)

    line_speed_gt, = ax_speed.plot([], [], color='#3fb950', linewidth=1.8, label='GT Speed')
    line_speed_pred, = ax_speed.plot([], [], color='#f85149', linewidth=1.8, linestyle='--', label='Raw AI Speed')
    ax_speed.legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d')

    ax_error = fig.add_subplot(gs[1, 1])
    ax_error.set_facecolor('#161b22')
    ax_error.grid(True, linestyle='--', color='#ffffff', alpha=0.15)
    ax_error.set_title("Positional Drift Error (ATE in Meters)", color='#f85149', fontsize=11, fontweight='bold')
    ax_error.set_xlim(0, len(pred_x))
    ax_error.set_ylim(0, max(30, np.max(ate_errors) + 15))

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
    gif_out = os.path.join(REPORT_DIR, "trajectory_uncalibrated_raw.gif")
    anim.save(gif_out, writer='pillow', fps=12)
    plt.close(fig)
    print(f"[Uncalibrated] Saved animation to {gif_out} ({os.path.getsize(gif_out):,} bytes)!")


if __name__ == "__main__":
    animate_uncalibrated_route()
