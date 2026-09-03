"""
Generate Trajectory & Metric Comparison Plot for SIH vs SIH-Rect
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data", "resampled_60hz")
MODELS_DIR = os.path.join(EXP_DIR, "models")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

def main():
    # Load test track
    track_path = os.path.join(DATA_DIR, "kitti_urban_60hz_drive.csv")
    df = pd.read_csv(track_path).iloc[600:1800].reset_index(drop=True) # 20s slice
    
    with open(os.path.join(MODELS_DIR, "rect_scaler.json"), "r") as f:
        scaler = json.load(f)
    feat_mean = np.array(scaler["mean"], dtype=np.float32)
    feat_std = np.array(scaler["std"], dtype=np.float32)
    
    sih_sess = ort.InferenceSession(os.path.join(EXP_DIR, "..", "..", "..", "public", "models", "inertial_mlp.onnx"), providers=['CPUExecutionProvider'])
    trans_sess = ort.InferenceSession(os.path.join(MODELS_DIR, "sih_rect_transformer.onnx"), providers=['CPUExecutionProvider'])
    
    raw_imu = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
    
    # Gaussian smooth
    radius = 3
    x_k = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x_k / 1.2) ** 2)
    kernel /= kernel.sum()
    
    smoothed = np.zeros_like(raw_imu)
    for c in range(6):
        smoothed[:, c] = np.convolve(raw_imu[:, c], kernel, mode='same')
        
    sih_in = np.stack([smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], smoothed[:, 5], smoothed[:, 3], smoothed[:, 4]], axis=1)
    
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    heading_deg = df['heading_deg'].values
    
    # Trajectory reconstruction
    gt_x = [0.0]
    gt_y = [0.0]
    sih_x = [0.0]
    sih_y = [0.0]
    rect_x = [0.0]
    rect_y = [0.0]
    
    step = 12 # every 0.2s (5Hz update rate)
    for t in range(60, len(df) - step, step):
        # GT step
        d_gt_x = pos_x[t + step] - pos_x[t]
        d_gt_y = pos_y[t + step] - pos_y[t]
        gt_x.append(gt_x[-1] + d_gt_x)
        gt_y.append(gt_y[-1] + d_gt_y)
        
        # SIH step
        th = np.radians(heading_deg[t])
        sih_feed = sih_in[t - 20 : t].reshape(1, 20, 6)
        sih_out = sih_sess.run(None, {'imu_sequence': sih_feed})[0][0]
        dx_s, dy_s = sih_out[0], sih_out[1]
        disp_s = np.sqrt(dx_s**2 + dy_s**2)
        
        sih_x.append(sih_x[-1] + disp_s * np.sin(th))
        sih_y.append(sih_y[-1] + disp_s * np.cos(th))
        
        # Transformer residual correction
        trans_feed = ((raw_imu[t - 60 : t] - feat_mean) / feat_std).reshape(1, 60, 6)
        res_out = trans_sess.run(None, {'imu_window_60hz': trans_feed})[0][0]
        res_d = res_out[0]
        
        disp_rect = max(0.0, disp_s + res_d * (step / 30.0)) # scaled to 0.2s step
        rect_x.append(rect_x[-1] + disp_rect * np.sin(th))
        rect_y.append(rect_y[-1] + disp_rect * np.cos(th))
        
    # Plot
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(gt_x, gt_y, 'g-', lw=2.5, label='Ground Truth (RTK GPS)')
    plt.plot(sih_x, sih_y, 'r--', lw=2.0, label='Raw SIH MLP (Baseline)')
    plt.plot(rect_x, rect_y, 'c-', lw=2.2, label='SIH-Rectified (+Transformer Drift Fix)')
    
    plt.title("Trajectory Benchmark: Raw SIH vs SIH-Rectified (60Hz Stream)", fontsize=13, fontweight='bold')
    plt.xlabel("East (m)", fontsize=11)
    plt.ylabel("North (m)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(REPORTS_DIR, "sih_vs_rect_trajectory.png")
    plt.savefig(plot_path)
    print(f"Saved trajectory benchmark plot to: {plot_path}")

if __name__ == '__main__':
    main()
