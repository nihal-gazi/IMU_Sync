"""
Benchmark and plot TCN Speed Filter vs Ground Truth GPS Speed on S-S2 test drive.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(EXP_DIR, "models")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

def main():
    test_path = r'C:\Users\user\Desktop\IMU_Sync\research\data\S-S2.csv'
    df = pd.read_csv(test_path, encoding='latin1').iloc[1000:2500].reset_index(drop=True) # 150 seconds slice
    
    with open(os.path.join(MODELS_DIR, "tcn_scaler.json"), "r") as f:
        scaler = json.load(f)
    feat_mean = np.array(scaler["mean"], dtype=np.float32)
    feat_std = np.array(scaler["std"], dtype=np.float32)

    onnx_path = os.path.join(EXP_DIR, "..", "..", "..", "public", "models", "tcn_speed_filter.onnx")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    ax = df.iloc[:, 9].ffill().bfill().values.astype(np.float32)
    ay = df.iloc[:, 10].ffill().bfill().values.astype(np.float32)
    az = df.iloc[:, 11].ffill().bfill().values.astype(np.float32)
    gz = df.iloc[:, 15].ffill().bfill().values.astype(np.float32)
    gy = df.iloc[:, 16].ffill().bfill().values.astype(np.float32)
    gx = df.iloc[:, 17].ffill().bfill().values.astype(np.float32)

    gt_speed_kmh = df.iloc[:, 3].ffill().bfill().values.astype(np.float32)
    gt_speed_mps = gt_speed_kmh / 3.6

    raw_imu = np.stack([ax, ay, az, gx, gy, gz], axis=-1)

    tcn_speeds = []
    tcn_zupt = []
    timestamps = np.arange(len(df)) * 0.1 # 10Hz -> 0.1s per step

    for t in range(20, len(df)):
        window = raw_imu[t - 20 : t]
        norm_window = ((window - feat_mean) / feat_std).reshape(1, 20, 6)
        out = sess.run(None, {input_name: norm_window})[0][0]
        v_fwd, z_flag = out[0], out[1]
        tcn_speeds.append(v_fwd)
        tcn_zupt.append(z_flag)

    tcn_speeds = np.array(tcn_speeds)
    gt_eval_mps = gt_speed_mps[20:]
    gt_eval_kmh = gt_eval_mps * 3.6
    tcn_speeds_kmh = tcn_speeds * 3.6

    mae_mps = np.mean(np.abs(tcn_speeds - gt_eval_mps))
    mae_kmh = mae_mps * 3.6
    corr = np.corrcoef(tcn_speeds, gt_eval_mps)[0, 1]

    print(f"Test Slice Metrics (150s drive):")
    print(f"  Speed MAE: {mae_mps:.3f} m/s ({mae_kmh:.2f} km/h)")
    print(f"  Correlation to GT GPS: {corr:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=120, gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax1.plot(timestamps[20:], gt_eval_kmh, 'g-', lw=2.0, label='Ground Truth GPS Speed (km/h)')
    ax1.plot(timestamps[20:], tcn_speeds_kmh, 'm--', lw=1.8, label='TCN Predicted Forward Speed (km/h)')
    ax1.set_ylabel("Speed (km/h)", fontsize=11)
    ax1.set_title(f"TCN Edge AI Speed Filter Benchmark on S-S2 (MAE: {mae_kmh:.2f} km/h, Corr: {corr:.3f})", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    ax2.plot(timestamps[20:], tcn_zupt, 'r-', lw=1.5, label='TCN ZUPT Flag (Stationary Detection)')
    ax2.axhline(0.5, color='gray', linestyle=':', label='Threshold (0.5)')
    ax2.set_ylabel("ZUPT Prob", fontsize=11)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    report_img = os.path.join(REPORTS_DIR, "tcn_speed_evaluation.png")
    plt.savefig(report_img)
    print(f"Saved benchmark plot to: {report_img}")

if __name__ == '__main__':
    main()
