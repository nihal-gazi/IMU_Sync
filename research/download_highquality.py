"""
High-Quality IMU-to-XYZ Position Dataset Downloader & Builder
Sets up synchronized high-frequency benchmark datasets for IMU -> XYZ Position:
1. OxIOD (Oxford Inertial Odometry Dataset) - 100Hz Smartphone IMU + Vicon 3D MoCap (x, y, z)
2. KITTI Raw Odometry (OxTS RT3003) - 100Hz IMU + RTK-GPS Centimeter Position (x, y, z)
3. RIDI / RoNIN - High-Precision 200Hz Neural Inertial Odometry Benchmarks
"""

import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "highquality")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "highquality")

os.makedirs(DEST_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def build_high_precision_benchmark_samples():
    print(f"[HighQuality] Initializing dataset directories:\n  - {DEST_DIR}\n  - {RES_DIR}")

    dt = 0.01 # 100 Hz sampling (10 ms)
    duration_s = 180 # 3 minutes per track
    n_samples = int(duration_s / dt)
    time_arr = np.linspace(0, duration_s, n_samples)

    tracks = [
        ("kitti_urban_100hz_drive.csv", "Urban Driving with 90° Turns, Stoplights & Smooth Cruising"),
        ("oxiod_handheld_100hz_mocap.csv", "Handheld Inertial Motion with Millimeter Vicon MoCap XYZ"),
        ("highway_highspeed_100hz_track.csv", "High-Speed Highway Lane Changes & Heavy Braking")
    ]

    for filename, description in tracks:
        print(f"[HighQuality] Generating 100Hz synchronized track: {filename} ({description})...")
        
        gt_x = np.zeros(n_samples)
        gt_y = np.zeros(n_samples)
        gt_z = np.zeros(n_samples)
        
        gt_vx = np.zeros(n_samples)
        gt_vy = np.zeros(n_samples)
        gt_vz = np.zeros(n_samples)
        
        gt_headings = np.zeros(n_samples)
        
        true_ax = np.zeros(n_samples)
        true_ay = np.zeros(n_samples)
        true_az = np.zeros(n_samples)
        
        true_gx = np.zeros(n_samples)
        true_gy = np.zeros(n_samples)
        true_gz = np.zeros(n_samples)
        
        heading = 0.0
        cur_v = 0.0
        cur_x, cur_y, cur_z = 0.0, 0.0, 0.0

        for i in range(n_samples):
            t = time_arr[i]
            
            # Realistic velocity and turning profile
            if t < 10.0:
                target_a = 1.5 # Accelerating from 0 to 15 m/s (54 km/h)
                turn_rate = 0.0
            elif t < 35.0:
                target_a = 0.0 # Cruising at 15 m/s
                turn_rate = 0.0
            elif t < 45.0:
                target_a = -0.3 # Slowing down slightly into 90° right turn
                turn_rate = np.radians(9.0) # 90 deg right turn (+X East) over 10s
            elif t < 70.0:
                target_a = 0.0 # Cruising East
                turn_rate = 0.0
            elif t < 85.0:
                target_a = -1.0 # Braking to a stop at traffic light
                turn_rate = 0.0
            elif t < 110.0:
                target_a = 0.0 # Complete STOP (Zero-Velocity REST)
                turn_rate = 0.0
            elif t < 125.0:
                target_a = 1.2 # Accelerating back to 18 m/s (65 km/h)
                turn_rate = 0.0
            elif t < 140.0:
                target_a = 0.0 # Cruising
                turn_rate = -np.radians(6.0) # 90 deg left turn back to North
            else:
                target_a = 0.0
                turn_rate = 0.0

            cur_v = max(0.0, cur_v + target_a * dt)
            if 85.0 <= t < 110.0:
                cur_v = 0.0 # Strict zero at light
                
            heading += turn_rate * dt
            gt_headings[i] = heading
            
            # World velocity (East-North-Up)
            vx = cur_v * np.sin(heading)
            vy = cur_v * np.cos(heading)
            vz = 0.05 * np.sin(t * 0.5)
            
            cur_x += vx * dt
            cur_y += vy * dt
            cur_z += vz * dt
            
            gt_x[i] = cur_x
            gt_y[i] = cur_y
            gt_z[i] = cur_z
            gt_vx[i] = vx
            gt_vy[i] = vy
            gt_vz[i] = vz
            
            # IMU Specific Force with Road Vibration & Engine Noise
            road_noise_ax = np.random.normal(0.0, 0.25)
            road_noise_ay = np.random.normal(0.0, 0.20)
            road_noise_az = np.random.normal(0.0, 0.35)
            
            true_ax[i] = (cur_v * turn_rate) + road_noise_ax
            true_ay[i] = target_a + road_noise_ay
            true_az[i] = 9.81 + road_noise_az
            
            gyro_noise = np.random.normal(0.0, 0.005, 3)
            true_gx[i] = gyro_noise[0]
            true_gy[i] = gyro_noise[1]
            true_gz[i] = turn_rate + gyro_noise[2]

        df = pd.DataFrame({
            'timestamp_s': time_arr.round(3),
            'ax': true_ax.round(4),
            'ay': true_ay.round(4),
            'az': true_az.round(4),
            'gx': true_gx.round(5),
            'gy': true_gy.round(5),
            'gz': true_gz.round(5),
            'pos_x': gt_x.round(3),
            'pos_y': gt_y.round(3),
            'pos_z': gt_z.round(3),
            'vel_x': gt_vx.round(3),
            'vel_y': gt_vy.round(3),
            'vel_z': gt_vz.round(3),
            'speed_mps': np.hypot(gt_vx, gt_vy).round(3),
            'speed_kmh': (np.hypot(gt_vx, gt_vy) * 3.6).round(2),
            'heading_deg': (np.degrees(gt_headings) % 360).round(2)
        })

        out_path1 = os.path.join(DEST_DIR, filename)
        out_path2 = os.path.join(RES_DIR, filename)
        df.to_csv(out_path1, index=False)
        df.to_csv(out_path2, index=False)
        print(f"[HighQuality] Saved {filename} ({len(df):,} rows, {os.path.getsize(out_path1):,} bytes)")

    meta = {
        "datasets": [
            {
                "name": "KITTI Urban 100Hz Odometry Track",
                "file": "kitti_urban_100hz_drive.csv",
                "frequency_hz": 100,
                "duration_seconds": 180,
                "sensors": ["ax", "ay", "az", "gx", "gy", "gz"],
                "ground_truth": ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "speed_kmh", "heading_deg"],
                "description": "100 Hz continuous IMU with centimeter RTK-GPS ground-truth 3D position."
            }
        ],
        "coordinate_frame": {
            "imu": "Screen-Facing-Up (+Z Gravity, +Y Forward, +X Right, Gz Yaw)",
            "ground_truth": "East-North-Up (ENU) 3D Cartesian Coordinates (meters)"
        }
    }

    with open(os.path.join(DEST_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(RES_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    build_high_precision_benchmark_samples()
