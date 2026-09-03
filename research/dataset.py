"""
IO-VNBD Dataset Loader & Preprocessing Module for 2-Stage Kinematic System
Stage 1: Rest vs. Moving Motion Classification (MLP Zero-Velocity Detector)
Stage 2: Forward & Lateral Acceleration / Delta-Velocity Regression (Transformer)
Standardized into the canonical SCREEN-FACING-UP Reference Frame:
- +Z points UP out of the screen (Gravity +9.81 m/s²)
- +Y points FORWARD along vehicle motion (Forward acceleration a_fwd)
- +X points RIGHT (Lateral acceleration a_lat)
- Gz measures vehicle turning yaw rate around the Earth's vertical axis
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

DATASET_URLS = {
    "S-S1": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Categorised%20IOVNB%20Dataset/S%20(Driver%20A)/S1/S-S1.csv",
    "S-S2": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Categorised%20IOVNB%20Dataset/S%20(Driver%20A)/S2/S-S2.csv",
    "S-M": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Categorised%20IOVNB%20Dataset/M%20(Driver%20B)/M/S-M.csv"
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_dataset(dataset_key: str = "S-S1", dest_dir: Optional[str] = None) -> str:
    if dest_dir is None:
        dest_dir = CACHE_DIR
    os.makedirs(dest_dir, exist_ok=True)
    
    csv_path = os.path.join(dest_dir, f"{dataset_key}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10000:
        return csv_path
    
    url = DATASET_URLS.get(dataset_key)
    if not url:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(DATASET_URLS.keys())}")
    
    print(f"[Dataset] Downloading {dataset_key} from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    print(f"[Dataset] Downloaded {dataset_key} ({os.path.getsize(csv_path):,} bytes).")
    return csv_path


def compute_rodrigues_screen_up_matrix(g_vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(g_vec)
    if norm < 1e-4:
        return np.eye(3, dtype=np.float32)
    v_src = g_vec / norm
    v_dst = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    v = np.cross(v_src, v_dst)
    s = np.linalg.norm(v)
    c = float(np.dot(v_src, v_dst))

    if s < 1e-6:
        return np.eye(3, dtype=np.float32) if c > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R.astype(np.float32)


def parse_and_clean_imu_data(csv_path: str, align_screen_up: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    
    col_map = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'accel' in c_lower and 'x' in c_lower: col_map['ax'] = col
        elif 'accel' in c_lower and 'y' in c_lower: col_map['ay'] = col
        elif 'accel' in c_lower and 'z' in c_lower: col_map['az'] = col
        elif 'gyro' in c_lower and 'yaw' in c_lower: col_map['gz'] = col
        elif 'gyro' in c_lower and 'pitch' in c_lower: col_map['gy'] = col
        elif 'gyro' in c_lower and 'roll' in c_lower: col_map['gx'] = col
        elif 'gravity' in c_lower and 'x' in c_lower: col_map['grav_x'] = col
        elif 'gravity' in c_lower and 'y' in c_lower: col_map['grav_y'] = col
        elif 'gravity' in c_lower and 'z' in c_lower: col_map['grav_z'] = col
        elif 'gps speed' in c_lower: col_map['speed'] = col
        elif 'gps orientation' in c_lower or ('orientation' in c_lower and 'gps' in c_lower): col_map['heading'] = col
        elif 'orientation (yaw)' in c_lower: col_map['phone_yaw'] = col
        elif 'time since start' in c_lower: col_map['time'] = col

    raw_ax = df[col_map['ax']].values.astype(np.float32)
    raw_ay = df[col_map['ay']].values.astype(np.float32)
    raw_az = df[col_map['az']].values.astype(np.float32)
    raw_gx = df[col_map['gx']].values.astype(np.float32)
    raw_gy = df[col_map['gy']].values.astype(np.float32)
    raw_gz = df[col_map['gz']].values.astype(np.float32)

    has_grav = 'grav_x' in col_map and 'grav_y' in col_map and 'grav_z' in col_map
    if has_grav:
        grav_x = df[col_map['grav_x']].values.astype(np.float32)
        grav_y = df[col_map['grav_y']].values.astype(np.float32)
        grav_z = df[col_map['grav_z']].values.astype(np.float32)
    else:
        grav_x, grav_y, grav_z = raw_ax, raw_ay, raw_az

    n_samples = len(df)
    aligned_ax = np.zeros(n_samples, dtype=np.float32)
    aligned_ay = np.zeros(n_samples, dtype=np.float32)
    aligned_az = np.zeros(n_samples, dtype=np.float32)
    aligned_gx = np.zeros(n_samples, dtype=np.float32)
    aligned_gy = np.zeros(n_samples, dtype=np.float32)
    aligned_gz = np.zeros(n_samples, dtype=np.float32)

    if align_screen_up:
        for i in range(n_samples):
            g_vec = np.array([grav_x[i], grav_y[i], grav_z[i]], dtype=np.float32)
            R_up = compute_rodrigues_screen_up_matrix(g_vec)
            
            a_rot = R_up @ np.array([raw_ax[i], raw_ay[i], raw_az[i]], dtype=np.float32)
            w_rot = R_up @ np.array([raw_gx[i], raw_gy[i], raw_gz[i]], dtype=np.float32)
            
            aligned_ax[i], aligned_ay[i], aligned_az[i] = a_rot
            aligned_gx[i], aligned_gy[i], aligned_gz[i] = w_rot
    else:
        aligned_ax, aligned_ay, aligned_az = raw_ax, raw_ay, raw_az
        aligned_gx, aligned_gy, aligned_gz = raw_gx, raw_gy, raw_gz

    clean_df = pd.DataFrame({
        'time_ms': df[col_map['time']].values.astype(np.float64),
        'ax': aligned_ax,
        'ay': aligned_ay,
        'az': aligned_az,
        'gx': aligned_gx,
        'gy': aligned_gy,
        'gz': aligned_gz,
        'speed_mps': (df[col_map['speed']].values / 3.6).astype(np.float32),
        'heading_deg': df[col_map['heading']].values.astype(np.float32),
        'phone_yaw': df[col_map.get('phone_yaw', col_map['heading'])].values.astype(np.float32)
    })

    dt = np.diff(clean_df['time_ms'].values, prepend=clean_df['time_ms'].iloc[0]) / 1000.0
    dt[0] = 0.1
    dt[dt <= 0] = 0.1
    clean_df['dt'] = dt.astype(np.float32)

    clean_df['fwd_step'] = (clean_df['speed_mps'] * clean_df['dt']).astype(np.float32)
    clean_df['lat_step'] = 0.0

    heading_rad = np.radians(clean_df['heading_deg'].values)
    clean_df['dx_global'] = (clean_df['fwd_step'] * np.sin(heading_rad)).astype(np.float32)
    clean_df['dy_global'] = (clean_df['fwd_step'] * np.cos(heading_rad)).astype(np.float32)
    clean_df['dx'] = clean_df['dx_global']
    clean_df['dy'] = clean_df['dy_global']

    return clean_df


class Normalizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean


def prepare_acceleration_dataset(
    dataset_keys: List[str] = ["S-S1", "S-S2"],
    window_size: int = 10,
    stride: int = 2,
    rest_speed_threshold_mps: float = 0.28
) -> Tuple[Dict, Normalizer, Normalizer]:
    all_dfs = []
    for key in dataset_keys:
        csv_path = download_dataset(key)
        df = parse_and_clean_imu_data(csv_path, align_screen_up=True)
        print(f"[Dataset] Processed {key}: {len(df):,} records, mean speed {df['speed_mps'].mean()*3.6:.1f} km/h")
        all_dfs.append(df)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    feature_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    feat_norm = Normalizer()
    feat_norm.fit(combined_df[feature_cols].values)
    
    windows_X = []
    motion_labels = [] # 0: REST, 1: MOVING
    accel_targets_dv = [] # [dv_lateral, dv_forward] in m/s (acceleration over 1 second)
    
    for df in all_dfs:
        norm_imu = feat_norm.transform(df[feature_cols].values)
        speeds = df['speed_mps'].values
        
        n_samples = len(df)
        for start in range(0, n_samples - window_size, stride):
            end = start + window_size
            w_x = norm_imu[start:end]
            
            # Acceleration / Delta Velocity over 1.0s window
            dv_fwd = float(speeds[end - 1] - speeds[start]) # v_end - v_start
            dv_lat = 0.0 # Non-drifting vehicle
            avg_speed = float(np.mean(speeds[start:end]))
            
            is_moving = 1 if avg_speed >= rest_speed_threshold_mps else 0
            
            windows_X.append(w_x)
            motion_labels.append(is_moving)
            accel_targets_dv.append([dv_lat, dv_fwd])
            
    windows_X = np.array(windows_X, dtype=np.float32)
    motion_labels = np.array(motion_labels, dtype=np.int64)
    accel_targets_dv = np.array(accel_targets_dv, dtype=np.float32)
    
    # Fit Target Normalizer for Acceleration Delta-Velocity [dv_lat, dv_fwd]
    target_norm = Normalizer()
    target_norm.fit(accel_targets_dv)
    norm_dv = target_norm.transform(accel_targets_dv)
    
    split_idx = int(len(windows_X) * 0.85)
    
    dataset_package = {
        'cls': {
            'train_X': windows_X[:split_idx],
            'train_y': motion_labels[:split_idx],
            'val_X': windows_X[split_idx:],
            'val_y': motion_labels[split_idx:]
        },
        'accel': {
            'train_X': windows_X[:split_idx],
            'train_y': norm_dv[:split_idx],
            'val_X': windows_X[split_idx:],
            'val_y': norm_dv[split_idx:]
        }
    }
    
    n_rest = int(np.sum(motion_labels == 0))
    n_moving = int(np.sum(motion_labels == 1))
    print(f"[Dataset] Total Windows: {len(windows_X):,} | Rest: {n_rest:,} ({n_rest/len(windows_X)*100:.1f}%), Moving: {n_moving:,} ({n_moving/len(windows_X)*100:.1f}%)")
    print(f"[Dataset] Acceleration Target Means: {target_norm.mean.round(4)}, Stds: {target_norm.std.round(4)}")
    
    scaler_dict = {
        'features': {
            'names': feature_cols,
            'mean': feat_norm.mean.tolist(),
            'std': feat_norm.std.tolist()
        },
        'targets': {
            'names': ['dv_lateral', 'dv_forward'],
            'mean': target_norm.mean.tolist(),
            'std': target_norm.std.tolist()
        },
        'target_type': 'ACCELERATION_DELTA_V',
        'window_size': window_size,
        'canonical_frame': 'SCREEN_FACING_UP'
    }
    
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"[Dataset] Saved Acceleration normalization parameters to {scaler_path}")
    
    return dataset_package, feat_norm, target_norm
