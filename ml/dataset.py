"""
IO-VNBD Dataset Loader & Preprocessing Module
Extracts smartphone accelerometer and gyroscope data and prepares normalized
sequential and non-sequential training sets for RNN and MLP models.
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

# Base URLs for IO-VNBD Smartphone Datasets on GitHub (Git LFS)
DATASET_URLS = {
    "S-S1": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Categorised%20IOVNB%20Dataset/S%20(Driver%20A)/S1/S-S1.csv",
    "S-S2": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Categorised%20IOVNB%20Dataset/S%20(Driver%20A)/S2/S-S2.csv",
    "S-M": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Categorised%20IOVNB%20Dataset/M%20(Driver%20B)/M/S-M.csv"
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def download_dataset(dataset_key: str = "S-S1", dest_dir: Optional[str] = None) -> str:
    """Downloads dataset CSV if not already cached."""
    if dest_dir is None:
        dest_dir = CACHE_DIR
    os.makedirs(dest_dir, exist_ok=True)
    
    csv_path = os.path.join(dest_dir, f"{dataset_key}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10000:
        print(f"[Dataset] Using cached {dataset_key} at {csv_path}")
        return csv_path
    
    url = DATASET_URLS.get(dataset_key)
    if not url:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(DATASET_URLS.keys())}")
    
    print(f"[Dataset] Downloading {dataset_key} from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    print(f"[Dataset] Downloaded {dataset_key} ({os.path.getsize(csv_path):,} bytes).")
    return csv_path


def parse_and_clean_imu_data(csv_path: str) -> pd.DataFrame:
    """
    Parses IO-VNBD smartphone CSV file with Latin-1 encoding,
    cleans column names, and extracts IMU signals and ground truth motion targets.
    """
    # IO-VNBD CSV files use Latin-1 due to ° and m/s² characters
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
        elif 'gps speed' in c_lower: col_map['speed'] = col
        elif 'gps orientation' in c_lower or ('orientation' in c_lower and 'gps' in c_lower): col_map['heading'] = col
        elif 'latitude' in c_lower: col_map['lat'] = col
        elif 'longitude' in c_lower: col_map['lon'] = col
        elif 'time since start' in c_lower: col_map['time'] = col

    # Check required fields
    required = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'speed', 'heading', 'time']
    for req in required:
        if req not in col_map:
            raise KeyError(f"Missing required IMU/GPS column mapping for '{req}'. Available: {df.columns.tolist()}")

    clean_df = pd.DataFrame({
        'time_ms': df[col_map['time']].values.astype(np.float64),
        'ax': df[col_map['ax']].values.astype(np.float32),
        'ay': df[col_map['ay']].values.astype(np.float32),
        'az': df[col_map['az']].values.astype(np.float32),
        'gx': df[col_map['gx']].values.astype(np.float32),
        'gy': df[col_map['gy']].values.astype(np.float32),
        'gz': df[col_map['gz']].values.astype(np.float32),
        'speed_mps': (df[col_map['speed']].values / 3.6).astype(np.float32), # Kmh to m/s
        'heading_deg': df[col_map['heading']].values.astype(np.float32),
    })

    # Compute continuous heading in radians and 2D velocity / displacement vectors
    heading_rad = np.radians(clean_df['heading_deg'].values)
    clean_df['heading_rad'] = heading_rad
    
    # Velocity vector (vx = East, vy = North)
    clean_df['vx'] = clean_df['speed_mps'] * np.sin(heading_rad)
    clean_df['vy'] = clean_df['speed_mps'] * np.cos(heading_rad)
    
    # Time delta in seconds (nominal 10Hz = 0.1s)
    dt = np.diff(clean_df['time_ms'].values, prepend=clean_df['time_ms'].iloc[0]) / 1000.0
    dt[0] = 0.1
    dt[dt <= 0] = 0.1 # handle duplicates or resets
    clean_df['dt'] = dt.astype(np.float32)
    
    # Step displacement vector (dx, dy)
    clean_df['dx'] = (clean_df['vx'] * clean_df['dt']).astype(np.float32)
    clean_df['dy'] = (clean_df['vy'] * clean_df['dt']).astype(np.float32)
    
    # Cumulative trajectory coords (X, Y in meters)
    clean_df['pos_x'] = np.cumsum(clean_df['dx'])
    clean_df['pos_y'] = np.cumsum(clean_df['dy'])

    # Direction unit vector
    clean_df['dir_x'] = np.sin(heading_rad).astype(np.float32)
    clean_df['dir_y'] = np.cos(heading_rad).astype(np.float32)

    return clean_df


class Normalizer:
    """Z-Score standard normalizer for IMU features and vector targets."""
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        # Avoid division by zero
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted.")
        return (data * self.std) + self.mean

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else []
        }

    def from_dict(self, d: Dict[str, List[float]]):
        self.mean = np.array(d["mean"], dtype=np.float32)
        self.std = np.array(d["std"], dtype=np.float32)


def prepare_datasets(
    dataset_keys: List[str] = ["S-S1"],
    seq_len: int = 30,
    stride: int = 5
) -> Tuple[Dict, Normalizer, Normalizer]:
    """
    Downloads datasets, normalizes features, and creates sequential (RNN) and flat (MLP) datasets.
    """
    all_dfs = []
    for key in dataset_keys:
        csv_path = download_dataset(key)
        df = parse_and_clean_imu_data(csv_path)
        print(f"[Dataset] Processed {key}: {len(df):,} records, mean speed {df['speed_mps'].mean()*3.6:.1f} km/h")
        all_dfs.append(df)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Feature columns: [ax, ay, az, gx, gy, gz]
    feature_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    # Target columns: 2D vector [vx, vy] or [dx, dy] & heading [dir_x, dir_y]
    # We predict 2D velocity vector [vx, vy] which scales smoothly to displacement dx=vx*dt, dy=vy*dt
    target_cols = ['vx', 'vy', 'speed_mps', 'dir_x', 'dir_y']
    
    X_raw = combined_df[feature_cols].values
    y_raw = combined_df[target_cols].values
    
    # Fit normalizers
    feat_norm = Normalizer()
    feat_norm.fit(X_raw)
    
    target_norm = Normalizer()
    target_norm.fit(y_raw)
    
    X_norm = feat_norm.transform(X_raw)
    y_norm = target_norm.transform(y_raw)
    
    # Create Flat dataset for MLP
    split_idx = int(len(X_norm) * 0.85)
    mlp_data = {
        'train_X': X_norm[:split_idx],
        'train_y': y_norm[:split_idx],
        'val_X': X_norm[split_idx:],
        'val_y': y_norm[split_idx:],
        'raw_val_df': combined_df.iloc[split_idx:].copy()
    }
    
    # Create Sequential dataset for RNN
    seq_X, seq_y = [], []
    for df in all_dfs:
        df_X = feat_norm.transform(df[feature_cols].values)
        df_y = target_norm.transform(df[target_cols].values)
        
        n_samples = len(df_X)
        for start in range(0, n_samples - seq_len, stride):
            end = start + seq_len
            seq_X.append(df_X[start:end])
            seq_y.append(df_y[start:end])
            
    seq_X = np.array(seq_X, dtype=np.float32)
    seq_y = np.array(seq_y, dtype=np.float32)
    
    seq_split = int(len(seq_X) * 0.85)
    rnn_data = {
        'train_X': seq_X[:seq_split],
        'train_y': seq_y[:seq_split],
        'val_X': seq_X[seq_split:],
        'val_y': seq_y[seq_split:]
    }
    
    # Save scaler params to JSON
    scaler_dict = {
        'features': {
            'names': feature_cols,
            'mean': feat_norm.mean.tolist(),
            'std': feat_norm.std.tolist()
        },
        'targets': {
            'names': target_cols,
            'mean': target_norm.mean.tolist(),
            'std': target_norm.std.tolist()
        }
    }
    
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"[Dataset] Saved normalization parameters to {scaler_path}")
    
    return {
        'mlp': mlp_data,
        'rnn': rnn_data,
        'scaler_dict': scaler_dict,
        'sample_df': all_dfs[0].head(1000) # For sample replay
    }, feat_norm, target_norm


if __name__ == "__main__":
    print("Testing dataset pipeline...")
    data, feat_norm, target_norm = prepare_datasets(["S-S1"])
    print(f"MLP Train X shape: {data['mlp']['train_X'].shape}, y shape: {data['mlp']['train_y'].shape}")
    print(f"RNN Train X shape: {data['rnn']['train_X'].shape}, y shape: {data['rnn']['train_y'].shape}")
