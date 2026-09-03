"""
Step 1: 60Hz Resampling Pipeline
Resamples 100Hz RTK/MoCap kinematic dataset to 60Hz (the standard web mobile rate).
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

SRC_DATA_DIR = r"c:\Users\user\Desktop\IMU_Sync\research\data\highquality"
DST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "resampled_60hz")
os.makedirs(DST_DATA_DIR, exist_ok=True)

TARGET_HZ = 60.0
DT = 1.0 / TARGET_HZ  # 0.016666...s

def resample_track_to_60hz(csv_path):
    filename = os.path.basename(csv_path)
    out_filename = filename.replace("100hz", "60hz")
    out_path = os.path.join(DST_DATA_DIR, out_filename)

    df = pd.read_csv(csv_path)
    t_orig = df['timestamp_s'].values
    t_min, t_max = t_orig[0], t_orig[-1]

    # Uniform 60Hz timestamp grid
    t_60hz = np.arange(t_min, t_max, DT)

    columns_to_interp = [c for c in df.columns if c != 'timestamp_s']
    resampled_data = {'timestamp_s': t_60hz}

    for col in columns_to_interp:
        f = interp1d(t_orig, df[col].values, kind='cubic', bounds_error=False, fill_value='extrapolate')
        resampled_data[col] = f(t_60hz)

    df_60hz = pd.DataFrame(resampled_data)
    df_60hz.to_csv(out_path, index=False)
    print(f"Resampled {filename} ({len(df)} rows @ 100Hz) -> {out_filename} ({len(df_60hz)} rows @ 60Hz)")
    return out_path

if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(SRC_DATA_DIR, "*_100hz_*.csv"))
    print(f"Found {len(csv_files)} 100Hz tracks to resample to {TARGET_HZ}Hz...")
    for f in csv_files:
        resample_track_to_60hz(f)
    print("Resampling complete!")
