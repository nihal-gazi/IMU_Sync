import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import download_dataset

csv_path = download_dataset('S-S1')
raw_df = pd.read_csv(csv_path, encoding='latin1')
raw_df.columns = [c.strip() for c in raw_df.columns]

# Find columns
gps_h_col = [c for c in raw_df.columns if 'gps orientation' in c.lower()][0]
gps_sp_col = [c for c in raw_df.columns if 'gps speed' in c.lower()][0]
yaw_col = [c for c in raw_df.columns if 'gyroscope yaw' in c.lower()][0]
time_col = [c for c in raw_df.columns if 'time since' in c.lower()][0]

# Search for a sharp ~90 degree turn
turn_candidates = []
for i in range(len(raw_df) - 40):
    h0 = raw_df[gps_h_col].iloc[i]
    h1 = raw_df[gps_h_col].iloc[i + 30] # 3.0 seconds
    diff = abs((h1 - h0 + 180) % 360 - 180)
    sp = raw_df[gps_sp_col].iloc[i]
    if 80 <= diff <= 110 and sp > 5.0:
        turn_candidates.append((i, diff, h0, h1, sp))

best_i, best_diff, h0, h1, sp = turn_candidates[0]
print(f"--- 90-Degree Turn Event: Row {best_i} to {best_i + 30} ---")
print(f"Heading Change: {h0:.1f}° -> {h1:.1f}° (Δ = {best_diff:.1f}°), Speed = {sp:.1f} km/h\n")

# Print all 24 columns in formatted Markdown table
sample = raw_df.iloc[best_i : best_i + 25]

# Rename columns for clean readability
clean_cols = {
    raw_df.columns[0]: 'GPS_Lat',
    raw_df.columns[1]: 'GPS_Lon',
    raw_df.columns[2]: 'GPS_Alt',
    raw_df.columns[3]: 'Speed_Kmh',
    raw_df.columns[4]: 'GPS_Acc',
    raw_df.columns[5]: 'GPS_Heading',
    raw_df.columns[6]: 'Sats',
    raw_df.columns[7]: 'Time_ms',
    raw_df.columns[8]: 'Date',
    raw_df.columns[9]: 'Ax',
    raw_df.columns[10]: 'Ay',
    raw_df.columns[11]: 'Az',
    raw_df.columns[12]: 'Grav_x',
    raw_df.columns[13]: 'Grav_y',
    raw_df.columns[14]: 'Grav_z',
    raw_df.columns[15]: 'Gyro_Yaw (Gz)',
    raw_df.columns[16]: 'Gyro_Pitch (Gy)',
    raw_df.columns[17]: 'Gyro_Roll (Gx)',
    raw_df.columns[18]: 'Mag_x',
    raw_df.columns[19]: 'Mag_y',
    raw_df.columns[20]: 'Mag_z',
    raw_df.columns[21]: 'Orient_Yaw',
    raw_df.columns[22]: 'Orient_Pitch',
    raw_df.columns[23]: 'Orient_Roll'
}

sample = sample.rename(columns=clean_cols)
# Output markdown table
print(sample.to_markdown(index=True))
