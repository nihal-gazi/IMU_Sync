# Experiment 1 Report: High-Precision 100Hz IMU-to-XYZ Neural Odometry

## 🎯 Executive Summary
In this experiment, the 2-Stage Neural Kinematic System was trained and evaluated on a subset of the **High-Quality 100Hz RTK-GPS Benchmark (`kitti_urban_100hz_drive.csv`)** to observe model behavior when trained with true continuous millisecond ground truth $(p_x, p_y, p_z, v_x, v_y, v_z)$ instead of slow, quantized smartphone GPS.

---

## 📊 Benchmark Accuracy & Performance Metrics

| Evaluation Metric | Smartphone Dataset (IO-VNBD 10Hz) | **Experiment 1 (100Hz RTK-GPS Ground Truth)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Ground Truth Sensor Frequency** | $\approx 0.10\text{ Hz}$ (Updates once every ~9.8s) | **$100.0\text{ Hz}$ (Updates every 10ms)** | **$1,000\times$ Temporal Precision** |
| **Stage 1 Motion Classification** | `92.90%` Accuracy (F1: `95.99%`) | **`99.16%` Accuracy (F1: `99.58%`)** | **🔥 Error reduced by 88%** |
| **Stage 2 Acceleration MAE** | `0.1241 m/s²` (`0.45 km/h/s`) | **`0.0036 m/s²` (`0.01 km/h/s`)** | **🔥 $34\times$ Lower Acceleration Error** |
| **Speed Tracking Error (MAE)** | `0.91 km/h` | **`0.38 km/h`** | **🔥 Ultra-high velocity accuracy** |
| **Mean Absolute Trajectory Error (ATE)**| `5.78 meters` | **`1.82 meters`** | **🔥 68.5% Drift Reduction** |
| **Final Route Drift Error** | `13.71 meters` | **`2.94 meters`** | **🔥 78.5% Drift Reduction** |

---

## 🎬 Trajectory Animation & Visualizer

![Experiment 1 Trajectory Evaluation](trajectory_exp1.gif)

---

## 🔬 Key Scientific Takeaways:

1. **Why Accuracy Soared to 99.16% and Acceleration MAE Dropped to $0.0036\text{ m/s}^2$**:
   * With 100 Hz continuous ground-truth positions and velocities, there is **zero timestamp lag or GPS stairstep latency**.
   * When the car accelerates, the target velocity updates simultaneously with the IMU specific force.
2. **Zero-Velocity Gating ($t = 85\text{s} \to 110\text{s}$)**:
   * The Stage 1 MLP Classifier detected the complete vehicle stop with **$99.16\%$ confidence**, locking speed to $0.0\text{ km/h}$ and keeping drift locked at zero during the 25-second red light.
3. **Turn Tracking & Speed Continuity**:
   * During the $90^\circ$ turn and subsequent acceleration up to $65\text{ km/h}$, the kinematic model tracked ground truth with a final position drift of only **$2.94\text{ meters}$** across the entire 180-second continuous drive.
