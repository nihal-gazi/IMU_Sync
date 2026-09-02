# IMU-Sync: Neural Inertial Odometry & IMU Testing Suite (React + ONNX Web)

> High-performance standalone React-JS frontend for testing IMU sensor data, dead-reckoning navigation with neural network models trained on the **IO-VNBD** benchmark dataset, and running client-side with **ONNX Runtime Web**.

---

## 🌟 Overview & Key Features

- **Frontend-Only Architecture (ONNX Runtime Web)**:
  - 100% Client-side inference powered by WebAssembly (`onnxruntime-web`).
  - Pre-compiled **SimpleRNN (Sequential)** and **SimpleMLP (Stateless)** ONNX models.
  - Zero backend requirement for production deployment (ready for GitHub Pages / Vercel / Netlify).

- **Black Infinite Draggable Canvas Grid**:
  - Deep black coordinate grid with dynamic Level-of-Detail (LOD) grid cells and real-time scale markers.
  - Mouse / touch drag panning, mouse wheel zoom, and camera recentering.
  - Compass Entity: Central position marker with an arrow line pointing dynamically according to the ONNX model's output vector.
  - Speed-coded glowing dead-reckoning trajectory trail.

- **Bottom Panel with Tabs**:
  - **Sensor Oscilloscope Tab**: Real-time dual 60 FPS oscilloscope graphs for 3-axis Accelerometer ($a_x, a_y, a_z$) and 3-axis Gyroscope ($g_x, g_y, g_z$).
  - **ML Vector & State Tab**: Polar Vector Radar visualizing vector orientation and magnitude at every millisecond, along with an interactive 32-neuron RNN hidden state ($h_t \in \mathbb{R}^{32}$) activation visualizer.
  - **Data Stream & Simulator Tab**: Replay real driving journeys from the IO-VNBD dataset (with scrubber and speed multipliers), stream live smartphone IMU sensors via `DeviceMotionEvent`, or test with manual joystick sliders.

- **Isolated Research Environment (`/research`)**:
  - All Python scripts, PyTorch neural architectures, datasets, and ONNX export pipelines are organized inside `research/` (kept in `.gitignore`).

---

## 📂 Project Structure

```
imu-sync/
├── public/
│   ├── models/
│   │   ├── rnn_model.onnx       # Compiled SimpleRNN ONNX model
│   │   ├── mlp_model.onnx       # Compiled SimpleMLP ONNX model
│   │   └── scaler_params.json   # Normalization parameters
│   └── data/
│       └── sample_journey.json  # 3,000 IO-VNBD dataset driving frames
├── src/
│   ├── components/
│   │   ├── TopNav.jsx           # Telemetry HUD header
│   │   ├── InfiniteCanvas.jsx   # Infinite draggable grid & compass arrow
│   │   ├── SensorOscilloscope.jsx # 60 FPS Accel & Gyro oscilloscopes
│   │   ├── MLVectorRadar.jsx    # Polar vector radar & RNN hidden state bars
│   │   └── BottomPanel.jsx      # Tabs, replay controls & IMU simulator
│   ├── services/
│   │   └── onnxInference.js     # ONNX Runtime Web inference engine
│   ├── App.jsx                  # Main application orchestrator & telemetry loop
│   ├── main.jsx                 # React root
│   └── index.css                # Cyberpunk dark theme styling
├── package.json                 # React, Vite, ONNX Runtime Web dependencies
├── vite.config.js               # Vite bundler configuration
└── research/                    # [Ignored in git] ML Training & Experiments
    ├── dataset.py               # IO-VNBD dataset loader & normalizer
    ├── models.py                # PyTorch SimpleRNN and SimpleMLP
    ├── train.py                 # GPU training script
    ├── export_onnx.py           # PyTorch -> ONNX exporter
    └── data/                    # Cached IO-VNBD CSV files
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Development Server
```bash
npm run dev
```
Open **`http://localhost:3000`** in your browser.

### 3. Build for Production
```bash
npm run build
```
The static build is output to `dist/` and can be served with any static web host.

---

## 🔬 AI / Research Development (`/research`)

All machine learning development and dataset exploration is contained inside `research/`:

```bash
cd research
pip install -r requirements.txt

# 1. Train models on IO-VNBD dataset
python train.py

# 2. Export updated ONNX models to public/models/
python export_onnx.py
```

---

## 📊 Dataset Reference
- **IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark Dataset)**: [GitHub Repository](https://github.com/onyekpeu/IO-VNBD)
- Authors: Uche Onyekpe, Vasile Palade, Stratis Kanarachos, Alicja Szkolnik (Coventry University).
