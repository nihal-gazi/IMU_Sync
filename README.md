# IMU-Sync: Neural Inertial Odometry & IMU Testing Suite

> Real-time Inertial Measurement Unit (IMU) testing application featuring an infinite draggable black canvas grid, compass-style vector tracking, live accelerometer/gyroscope oscilloscopes, and neural dead-reckoning models trained on the **IO-VNBD** benchmark dataset.

---

## 🌟 Overview & Features

- **Infinite Draggable Canvas Grid**:
  - Deep black sci-fi coordinate canvas with dynamic Level-of-Detail (LOD) grid cells and real-time scale markers.
  - Pan/drag, zoom (mouse wheel / touch / buttons), and one-click camera recentering.
  - Compass Entity: Central position marker with an arrow line pointing dynamically according to the ML model's predicted output vector.
  - Glowing motion trail with speed-coded heatmaps showing dead-reckoned vehicle trajectory.

- **Bottom Panel with Tabs**:
  - **Sensor Oscilloscope Tab**: Real-time dual 60 FPS oscilloscope graphs for 3-axis Accelerometer ($a_x, a_y, a_z$) and 3-axis Gyroscope ($g_x, g_y, g_z$) with digital telemetry.
  - **ML Vector & State Tab**: Polar Vector Radar visualizing magnitude and heading angle at every millisecond, along with an interactive 32-neuron RNN hidden state ($h_t \in \mathbb{R}^{32}$) activation bar visualizer.
  - **Data Stream & Simulator Tab**: Replay real driving journeys from the IO-VNBD dataset (with scrubbing and speed multiplier), connect live smartphone sensors, or test using manual acceleration/yaw joystick sliders.

- **Machine Learning Architectures (Trained on IO-VNBD)**:
  - **SimpleRNN (Sequential)**: Recurrent cell where each node receives input $x_t \in \mathbb{R}^6$ and previous hidden state $h_{t-1} \in \mathbb{R}^{32}$, producing an updated hidden representation $h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$ and projecting to 2D velocity vector $[v_x, v_y]$, speed, and direction vector $[\cos \theta, \sin \theta]$.
  - **SimpleMLP (Stateless)**: Multi-layer perceptron mapping instantaneous 6-axis IMU readings directly to 2D vector coordinates.
  - **Zero-Latency In-Browser Engine**: Pre-trained weights exported to `model_weights.json` allowing sub-millisecond client-side execution directly in JavaScript without backend lag.

---

## 📂 Project Structure

```
imu-sync/
├── ml/
│   ├── dataset.py            # IO-VNBD dataset loader, cleaner, and normalizer
│   ├── models.py             # SimpleRNN and SimpleMLP PyTorch architectures
│   ├── train.py              # Model training and evaluation script
│   ├── export_weights.py     # Exports weights to JSON & ONNX for client/server
│   └── scaler_params.json    # Normalization mean & std parameters
├── static/
│   ├── index.html            # Main web user interface
│   ├── style.css             # Cyberpunk dark theme styles
│   ├── app.js                # Main frontend controller & 60fps loop
│   ├── canvas_grid.js        # Infinite draggable canvas & compass visualizer
│   ├── sensor_chart.js       # Real-time 60fps dual oscilloscope
│   ├── ml_radar.js           # Polar vector radar & RNN hidden state bars
│   ├── nn_engine.js          # Pure JavaScript neural network inference engine
│   └── model_weights.json    # Exported trained weights for web inference
├── server.py                 # FastAPI backend with WebSocket streaming
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Export Weights (Optional - pre-trained weights included)
```bash
python ml/train.py
python ml/export_weights.py
```

### 3. Start the Application Server
```bash
python server.py
```
Open your browser at **`http://localhost:8000`**.

---

## 📱 Using Smartphone Sensors

1. Start the server on your computer.
2. Find your computer's local IP address (e.g., `192.168.1.50`).
3. Open `http://<your-ip>:8000` on your smartphone browser (connected to the same Wi-Fi).
4. Go to **Data Stream & Simulator** tab and tap **Enable Device Sensors**.
5. Move your phone and watch the center compass point navigate and trace your path on the infinite grid!

---

## 📊 Dataset Reference
- **IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark Dataset)**: [GitHub Repository](https://github.com/onyekpeu/IO-VNBD)
- Authors: Uche Onyekpe, Vasile Palade, Stratis Kanarachos, Alicja Szkolnik (Coventry University).
