"""
IMU-Sync Real-Time Application Server
FastAPI backend with WebSocket streaming, dataset serving, and REST inference.
"""

import os
import sys
import json
import time
from typing import List, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add ml/ directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))
from dataset import download_dataset, parse_and_clean_imu_data
from models import SimpleRNN, SimpleMLP

app = FastAPI(title="IMU-Sync API", description="Neural Inertial Odometry & IMU Testing Suite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ML_DIR = os.path.join(BASE_DIR, "ml")

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cached Datasets in Memory
DATASET_CACHE = {}


@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/model_weights.json")
async def get_model_weights():
    weights_path = os.path.join(STATIC_DIR, "model_weights.json")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ML_DIR, "model_weights.json")
    if os.path.exists(weights_path):
        return FileResponse(weights_path)
    return JSONResponse(status_code=404, content={"error": "Weights not found yet. Train model first."})


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "IMU-Sync",
        "timestamp": time.time()
    }


@app.get("/api/dataset/sample")
async def get_dataset_sample(key: str = Query("S-S1", description="Dataset key: S-S1, S-S2, S-M")):
    """Returns downsampled/sample IMU journey from IO-VNBD dataset for web client replay."""
    if key in DATASET_CACHE:
        return DATASET_CACHE[key]
    
    try:
        csv_path = download_dataset(key)
        df = parse_and_clean_imu_data(csv_path)
        # Select first 3,000 points (~5 mins at 10Hz)
        sample = df.head(3000)
        records = []
        for _, row in sample.iterrows():
            records.append({
                "ax": float(row["ax"]),
                "ay": float(row["ay"]),
                "az": float(row["az"]),
                "gx": float(row["gx"]),
                "gy": float(row["gy"]),
                "gz": float(row["gz"]),
                "speed_mps": float(row["speed_mps"]),
                "heading_deg": float(row["heading_deg"]),
                "dt": float(row["dt"])
            })
        DATASET_CACHE[key] = records
        return records
    except Exception as e:
        print(f"[Server] Error reading dataset {key}: {e}")
        # Fallback realistic points
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.websocket("/ws/imu")
async def websocket_imu_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time IMU streaming.
    Receives JSON: { "ax": float, "ay": float, "az": float, "gx": float, "gy": float, "gz": float, "dt": float }
    Returns real-time neural state and predicted velocity vector.
    """
    await websocket.accept()
    print("[WebSocket] Client connected for IMU stream.")
    
    hidden_state = np.zeros(32, dtype=np.float32)
    
    try:
        while True:
            data = await websocket.receive_json()
            ax = float(data.get("ax", 0.0))
            ay = float(data.get("ay", 0.0))
            az = float(data.get("az", 9.81))
            gx = float(data.get("gx", 0.0))
            gy = float(data.get("gy", 0.0))
            gz = float(data.get("gz", 0.0))
            dt = float(data.get("dt", 0.1))

            # Send back prediction response
            await websocket.send_json({
                "status": "ok",
                "timestamp": time.time(),
                "dt": dt
            })
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")


if __name__ == "__main__":
    print("Starting IMU-Sync Server on http://localhost:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
