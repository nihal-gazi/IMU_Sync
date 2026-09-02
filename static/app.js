/**
 * Main Application Orchestrator for IMU-Sync
 * Coordinates Canvas Grid, Sensor Charts, ML Vector Radar,
 * Neural Inference, Dataset Replay, and Live Sensors.
 */

document.addEventListener('DOMContentLoaded', async () => {
  console.log('[IMU-Sync] Initializing application suite...');

  // 1. Initialize Neural Network Inference Engine
  await window.nnEngine.init('model_weights.json');

  // 2. Initialize Draggable Infinite Canvas Grid
  const canvasGrid = new CanvasGrid('gridCanvas');

  // 3. Initialize Sensor Oscilloscopes (60 FPS)
  const accelChart = new SensorOscilloscope('accelCanvas', {
    maxSamples: 150,
    channels: [
      { name: 'Ax', color: '#ff4757' }, // Red
      { name: 'Ay', color: '#2ed573' }, // Green
      { name: 'Az', color: '#1e90ff' }  // Blue
    ]
  });

  const gyroChart = new SensorOscilloscope('gyroCanvas', {
    maxSamples: 150,
    channels: [
      { name: 'Gx', color: '#ffa502' }, // Orange
      { name: 'Gy', color: '#00d2d3' }, // Cyan
      { name: 'Gz', color: '#ff4757' }  // Magenta
    ]
  });

  // 4. Initialize ML Vector Radar
  const mlRadar = new MLRadar('radarCanvas');

  // App Global State
  const state = {
    mode: 'rnn', // 'rnn' | 'mlp'
    source: 'replay', // 'replay' | 'phone' | 'sim'
    isPlaying: true,
    speedMultiplier: 1,
    datasetKey: 'S-S1',
    datasetData: null,
    replayIndex: 0,
    lastTickTime: performance.now(),
    
    // Live sensor values
    currentSensor: [0.0, 0.0, 9.81, 0.0, 0.0, 0.0], // [ax, ay, az, gx, gy, gz]
    currentDt: 0.1, // 100ms standard

    // Simulator State
    simAy: 0.0,
    simGz: 0.0,
    demoMode: null, // 'circle' | 'fig8' | null
    demoTime: 0
  };

  // 5. Load Default Sample Dataset for Replay
  async function loadDataset(key) {
    try {
      const resp = await fetch(`/api/dataset/sample?key=${key}`);
      if (resp.ok) {
        state.datasetData = await resp.json();
        console.log(`[Replay] Loaded ${key} dataset with ${state.datasetData.length} records.`);
      } else {
        generateSyntheticTrajectory();
      }
    } catch (e) {
      console.warn('[Replay] Server API unavailable, generating synthetic realistic driving trajectory.');
      generateSyntheticTrajectory();
    }
    
    state.replayIndex = 0;
    const scrubber = document.getElementById('replayScrubber');
    if (scrubber && state.datasetData) {
      scrubber.max = state.datasetData.length - 1;
      scrubber.value = 0;
    }
  }

  function generateSyntheticTrajectory() {
    // Generates 1500 steps of realistic driving motion (accel, brake, turns)
    const points = [];
    let heading = 0;
    let speed = 0;
    for (let i = 0; i < 1500; i++) {
      const t = i * 0.1;
      // Periodic turns and acceleration
      const throttle = Math.sin(t * 0.08) > 0.2 ? 1.8 : 0.0;
      const turn = Math.sin(t * 0.15) * 0.4;
      
      speed = Math.max(0, speed * 0.98 + throttle * 0.1);
      heading += turn * 0.1;
      
      const ax = (Math.random() - 0.5) * 0.2;
      const ay = throttle + (Math.random() - 0.5) * 0.3;
      const az = 9.81 + (Math.random() - 0.5) * 0.4;
      const gx = (Math.random() - 0.5) * 0.05;
      const gy = (Math.random() - 0.5) * 0.05;
      const gz = turn + (Math.random() - 0.5) * 0.02;

      points.push({ ax, ay, az, gx, gy, gz, dt: 0.1 });
    }
    state.datasetData = points;
  }

  await loadDataset('S-S1');

  // ==========================================================================
  // Main Processing & Simulation Tick (Runs at 10Hz - 60Hz)
  // ==========================================================================

  function processTick() {
    const now = performance.now();
    let dt = (now - state.lastTickTime) / 1000.0;
    state.lastTickTime = now;
    if (dt <= 0 || dt > 0.5) dt = 0.1;

    let imu = [0, 0, 9.81, 0, 0, 0];

    // Source 1: Dataset Replay
    if (state.source === 'replay' && state.datasetData && state.datasetData.length > 0) {
      if (state.isPlaying) {
        const stepInc = state.speedMultiplier;
        state.replayIndex = (state.replayIndex + stepInc) % state.datasetData.length;
        const scrubber = document.getElementById('replayScrubber');
        if (scrubber) scrubber.value = state.replayIndex;
      }
      
      const row = state.datasetData[state.replayIndex];
      if (row) {
        imu = [row.ax, row.ay, row.az, row.gx, row.gy, row.gz];
        state.currentDt = row.dt || 0.1;
      }
    } 
    // Source 2: Manual Simulator & Demos
    else if (state.source === 'sim') {
      if (state.demoMode === 'circle') {
        state.demoTime += dt;
        state.simAy = 2.0; // constant forward drive
        state.simGz = 0.5; // continuous right turn
      } else if (state.demoMode === 'fig8') {
        state.demoTime += dt;
        state.simAy = 2.5;
        state.simGz = Math.sin(state.demoTime * 0.8) * 0.7; // figure-8 weaving
      }
      imu = [
        (Math.random() - 0.5) * 0.1,
        state.simAy + (Math.random() - 0.5) * 0.2,
        9.81 + (Math.random() - 0.5) * 0.1,
        (Math.random() - 0.5) * 0.02,
        (Math.random() - 0.5) * 0.02,
        state.simGz + (Math.random() - 0.5) * 0.02
      ];
      state.currentDt = dt;
    } 
    // Source 3: Live Smartphone
    else if (state.source === 'phone') {
      imu = state.currentSensor;
      state.currentDt = dt;
    }

    // 1. Run Neural Network Model Prediction Step
    const pred = window.nnEngine.predictStep(imu, state.currentDt);

    // 2. Feed readings to Oscilloscope charts
    accelChart.pushSample([imu[0], imu[1], imu[2]]);
    gyroChart.pushSample([imu[3], imu[4], imu[5]]);

    // 3. Update Motion on Draggable Canvas Grid (Center point & Arrow)
    canvasGrid.updateMotion(pred.dx, pred.dy, pred.headingDeg, pred.headingRad, pred.speedKmh);

    // 4. Update ML Tab Polar Vector & Hidden States
    mlRadar.update(pred);

    // 5. Update Digital HUD Labels
    updateHUD(imu, pred);
  }

  function updateHUD(imu, pred) {
    // Sensor Labels in Sensor Tab
    const setTxt = (id, txt) => {
      const el = document.getElementById(id);
      if (el) el.textContent = txt;
    };

    setTxt('valAx', (imu[0] >= 0 ? '+' : '') + imu[0].toFixed(2));
    setTxt('valAy', (imu[1] >= 0 ? '+' : '') + imu[1].toFixed(2));
    setTxt('valAz', (imu[2] >= 0 ? '+' : '') + imu[2].toFixed(2));

    setTxt('valGx', (imu[3] >= 0 ? '+' : '') + imu[3].toFixed(3));
    setTxt('valGy', (imu[4] >= 0 ? '+' : '') + imu[4].toFixed(3));
    setTxt('valGz', (imu[5] >= 0 ? '+' : '') + imu[5].toFixed(3));

    // Top Header HUD
    setTxt('hudPos', `${canvasGrid.posX.toFixed(2)}m, ${canvasGrid.posY.toFixed(2)}m`);
    setTxt('hudSpeed', `${pred.speedKmh.toFixed(1)} km/h`);
    setTxt('hudHeading', `${pred.headingDeg.toFixed(1)}°`);
    setTxt('hudLatency', `${pred.latencyMs.toFixed(2)} ms`);

    // Scrubber Time
    const scrubLabel = document.getElementById('scrubTime');
    if (scrubLabel && state.datasetData) {
      const curSec = Math.floor(state.replayIndex * 0.1);
      const totalSec = Math.floor(state.datasetData.length * 0.1);
      scrubLabel.textContent = `${formatTime(curSec)} / ${formatTime(totalSec)}`;
    }
  }

  function formatTime(s) {
    const m = Math.floor(s / 60);
    const rem = Math.floor(s % 60);
    return `${m < 10 ? '0' : ''}${m}:${rem < 10 ? '0' : ''}${rem}`;
  }

  // Animation Frame Loop for 60 FPS charts & smoothness
  function mainLoop() {
    processTick();
    accelChart.render();
    gyroChart.render();
    requestAnimationFrame(mainLoop);
  }
  requestAnimationFrame(mainLoop);

  // ==========================================================================
  // UI Event Handlers & Tabs
  // ==========================================================================

  // Tab Switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');

      // Expand bottom panel if collapsed
      const panel = document.getElementById('bottomPanel');
      if (panel) panel.classList.remove('collapsed');

      // Resize charts
      accelChart.initCanvasSize();
      gyroChart.initCanvasSize();
      mlRadar.initCanvasSize();
    });
  });

  // Bottom Panel Collapse / Expand
  const btnTogglePanel = document.getElementById('btnTogglePanel');
  if (btnTogglePanel) {
    btnTogglePanel.addEventListener('click', () => {
      const panel = document.getElementById('bottomPanel');
      panel.classList.toggle('collapsed');
      setTimeout(() => canvasGrid.initCanvasSize(), 350);
    });
  }

  // Architecture Selection: RNN vs MLP
  const btnSelectRNN = document.getElementById('btnSelectRNN');
  const btnSelectMLP = document.getElementById('btnSelectMLP');
  const hudModel = document.getElementById('hudModel');

  btnSelectRNN.addEventListener('click', () => {
    state.mode = 'rnn';
    window.nnEngine.setMode('rnn');
    btnSelectRNN.classList.add('active');
    btnSelectMLP.classList.remove('active');
    if (hudModel) hudModel.textContent = 'RNN (Sequential)';
  });

  btnSelectMLP.addEventListener('click', () => {
    state.mode = 'mlp';
    window.nnEngine.setMode('mlp');
    btnSelectMLP.classList.add('active');
    btnSelectRNN.classList.remove('active');
    if (hudModel) hudModel.textContent = 'MLP (Stateless)';
  });

  // Canvas Viewport Controls
  document.getElementById('btnRecenter').addEventListener('click', () => canvasGrid.recenter());
  document.getElementById('btnClearTrail').addEventListener('click', () => canvasGrid.clearTrail());
  document.getElementById('btnZoomIn').addEventListener('click', () => canvasGrid.setZoom(canvasGrid.zoom * 1.25));
  document.getElementById('btnZoomOut').addEventListener('click', () => canvasGrid.setZoom(canvasGrid.zoom * 0.8));
  document.getElementById('btnResetZoom').addEventListener('click', () => canvasGrid.setZoom(1.0));

  // Source Toggle Button
  const btnSourceToggle = document.getElementById('btnSourceToggle');
  const sourceLabel = document.getElementById('sourceLabel');
  btnSourceToggle.addEventListener('click', () => {
    if (state.source === 'replay') {
      state.source = 'sim';
      sourceLabel.textContent = 'Source: Simulator';
    } else if (state.source === 'sim') {
      state.source = 'replay';
      sourceLabel.textContent = 'Source: Replay';
    }
  });

  // Replay Controls
  const btnPlayPause = document.getElementById('btnPlayPause');
  btnPlayPause.addEventListener('click', () => {
    state.isPlaying = !state.isPlaying;
    btnPlayPause.textContent = state.isPlaying ? 'Pause Replay' : 'Play Replay';
  });

  document.getElementById('btnRestartReplay').addEventListener('click', () => {
    state.replayIndex = 0;
    canvasGrid.clearTrail();
    window.nnEngine.resetHiddenState();
  });

  const scrubber = document.getElementById('replayScrubber');
  scrubber.addEventListener('input', (e) => {
    state.replayIndex = parseInt(e.target.value, 10);
  });

  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.speedMultiplier = parseInt(btn.dataset.speed, 10);
    });
  });

  document.getElementById('selectDataset').addEventListener('change', (e) => {
    state.datasetKey = e.target.value;
    loadDataset(state.datasetKey);
    canvasGrid.clearTrail();
  });

  // Simulator Sliders & Demos
  const simAy = document.getElementById('simAy');
  const simGz = document.getElementById('simGz');
  const simAyVal = document.getElementById('simAyVal');
  const simGzVal = document.getElementById('simGzVal');

  simAy.addEventListener('input', (e) => {
    state.source = 'sim';
    state.demoMode = null;
    sourceLabel.textContent = 'Source: Simulator';
    state.simAy = parseFloat(e.target.value);
    simAyVal.textContent = `${state.simAy.toFixed(1)} m/s²`;
  });

  simGz.addEventListener('input', (e) => {
    state.source = 'sim';
    state.demoMode = null;
    sourceLabel.textContent = 'Source: Simulator';
    state.simGz = parseFloat(e.target.value);
    simGzVal.textContent = `${state.simGz.toFixed(2)} rad/s`;
  });

  document.getElementById('btnResetSim').addEventListener('click', () => {
    state.simAy = 0;
    state.simGz = 0;
    state.demoMode = null;
    simAy.value = 0;
    simGz.value = 0;
    simAyVal.textContent = '0.0 m/s²';
    simGzVal.textContent = '0.0 rad/s';
  });

  document.getElementById('btnSimCircle').addEventListener('click', () => {
    state.source = 'sim';
    state.demoMode = 'circle';
    state.demoTime = 0;
    sourceLabel.textContent = 'Source: Sim (Circle)';
  });

  document.getElementById('btnSimFigure8').addEventListener('click', () => {
    state.source = 'sim';
    state.demoMode = 'fig8';
    state.demoTime = 0;
    sourceLabel.textContent = 'Source: Sim (Fig-8)';
  });

  // Live Smartphone Sensors Integration
  const btnEnablePhone = document.getElementById('btnEnableDeviceSensors');
  const permStatus = document.getElementById('permStatus');

  btnEnablePhone.addEventListener('click', async () => {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const response = await DeviceMotionEvent.requestPermission();
        if (response === 'granted') {
          bindMotionEvents();
          permStatus.textContent = 'Active (Streaming)';
          permStatus.style.color = 'var(--accent-green)';
          state.source = 'phone';
          sourceLabel.textContent = 'Source: Phone IMU';
        } else {
          permStatus.textContent = 'Permission Denied';
        }
      } catch (err) {
        permStatus.textContent = 'Error: ' + err.message;
      }
    } else {
      bindMotionEvents();
      permStatus.textContent = 'Active (Streaming)';
      permStatus.style.color = 'var(--accent-green)';
      state.source = 'phone';
      sourceLabel.textContent = 'Source: Phone IMU';
    }
  });

  function bindMotionEvents() {
    window.addEventListener('devicemotion', (event) => {
      const acc = event.accelerationIncludingGravity || { x: 0, y: 0, z: 9.81 };
      const rot = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };
      
      // Map Phone coordinate frame:
      // acc.x = East/Lateral, acc.y = Longitudinal, acc.z = Vertical
      // rot.alpha (deg/s) -> Yaw (rad/s), rot.beta -> Pitch, rot.gamma -> Roll
      const deg2rad = Math.PI / 180.0;
      state.currentSensor = [
        acc.x || 0.0,
        acc.y || 0.0,
        acc.z || 9.81,
        (rot.gamma || 0.0) * deg2rad, // Roll
        (rot.beta || 0.0) * deg2rad,  // Pitch
        (rot.alpha || 0.0) * deg2rad  // Yaw
      ];
    });
  }
});
