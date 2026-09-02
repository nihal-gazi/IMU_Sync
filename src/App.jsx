import React, { useState, useEffect, useRef, useCallback } from 'react';
import TopNav from './components/TopNav';
import InfiniteCanvas from './components/InfiniteCanvas';
import BottomPanel from './components/BottomPanel';
import { onnxInferenceService } from './services/onnxInference';

export default function App() {
  const [isONNXReady, setIsONNXReady] = useState(false);
  const [modelMode, setModelMode] = useState('rnn'); // 'rnn' | 'mlp'
  const [source, setSource] = useState('replay'); // 'replay' | 'phone' | 'sim'
  const [activeTab, setActiveTab] = useState('sensors');
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Replay State
  const [isPlaying, setIsPlaying] = useState(true);
  const [replayIndex, setReplayIndex] = useState(0);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [datasetFrames, setDatasetFrames] = useState([]);

  // Telemetry HUD State
  const [telemetry, setTelemetry] = useState({
    posX: 0,
    posY: 0,
    speedKmh: 0,
    headingDeg: 0,
    latencyMs: 0.4
  });

  // Simulator & Mobile State
  const [simAy, setSimAy] = useState(0);
  const [simGz, setSimGz] = useState(0);
  const [demoMode, setDemoMode] = useState(null);
  const [mobileSensorStatus, setMobileSensorStatus] = useState('Ready');
  const [currentImu, setCurrentImu] = useState([0, 0, 9.81, 0, 0, 0]);

  // High-frequency mutable refs for 60fps render loop
  const motionState = useRef({
    posX: 0,
    posY: 0,
    headingDeg: 0,
    headingRad: 0,
    speed: 0,
    speedKmh: 0,
    dx: 0,
    dy: 0,
    dt: 0.1
  });

  const trailRef = useRef([]);
  const hiddenStateRef = useRef(new Float32Array(32));
  const recenterRef = useRef(null);

  const accelDataRef = useRef([[], [], []]); // [ax, ay, az]
  const gyroDataRef = useRef([[], [], []]);  // [gx, gy, gz]

  const phoneImuRef = useRef([0, 0, 9.81, 0, 0, 0]);
  const lastTickTimeRef = useRef(performance.now());
  const replayIndexRef = useRef(0);
  const isPlayingRef = useRef(true);
  const speedMultRef = useRef(1);
  const sourceRef = useRef('replay');
  const demoTimeRef = useRef(0);

  useEffect(() => { replayIndexRef.current = replayIndex; }, [replayIndex]);
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { speedMultRef.current = speedMultiplier; }, [speedMultiplier]);
  useEffect(() => { sourceRef.current = source; }, [source]);

  // Initialize ONNX Runtime Web and Fetch Dataset
  useEffect(() => {
    async function setup() {
      console.log('[IMU-Sync] Initializing ONNX Runtime Web...');
      const ready = await onnxInferenceService.init('/models');
      setIsONNXReady(ready);

      try {
        const resp = await fetch('/data/sample_journey.json');
        if (resp.ok) {
          const data = await resp.json();
          setDatasetFrames(data);
          console.log(`[Replay] Loaded ${data.length} driving frames.`);
        }
      } catch (e) {
        console.warn('[Replay] Could not load sample journey JSON:', e);
      }
    }
    setup();
  }, []);

  // Sync Architecture Mode with ONNX Service
  useEffect(() => {
    onnxInferenceService.setMode(modelMode);
  }, [modelMode]);

  // Push sample to oscilloscope buffers
  const pushOscilloscopeSample = useCallback((imu) => {
    const max = 120;
    const [ax, ay, az, gx, gy, gz] = imu;

    const acc = accelDataRef.current;
    acc[0].push(ax); if (acc[0].length > max) acc[0].shift();
    acc[1].push(ay); if (acc[1].length > max) acc[1].shift();
    acc[2].push(az); if (acc[2].length > max) acc[2].shift();

    const gyr = gyroDataRef.current;
    gyr[0].push(gx); if (gyr[0].length > max) gyr[0].shift();
    gyr[1].push(gy); if (gyr[1].length > max) gyr[1].shift();
    gyr[2].push(gz); if (gyr[2].length > max) gyr[2].shift();
  }, []);

  // Main 60 FPS Telemetry & Inference Loop
  useEffect(() => {
    let animId;
    let hudCounter = 0;

    const tick = async () => {
      const now = performance.now();
      let dt = (now - lastTickTimeRef.current) / 1000.0;
      lastTickTimeRef.current = now;
      if (dt <= 0 || dt > 0.5) dt = 0.1;

      let imu = [0, 0, 9.81, 0, 0, 0];
      let stepDt = dt;

      // 1. Determine IMU Source
      if (sourceRef.current === 'replay' && datasetFrames.length > 0) {
        if (isPlayingRef.current) {
          replayIndexRef.current = (replayIndexRef.current + speedMultRef.current) % datasetFrames.length;
          setReplayIndex(replayIndexRef.current);
        }
        const row = datasetFrames[replayIndexRef.current];
        if (row) {
          imu = [row.ax, row.ay, row.az, row.gx, row.gy, row.gz];
          stepDt = row.dt || 0.1;
        }
      } else if (sourceRef.current === 'sim') {
        if (demoMode === 'circle') {
          demoTimeRef.current += dt;
          imu = [(Math.random() - 0.5) * 0.1, 2.0, 9.81, 0, 0, 0.45];
        } else if (demoMode === 'fig8') {
          demoTimeRef.current += dt;
          imu = [(Math.random() - 0.5) * 0.1, 2.2, 9.81, 0, 0, Math.sin(demoTimeRef.current * 0.8) * 0.6];
        } else {
          imu = [
            (Math.random() - 0.5) * 0.1,
            simAy + (Math.random() - 0.5) * 0.15,
            9.81,
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02,
            simGz
          ];
        }
        stepDt = dt;
      } else if (sourceRef.current === 'phone') {
        imu = phoneImuRef.current;
        stepDt = dt;
      }

      setCurrentImu(imu);
      pushOscilloscopeSample(imu);

      // 2. Run ONNX Inference
      const pred = await onnxInferenceService.predictStep(imu, stepDt);

      // 3. Update Motion State & Trajectory
      const motion = motionState.current;
      motion.posX += pred.dx;
      motion.posY += pred.dy;
      motion.headingDeg = pred.headingDeg;
      motion.headingRad = pred.headingRad;
      motion.speed = pred.speed;
      motion.speedKmh = pred.speedKmh;
      motion.dx = pred.dx;
      motion.dy = pred.dy;
      motion.dt = stepDt;

      hiddenStateRef.current = pred.hiddenState;

      // Append to Trail
      const trail = trailRef.current;
      trail.push({ x: motion.posX, y: motion.posY, speed: motion.speedKmh });
      if (trail.length > 3000) trail.shift();

      // 4. Update HUD Telemetry (Throttled for React State)
      hudCounter++;
      if (hudCounter % 4 === 0) {
        setTelemetry({
          posX: motion.posX,
          posY: motion.posY,
          speedKmh: motion.speedKmh,
          headingDeg: motion.headingDeg,
          latencyMs: pred.latencyMs
        });
      }

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [datasetFrames, demoMode, simAy, simGz, pushOscilloscopeSample]);

  // Actions
  const handleToggleSource = () => {
    if (source === 'replay') setSource('sim');
    else if (source === 'sim') setSource('replay');
    else setSource('replay');
  };

  const handleRecenter = () => {
    if (recenterRef.current) recenterRef.current();
  };

  const handleClearTrail = () => {
    trailRef.current = [];
    const motion = motionState.current;
    motion.posX = 0;
    motion.posY = 0;
    if (recenterRef.current) recenterRef.current();
  };

  const handleRestartReplay = () => {
    setReplayIndex(0);
    replayIndexRef.current = 0;
    handleClearTrail();
    onnxInferenceService.resetHiddenState();
  };

  const handleEnableMobileSensors = async () => {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const resp = await DeviceMotionEvent.requestPermission();
        if (resp === 'granted') {
          bindDeviceMotion();
          setMobileSensorStatus('Active (Streaming)');
          setSource('phone');
        } else {
          setMobileSensorStatus('Permission Denied');
        }
      } catch (err) {
        setMobileSensorStatus('Error: ' + err.message);
      }
    } else {
      bindDeviceMotion();
      setMobileSensorStatus('Active (Streaming)');
      setSource('phone');
    }
  };

  const bindDeviceMotion = () => {
    window.addEventListener('devicemotion', (event) => {
      const acc = event.accelerationIncludingGravity || { x: 0, y: 0, z: 9.81 };
      const rot = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };
      const deg2rad = Math.PI / 180.0;
      phoneImuRef.current = [
        acc.x || 0.0,
        acc.y || 0.0,
        acc.z || 9.81,
        (rot.gamma || 0.0) * deg2rad,
        (rot.beta || 0.0) * deg2rad,
        (rot.alpha || 0.0) * deg2rad
      ];
    });
  };

  return (
    <div className="app-container">
      {/* Top Telemetry Navigation */}
      <TopNav
        modelMode={modelMode}
        posX={telemetry.posX}
        posY={telemetry.posY}
        speedKmh={telemetry.speedKmh}
        headingDeg={telemetry.headingDeg}
        latencyMs={telemetry.latencyMs}
        isONNXReady={isONNXReady}
        source={source}
        onToggleSource={handleToggleSource}
        onRecenter={handleRecenter}
        onClearTrail={handleClearTrail}
      />

      {/* Infinite Draggable Black Canvas Grid */}
      <InfiniteCanvas
        motionState={motionState}
        trailRef={trailRef}
        onRecenterRef={recenterRef}
      />

      {/* Bottom Panel with Tabs */}
      <BottomPanel
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelMode={modelMode}
        setModelMode={setModelMode}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        accelDataRef={accelDataRef}
        gyroDataRef={gyroDataRef}
        currentImu={currentImu}
        motionState={motionState}
        hiddenStateRef={hiddenStateRef}
        isONNXReady={isONNXReady}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        replayIndex={replayIndex}
        setReplayIndex={setReplayIndex}
        totalFrames={datasetFrames.length || 1000}
        speedMultiplier={speedMultiplier}
        setSpeedMultiplier={setSpeedMultiplier}
        onRestartReplay={handleRestartReplay}
        onEnableMobileSensors={handleEnableMobileSensors}
        mobileSensorStatus={mobileSensorStatus}
        simAy={simAy}
        setSimAy={setSimAy}
        simGz={simGz}
        setSimGz={setSimGz}
        onResetSim={() => { setSimAy(0); setSimGz(0); setDemoMode(null); }}
        onSimCircle={() => { setSource('sim'); setDemoMode('circle'); }}
        onSimFig8={() => { setSource('sim'); setDemoMode('fig8'); }}
      />
    </div>
  );
}
