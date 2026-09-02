import React, { useState, useEffect, useRef, useCallback } from 'react';
import TopNav from './components/TopNav';
import InfiniteCanvas from './components/InfiniteCanvas';
import BottomPanel from './components/BottomPanel';
import { onnxInferenceService } from './services/onnxInference';

export default function App() {
  const [isONNXReady, setIsONNXReady] = useState(false);
  const [modelMode, setModelMode] = useState('rnn'); // 'rnn' | 'mlp'
  // Default to PHONE sensor as requested
  const [source, setSource] = useState('phone'); // 'phone' | 'replay' | 'sim'
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
    vx: 0,
    vy: 0,
    speedKmh: 0,
    latencyMs: 0.4
  });

  // Simulator & Mobile State
  const [simAy, setSimAy] = useState(0);
  const [simGz, setSimGz] = useState(0);
  const [demoMode, setDemoMode] = useState(null);
  const [mobileSensorStatus, setMobileSensorStatus] = useState('Active (Listening)');
  const [scalers, setScalers] = useState(onnxInferenceService.scalers);

  // Mutable refs for 60fps telemetry & inference loop
  const motionState = useRef({
    posX: 0,
    posY: 0,
    vx: 0,
    vy: 0,
    speed: 0,
    speedKmh: 0
  });

  const trailRef = useRef([]);
  const hiddenStateRef = useRef(new Float32Array(32));
  const recenterRef = useRef(null);

  // 120-sample circular buffers for Oscilloscopes
  const BUFFER_LEN = 120;
  const accelDataRef = useRef([
    new Array(BUFFER_LEN).fill(0),
    new Array(BUFFER_LEN).fill(0),
    new Array(BUFFER_LEN).fill(9.81)
  ]);
  const gyroDataRef = useRef([
    new Array(BUFFER_LEN).fill(0),
    new Array(BUFFER_LEN).fill(0),
    new Array(BUFFER_LEN).fill(0)
  ]);

  const currentImuRef = useRef([0, 0, 9.81, 0, 0, 0]);
  const phoneImuRef = useRef([0, 0, 9.81, 0, 0, 0]);
  const lastTickTimeRef = useRef(performance.now());
  const replayIndexRef = useRef(0);
  const isPlayingRef = useRef(true);
  const speedMultRef = useRef(1);
  const sourceRef = useRef('phone');
  const demoTimeRef = useRef(0);
  const simAyRef = useRef(0);
  const simGzRef = useRef(0);
  const demoModeRef = useRef(null);
  const datasetFramesRef = useRef([]);

  useEffect(() => { replayIndexRef.current = replayIndex; }, [replayIndex]);
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { speedMultRef.current = speedMultiplier; }, [speedMultiplier]);
  useEffect(() => { sourceRef.current = source; }, [source]);
  useEffect(() => { simAyRef.current = simAy; }, [simAy]);
  useEffect(() => { simGzRef.current = simGz; }, [simGz]);
  useEffect(() => { demoModeRef.current = demoMode; }, [demoMode]);
  useEffect(() => { datasetFramesRef.current = datasetFrames; }, [datasetFrames]);

  // Bind phone sensors immediately on startup
  const bindDeviceMotion = useCallback(() => {
    const handleMotion = (event) => {
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
    };

    window.addEventListener('devicemotion', handleMotion);
    return () => window.removeEventListener('devicemotion', handleMotion);
  }, []);

  // Initialize ONNX Runtime Web, Datasets & Phone Sensors
  useEffect(() => {
    async function setup() {
      console.log('[IMU-Sync] Initializing ONNX Runtime Web v0.0.1...');
      const ready = await onnxInferenceService.init('/models');
      setIsONNXReady(ready);
      setScalers(onnxInferenceService.scalers);

      try {
        const resp = await fetch('/data/sample_journey.json');
        if (resp.ok) {
          const data = await resp.json();
          setDatasetFrames(data);
          datasetFramesRef.current = data;
        }
      } catch (e) {
        console.warn('[Replay] Could not load sample journey JSON:', e);
      }
    }
    setup();
    const unbind = bindDeviceMotion();
    return () => unbind();
  }, [bindDeviceMotion]);

  // Sync Architecture Mode with ONNX Service
  useEffect(() => {
    onnxInferenceService.setMode(modelMode);
  }, [modelMode]);

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
      const frames = datasetFramesRef.current;
      const curSource = sourceRef.current;

      // 1. Source: Live Phone Sensors (Default)
      if (curSource === 'phone') {
        imu = phoneImuRef.current;
      }
      // 2. Source: Dataset Replay
      else if (curSource === 'replay' && frames && frames.length > 0) {
        if (isPlayingRef.current) {
          replayIndexRef.current = (replayIndexRef.current + speedMultRef.current) % frames.length;
          setReplayIndex(replayIndexRef.current);
        }
        const row = frames[replayIndexRef.current];
        if (row) {
          imu = [row.ax, row.ay, row.az, row.gx, row.gy, row.gz];
        }
      } 
      // 3. Source: Simulator & Manual Joystick
      else if (curSource === 'sim') {
        const dMode = demoModeRef.current;
        if (dMode === 'circle') {
          demoTimeRef.current += dt;
          imu = [
            (Math.random() - 0.5) * 0.15,
            2.5 + (Math.random() - 0.5) * 0.2,
            9.81 + (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02,
            0.55 + (Math.random() - 0.5) * 0.03
          ];
        } else if (dMode === 'fig8') {
          demoTimeRef.current += dt;
          const turnRate = Math.sin(demoTimeRef.current * 0.8) * 0.7;
          imu = [
            (Math.random() - 0.5) * 0.15,
            2.2 + (Math.random() - 0.5) * 0.2,
            9.81 + (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02,
            turnRate + (Math.random() - 0.5) * 0.03
          ];
        } else {
          const noiseAx = (Math.random() - 0.5) * 0.08;
          const noiseAy = (Math.random() - 0.5) * 0.12;
          const noiseAz = (Math.random() - 0.5) * 0.1;
          const noiseGx = (Math.random() - 0.5) * 0.01;
          const noiseGy = (Math.random() - 0.5) * 0.01;
          const noiseGz = (Math.random() - 0.5) * 0.015;

          imu = [
            noiseAx,
            simAyRef.current + noiseAy,
            9.81 + noiseAz,
            noiseGx,
            noiseGy,
            simGzRef.current + noiseGz
          ];
        }
      }

      currentImuRef.current = imu;

      // 2. Push sample to Oscilloscope circular buffers
      const [ax, ay, az, gx, gy, gz] = imu;
      const acc = accelDataRef.current;
      acc[0].push(ax); acc[0].shift();
      acc[1].push(ay); acc[1].shift();
      acc[2].push(az); acc[2].shift();

      const gyr = gyroDataRef.current;
      gyr[0].push(gx); gyr[0].shift();
      gyr[1].push(gy); gyr[1].shift();
      gyr[2].push(gz); gyr[2].shift();

      // 3. Run ONNX Inference: Predicts [vx, vy]
      const pred = await onnxInferenceService.predictStep(imu);

      // 4. Update Particle Position: px = px + vx, py = py + vy (Direct Velocity Accumulation)
      const motion = motionState.current;
      motion.vx = pred.vx;
      motion.vy = pred.vy;
      motion.speed = pred.speed;
      motion.speedKmh = pred.speedKmh;

      // Direct integration
      motion.posX += pred.vx;
      motion.posY += pred.vy;

      hiddenStateRef.current = pred.hiddenState;

      // 5. Append to Trail
      const trail = trailRef.current;
      trail.push({ x: motion.posX, y: motion.posY, speed: motion.speedKmh });
      if (trail.length > 3000) trail.shift();

      // 6. Update HUD Telemetry
      hudCounter++;
      if (hudCounter % 4 === 0) {
        setTelemetry({
          posX: motion.posX,
          posY: motion.posY,
          vx: pred.vx,
          vy: pred.vy,
          speedKmh: motion.speedKmh,
          latencyMs: pred.latencyMs
        });
      }

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Actions
  const handleToggleSource = () => {
    if (source === 'phone') setSource('replay');
    else if (source === 'replay') {
      setSource('sim');
      setDemoMode(null);
    } else setSource('phone');
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

  return (
    <div className="app-container">
      {/* Top Telemetry Navigation with v0.0.1 */}
      <TopNav
        modelMode={modelMode}
        posX={telemetry.posX}
        posY={telemetry.posY}
        vx={telemetry.vx}
        vy={telemetry.vy}
        speedKmh={telemetry.speedKmh}
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
        currentImuRef={currentImuRef}
        motionState={motionState}
        hiddenStateRef={hiddenStateRef}
        scalers={scalers}
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
