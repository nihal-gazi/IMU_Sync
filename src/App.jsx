import React, { useState, useEffect, useRef, useCallback } from 'react';
import TopNav from './components/TopNav';
import InfiniteCanvas from './components/InfiniteCanvas';
import BottomPanel from './components/BottomPanel';
import { onnxInferenceService } from './services/onnxInference';
import { orientationAligner } from './services/orientationAligner';

export default function App() {
  const [isONNXReady, setIsONNXReady] = useState(false);
  const [modelMode, setModelMode] = useState('tlio');
  const [source, setSource] = useState('phone'); // 'phone' | 'replay' | 'sim'
  const [activeTab, setActiveTab] = useState('sensors');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isAlignEnabled, setIsAlignEnabled] = useState(true);

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
    aFwd: 0,
    latencyMs: 0.35,
    pitchDeg: 0,
    rollDeg: 0,
    headingDeg: 0,
    isMoving: false
  });

  // 1-Second Transformer & Alignment Metrics State
  const [transformerMetrics, setTransformerMetrics] = useState({
    lastPredDx: 0,
    lastPredDy: 0,
    aFwd: 0,
    stepCount: 0,
    lastUpdateSec: 0,
    pitchDeg: 0,
    rollDeg: 0,
    rawGyr: [0, 0, 0],
    alignedGyr: [0, 0, 0],
    isMoving: false
  });

  // Simulator & Mobile State
  const [simAy, setSimAy] = useState(0);
  const [simGz, setSimGz] = useState(0);
  const [demoMode, setDemoMode] = useState(null);
  const [mobileSensorStatus, setMobileSensorStatus] = useState('Active (Listening)');
  const [scalers, setScalers] = useState(onnxInferenceService.scalers);

  // Mutable motion state for canvas renderer
  const motionState = useRef({
    posX: 0,
    posY: 0,
    vx: 0,
    vy: 0,
    speed: 0,
    speedKmh: 0
  });

  const headingRadRef = useRef(0.0); // 0 = North (+Y)
  const curSpeedMpsRef = useRef(0.0); // Kinematic integrated speed in m/s
  const trailRef = useRef([{ x: 0, y: 0, speed: 0 }]);
  const hiddenStateRef = useRef(new Float32Array(32));
  const recenterRef = useRef(null);

  // Circular buffers for Oscilloscopes
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

  // 1-Second Sliding Window Buffer (10 samples @ 10Hz)
  const windowBufferRef = useRef([]);
  const timeSinceLast1sUpdateRef = useRef(0.0);
  const stepCountRef = useRef(0);

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
  useEffect(() => { orientationAligner.enabled = isAlignEnabled; }, [isAlignEnabled]);

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

  useEffect(() => {
    async function setup() {
      console.log('[IMU-Sync] Initializing 100Hz 2-Stage TLIO Transformer v0.1.10 (Exp 1)...');
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

  useEffect(() => {
    onnxInferenceService.setMode(modelMode);
  }, [modelMode]);

  // Main Telemetry, Gyro Heading Integration & 2-Stage Kinematic Loop
  useEffect(() => {
    let animId;

    const tick = async () => {
      const now = performance.now();
      let dt = (now - lastTickTimeRef.current) / 1000.0;
      lastTickTimeRef.current = now;
      if (dt <= 0 || dt > 0.5) dt = 0.1;

      let rawImu = [0, 0, 9.81, 0, 0, 0];
      const frames = datasetFramesRef.current;
      const curSource = sourceRef.current;

      // 1. Read Raw IMU
      if (curSource === 'phone') {
        rawImu = phoneImuRef.current;
      } else if (curSource === 'replay' && frames && frames.length > 0) {
        if (isPlayingRef.current) {
          replayIndexRef.current = (replayIndexRef.current + speedMultRef.current) % frames.length;
          setReplayIndex(replayIndexRef.current);
        }
        const row = frames[replayIndexRef.current];
        if (row) {
          rawImu = [row.ax, row.ay, row.az, row.gx, row.gy, row.gz];
        }
      } else if (curSource === 'sim') {
        const dMode = demoModeRef.current;
        if (dMode === 'circle') {
          demoTimeRef.current += dt;
          rawImu = [
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
          rawImu = [
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

          rawImu = [
            noiseAx,
            simAyRef.current + noiseAy,
            9.81 + noiseAz,
            noiseGx,
            noiseGy,
            simGzRef.current + noiseGz
          ];
        }
      }

      // 2. 3D Gravity Orientation Alignment (Screen-Up Frame)
      const alignResult = orientationAligner.alignIMU(rawImu);
      const imu = alignResult.alignedImu;
      currentImuRef.current = imu;

      // 3. Stream Oscilloscopes
      const [ax, ay, az, gx, gy, gz] = imu;
      const acc = accelDataRef.current;
      acc[0].push(ax); acc[0].shift();
      acc[1].push(ay); acc[1].shift();
      acc[2].push(az); acc[2].shift();

      const gyr = gyroDataRef.current;
      gyr[0].push(gx); gyr[0].shift();
      gyr[1].push(gy); gyr[1].shift();
      gyr[2].push(gz); gyr[2].shift();

      // 4. Continuous Gyroscope Heading Integration
      headingRadRef.current += gz * dt;

      // 5. Accumulate sliding 1-second continuous window
      const win = windowBufferRef.current;
      win.push(imu);
      if (win.length > 10) win.shift();

      timeSinceLast1sUpdateRef.current += dt;

      // 6. Every 1.0 Second: Run 2-Stage Kinematic Acceleration Pipeline
      if (timeSinceLast1sUpdateRef.current >= 1.0 && win.length >= 10) {
        timeSinceLast1sUpdateRef.current = 0.0;
        stepCountRef.current++;

        // Stage 1 Classifier & Stage 2 Acceleration Estimator
        const pred = await onnxInferenceService.predict1sAcceleration(win);
        const vPrev = curSpeedMpsRef.current;

        // Kinematic Speed Integration with Zero-Velocity (ZUPT) Gating
        if (pred.isMoving) {
          curSpeedMpsRef.current = Math.max(0.0, curSpeedMpsRef.current + pred.aFwd * 1.0);
        } else {
          curSpeedMpsRef.current = 0.0;
        }

        const curSpeedKmh = curSpeedMpsRef.current * 3.6;
        const fwdDisp = ((vPrev + curSpeedMpsRef.current) / 2.0) * 1.0;

        // Rotate Body Displacement into Global Coordinates using Integrated Heading
        const heading = headingRadRef.current;
        const dxWorld = fwdDisp * Math.sin(heading);
        const dyWorld = fwdDisp * Math.cos(heading);

        const motion = motionState.current;
        motion.posX += dxWorld;
        motion.posY += dyWorld;
        motion.vx = dxWorld;
        motion.vy = dyWorld;
        motion.speed = fwdDisp;
        motion.speedKmh = curSpeedKmh;

        // Append to trail
        const trail = trailRef.current;
        trail.push({ x: motion.posX, y: motion.posY, speed: motion.speedKmh });
        if (trail.length > 3000) trail.shift();

        // Update Telemetry HUD
        setTelemetry({
          posX: motion.posX,
          posY: motion.posY,
          vx: dxWorld,
          vy: dyWorld,
          speedKmh: curSpeedKmh,
          aFwd: pred.aFwd,
          latencyMs: pred.latencyMs,
          pitchDeg: alignResult.pitchDeg,
          rollDeg: alignResult.rollDeg,
          headingDeg: ((heading * 180.0) / Math.PI) % 360,
          isMoving: pred.isMoving
        });

        setTransformerMetrics({
          lastPredDx: dxWorld,
          lastPredDy: dyWorld,
          aFwd: pred.aFwd,
          fwdDispMeters: fwdDisp,
          stepCount: stepCountRef.current,
          lastUpdateSec: Math.round(performance.now() / 1000),
          pitchDeg: alignResult.pitchDeg,
          rollDeg: alignResult.rollDeg,
          rawGyr: alignResult.rawGyr || [0, 0, 0],
          alignedGyr: [gx, gy, gz],
          isMoving: pred.isMoving
        });
      }

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, []);

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
    trailRef.current = [{ x: 0, y: 0, speed: 0 }];
    const motion = motionState.current;
    motion.posX = 0;
    motion.posY = 0;
    motion.vx = 0;
    motion.vy = 0;
    motion.speed = 0;
    motion.speedKmh = 0;
    curSpeedMpsRef.current = 0.0;
    headingRadRef.current = 0.0;
    stepCountRef.current = 0;
    orientationAligner.reset();
    if (recenterRef.current) recenterRef.current();
  };

  const handleRestartReplay = () => {
    setReplayIndex(0);
    replayIndexRef.current = 0;
    handleClearTrail();
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
      {/* Top Telemetry Navigation with v0.1.8 */}
      <TopNav
        modelMode={modelMode}
        posX={telemetry.posX}
        posY={telemetry.posY}
        vx={telemetry.vx}
        vy={telemetry.vy}
        speedKmh={telemetry.speedKmh}
        aFwd={telemetry.aFwd}
        latencyMs={telemetry.latencyMs}
        pitchDeg={telemetry.pitchDeg}
        rollDeg={telemetry.rollDeg}
        headingDeg={telemetry.headingDeg}
        isMoving={telemetry.isMoving}
        isAlignEnabled={isAlignEnabled}
        onToggleAlign={() => setIsAlignEnabled(!isAlignEnabled)}
        isONNXReady={isONNXReady}
        source={source}
        onToggleSource={handleToggleSource}
        onRecenter={handleRecenter}
        onClearTrail={handleClearTrail}
      />

      {/* Infinite Canvas */}
      <InfiniteCanvas
        motionState={motionState}
        trailRef={trailRef}
        onRecenterRef={recenterRef}
      />

      {/* Bottom Panel */}
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
        ekfMetrics={transformerMetrics}
        isAlignEnabled={isAlignEnabled}
        onToggleAlign={() => setIsAlignEnabled(!isAlignEnabled)}
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
