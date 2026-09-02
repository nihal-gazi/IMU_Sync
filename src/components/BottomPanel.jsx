import React from 'react';
import { Activity, Radio, Cpu, ChevronDown, Play, Pause, RotateCcw } from 'lucide-react';
import SensorOscilloscope from './SensorOscilloscope';
import MLVectorRadar from './MLVectorRadar';

export default function BottomPanel({
  activeTab,
  setActiveTab,
  modelMode,
  setModelMode,
  isCollapsed,
  setIsCollapsed,
  // Sensor Oscilloscope Props
  accelDataRef,
  gyroDataRef,
  currentImuRef,
  // ML Vector Radar Props
  motionState,
  hiddenStateRef,
  scalers,
  isONNXReady,
  // Control Panel Props
  isPlaying,
  setIsPlaying,
  replayIndex,
  setReplayIndex,
  totalFrames,
  speedMultiplier,
  setSpeedMultiplier,
  onRestartReplay,
  onEnableMobileSensors,
  mobileSensorStatus,
  simAy,
  setSimAy,
  simGz,
  setSimGz,
  onResetSim,
  onSimCircle,
  onSimFig8
}) {
  const formatTime = (frameIdx) => {
    const s = Math.floor(frameIdx * 0.1);
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m < 10 ? '0' : ''}${m}:${rem < 10 ? '0' : ''}${rem}`;
  };

  return (
    <footer className={`bottom-panel ${isCollapsed ? 'collapsed' : ''}`}>
      {/* Tab Navigation Header */}
      <div className="tab-bar">
        <div className="tab-nav-group">
          <button
            className={`tab-btn ${activeTab === 'sensors' ? 'active' : ''}`}
            onClick={() => { setActiveTab('sensors'); setIsCollapsed(false); }}
          >
            <Activity size={15} />
            Sensor Oscilloscope
          </button>
          <button
            className={`tab-btn ${activeTab === 'ml' ? 'active' : ''}`}
            onClick={() => { setActiveTab('ml'); setIsCollapsed(false); }}
          >
            <Radio size={15} />
            ML Vector & State
          </button>
          <button
            className={`tab-btn ${activeTab === 'controls' ? 'active' : ''}`}
            onClick={() => { setActiveTab('controls'); setIsCollapsed(false); }}
          >
            <Cpu size={15} />
            Data Stream & Simulator
          </button>
        </div>

        <div className="tab-actions">
          {/* Architecture Selector Pill */}
          <div className="model-pill-selector">
            <span className="pill-label">Architecture:</span>
            <button
              className={`pill-btn ${modelMode === 'rnn' ? 'active' : ''}`}
              onClick={() => setModelMode('rnn')}
            >
              RNN (Sequential)
            </button>
            <button
              className={`pill-btn ${modelMode === 'mlp' ? 'active' : ''}`}
              onClick={() => setModelMode('mlp')}
            >
              MLP (Stateless)
            </button>
          </div>

          {/* Panel Collapse Toggle */}
          <button
            className="panel-collapse-btn"
            onClick={() => setIsCollapsed(!isCollapsed)}
            title="Collapse / Expand Panel"
          >
            <ChevronDown size={16} />
          </button>
        </div>
      </div>

      {/* Tab Content Panes (Kept mounted with display: block/none to preserve canvas streams) */}
      {!isCollapsed && (
        <div className="tab-content-wrapper">
          <div style={{ display: activeTab === 'sensors' ? 'block' : 'none', height: '100%' }}>
            <SensorOscilloscope
              accelDataRef={accelDataRef}
              gyroDataRef={gyroDataRef}
              currentImuRef={currentImuRef}
            />
          </div>

          <div style={{ display: activeTab === 'ml' ? 'block' : 'none', height: '100%' }}>
            <MLVectorRadar
              motionState={motionState}
              hiddenStateRef={hiddenStateRef}
              scalers={scalers}
              isONNXReady={isONNXReady}
            />
          </div>

          <div style={{ display: activeTab === 'controls' ? 'block' : 'none', height: '100%' }}>
            <div className="control-grid">
              {/* Replay Controls */}
              <div className="control-card">
                <h4 className="control-title">IO-VNBD Dataset Driving Replay</h4>
                <p className="card-desc">Replay real 10Hz smartphone IMU drive from the benchmark dataset.</p>

                <div className="replay-scrubber-row">
                  <input
                    type="range"
                    min="0"
                    max={Math.max(0, totalFrames - 1)}
                    value={replayIndex}
                    onChange={(e) => setReplayIndex(parseInt(e.target.value, 10))}
                    className="custom-slider"
                  />
                  <span className="scrub-label">
                    {formatTime(replayIndex)} / {formatTime(totalFrames)}
                  </span>
                </div>

                <div className="playback-btn-group">
                  <button
                    className="btn btn-primary"
                    onClick={() => setIsPlaying(!isPlaying)}
                  >
                    {isPlaying ? <Pause size={13} /> : <Play size={13} />}
                    {isPlaying ? 'Pause' : 'Play'}
                  </button>
                  <button className="btn btn-outline" onClick={onRestartReplay}>
                    <RotateCcw size={13} />
                    Restart
                  </button>
                  <div className="speed-selector">
                    <span>Speed:</span>
                    {[1, 2, 5, 10].map((s) => (
                      <button
                        key={s}
                        className={`speed-btn ${speedMultiplier === s ? 'active' : ''}`}
                        onClick={() => setSpeedMultiplier(s)}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Mobile Phone Bridge */}
              <div className="control-card">
                <h4 className="control-title">Live Smartphone IMU Bridge</h4>
                <p className="card-desc">Stream live accelerometer & gyroscope readings from your phone browser.</p>
                <div className="phone-connect-actions">
                  <button className="btn btn-outline" onClick={onEnableMobileSensors}>
                    Enable Mobile Sensors
                  </button>
                  <div className="sensor-permission-status">{mobileSensorStatus}</div>
                </div>
              </div>

              {/* Manual Joystick Simulator */}
              <div className="control-card">
                <h4 className="control-title">Manual IMU Testing Joystick</h4>
                <div className="slider-row">
                  <label>Forward Accel (Ay):</label>
                  <input
                    type="range"
                    min="-10"
                    max="10"
                    step="0.1"
                    value={simAy}
                    onChange={(e) => setSimAy(parseFloat(e.target.value))}
                    className="custom-slider"
                  />
                  <span>{simAy.toFixed(1)} m/s²</span>
                </div>
                <div className="slider-row">
                  <label>Lateral Yaw (Gz):</label>
                  <input
                    type="range"
                    min="-3"
                    max="3"
                    step="0.05"
                    value={simGz}
                    onChange={(e) => setSimGz(parseFloat(e.target.value))}
                    className="custom-slider"
                  />
                  <span>{simGz.toFixed(2)} rad/s</span>
                </div>
                <div className="slider-actions">
                  <button className="btn btn-outline" onClick={onResetSim}>Zero Controls</button>
                  <button className="btn btn-outline" onClick={onSimCircle}>Circle Demo</button>
                  <button className="btn btn-outline" onClick={onSimFig8}>Fig-8 Demo</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </footer>
  );
}
