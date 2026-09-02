import React from 'react';
import { Compass, Trash2 } from 'lucide-react';

export default function TopNav({
  modelMode,
  posX,
  posY,
  vx,
  vy,
  speedKmh,
  latencyMs,
  isONNXReady,
  source,
  onToggleSource,
  onRecenter,
  onClearTrail
}) {
  const getEngineName = () => {
    if (!isONNXReady) return 'Loading WASM...';
    if (modelMode === 'tlio') return 'TLIO Transformer (1s EKF)';
    if (modelMode === 'rnn') return 'ONNX SimpleRNN';
    return 'ONNX SimpleMLP';
  };

  return (
    <header className="top-nav">
      <div className="brand">
        <div className="logo-dot"></div>
        <div className="brand-text">
          <span className="brand-title">IMU-SYNC</span>
          <span className="brand-sub">v0.1.0 // TLIO TRANSFORMER + EKF</span>
        </div>
      </div>

      {/* Real-time Telemetry HUD */}
      <div className="hud-metrics">
        <div className="hud-item">
          <span className="hud-label">AI ENGINE</span>
          <span className="hud-val" style={{ color: isONNXReady ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
            {getEngineName()}
          </span>
        </div>
        <div className="hud-item">
          <span className="hud-label">POS (Px, Py)</span>
          <span className="hud-val highlight-cyan">{(posX || 0).toFixed(2)}m, {(posY || 0).toFixed(2)}m</span>
        </div>
        <div className="hud-item">
          <span className="hud-label">VELOCITY (Vx, Vy)</span>
          <span className="hud-val">{(vx || 0).toFixed(2)}, {(vy || 0).toFixed(2)} m/s</span>
        </div>
        <div className="hud-item">
          <span className="hud-label">SPEED</span>
          <span className="hud-val">{(speedKmh || 0).toFixed(1)} km/h</span>
        </div>
        <div className="hud-item">
          <span className="hud-label">LATENCY</span>
          <span className="hud-val hud-green">{(latencyMs || 0).toFixed(2)} ms</span>
        </div>
      </div>

      {/* Actions */}
      <div className="header-actions">
        <button className="btn btn-outline" onClick={onRecenter} title="Recenter Camera on Particle">
          <Compass size={14} />
          Recenter
        </button>
        <button className="btn btn-outline" onClick={onClearTrail} title="Clear Trajectory Trail">
          <Trash2 size={14} />
          Clear Path
        </button>
        <button className="btn btn-primary" onClick={onToggleSource}>
          <span className="status-indicator"></span>
          <span>Source: {source.toUpperCase()}</span>
        </button>
      </div>
    </header>
  );
}
