import React, { useRef, useEffect } from 'react';

export default function MLVectorRadar({
  motionState,
  hiddenStateRef,
  scalers,
  ekfMetrics,
  isAlignEnabled,
  onToggleAlign,
  modelMode,
  isONNXReady
}) {
  const radarCanvasRef = useRef(null);

  useEffect(() => {
    let animId;

    const render = () => {
      const canvas = radarCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;

      const rect = canvas.parentElement.getBoundingClientRect();
      const targetW = Math.floor(rect.width);
      const targetH = Math.floor(rect.height);

      if (targetW > 0 && targetH > 0) {
        if (canvas.width !== targetW * dpr || canvas.height !== targetH * dpr) {
          canvas.width = targetW * dpr;
          canvas.height = targetH * dpr;
        }

        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, targetW, targetH);

        const cx = targetW / 2;
        const cy = targetH / 2;
        const radius = Math.min(cx, cy) - 12;

        if (radius > 10) {
          // 1. Concentric Range Circles
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
          ctx.lineWidth = 1;
          for (const r of [0.33, 0.66, 1.0]) {
            ctx.beginPath();
            ctx.arc(cx, cy, radius * r, 0, 2 * Math.PI);
            ctx.stroke();
          }

          // 2. Crosshairs
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
          ctx.beginPath();
          ctx.moveTo(cx, cy - radius);
          ctx.lineTo(cx, cy + radius);
          ctx.moveTo(cx - radius, cy);
          ctx.lineTo(cx + radius, cy);
          ctx.stroke();

          // 3. Cardinal Labels
          ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText('N (0°)', cx, cy - radius - 2);
          ctx.textBaseline = 'top';
          ctx.fillText('S (180°)', cx, cy + radius + 2);
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText('E (90°)', cx + radius + 3, cy);
          ctx.textAlign = 'right';
          ctx.fillText('W (270°)', cx - radius - 3, cy);

          // 4. Vector Arrow POINTING TOWARDS GYROSCOPE HEADING (θ)
          const motion = motionState?.current || { posX: 0, posY: 0, vx: 0, vy: 0, speed: 0, headingRad: 0 };
          const heading = (motion.headingRad !== undefined) ? motion.headingRad : 0.0;
          const dirX = Math.sin(heading);
          const dirY = -Math.cos(heading); // In canvas space, North is -Y, East is +X

          const speedKmh = motion.speedKmh || 0.0;
          const magRatio = Math.min(speedKmh / 60.0, 1.0);
          const arrowLen = Math.max(radius * 0.45, radius * (0.45 + 0.55 * magRatio));

          const tipX = cx + dirX * arrowLen;
          const tipY = cy + dirY * arrowLen;

          // Radial Glowing Sector
          const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, radius);
          grad.addColorStop(0, 'rgba(0, 240, 255, 0.25)');
          grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, arrowLen, 0, 2 * Math.PI);
          ctx.fill();

          // Vector Stem Line
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = 2.5;
          ctx.shadowColor = '#00f0ff';
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(tipX, tipY);
          ctx.stroke();
          ctx.shadowBlur = 0;

          // Vector Arrowhead
          const headLen = 9;
          const headAngle = Math.PI / 6;
          const stemAngle = Math.atan2(tipY - cy, tipX - cx);

          ctx.fillStyle = '#ffb800';
          ctx.beginPath();
          ctx.moveTo(tipX, tipY);
          ctx.lineTo(
            tipX - headLen * Math.cos(stemAngle - headAngle),
            tipY - headLen * Math.sin(stemAngle - headAngle)
          );
          ctx.lineTo(
            tipX - headLen * Math.cos(stemAngle + headAngle),
            tipY - headLen * Math.sin(stemAngle + headAngle)
          );
          ctx.closePath();
          ctx.fill();

          // Center Pivot Dot
          ctx.fillStyle = '#fff';
          ctx.beginPath();
          ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
          ctx.fill();
        }

        ctx.restore();
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [motionState]);

  const motion = motionState?.current || { posX: 0, posY: 0, vx: 0, vy: 0, speed: 0, speedKmh: 0 };
  const metrics = ekfMetrics || {
    lastPredDx: 0,
    lastPredDy: 0,
    stepCount: 0,
    lastUpdateSec: 0,
    pitchDeg: 0,
    rollDeg: 0,
    rawGyr: [0, 0, 0],
    alignedGyr: [0, 0, 0]
  };

  const featScalers = scalers?.features || {
    names: ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'],
    mean: [0.04, 0.06, 9.85, 0.002, -0.007, 0.002],
    std: [0.95, 0.89, 1.21, 0.124, 0.098, 0.089]
  };

  const targetScalers = scalers?.targets || {
    names: ['dx_1s', 'dy_1s'],
    mean: [0.0002, -0.215],
    std: [1.483, 1.852]
  };

  return (
    <div className="ml-view-grid">
      {/* 1. Vector Pointing Towards (p.x, p.y) & Position Metrics */}
      <div className="ml-card">
        <div className="radar-header">
          <span className="card-title">PARTICLE VECTOR // POINTING TOWARDS (Px, Py)</span>
          <span className="badge">{isONNXReady ? 'TRANSFORMER LIVE' : 'INITIALIZING'}</span>
        </div>
        <div className="radar-body">
          <div className="radar-canvas-box">
            <canvas ref={radarCanvasRef} />
          </div>
          <div className="vector-readouts">
            <div className="readout-box">
              <span className="r-label">INSTANTANEOUS VELOCITY VECTOR (Vx, Vy)</span>
              <span className="r-val highlight-cyan">
                Vx: {(motion.vx || 0).toFixed(2)} m/s, Vy: {(motion.vy || 0).toFixed(2)} m/s
              </span>
              <span className="r-sub">Speed: {(motion.speedKmh || 0).toFixed(1)} km/h ({((motion.speed || 0)).toFixed(2)} m/s)</span>
            </div>
            <div className="readout-box">
              <span className="r-label">INTEGRATED POSITION (Px, Py)</span>
              <span className="r-val highlight-amber">
                Px: {(motion.posX || 0).toFixed(2)}m, Py: {(motion.posY || 0).toFixed(2)}m
              </span>
              <span className="r-sub">Distance from origin: {Math.hypot(motion.posX || 0, motion.posY || 0).toFixed(2)}m</span>
            </div>
            <div className="readout-box">
              <span className="r-label">3D TILT ORIENTATION</span>
              <span className="r-val" style={{ color: 'var(--accent-green)' }}>
                Pitch: {(metrics.pitchDeg >= 0 ? '+' : '') + (metrics.pitchDeg || 0).toFixed(1)}°, Roll: {(metrics.rollDeg >= 0 ? '+' : '') + (metrics.rollDeg || 0).toFixed(1)}°
              </span>
              <span className="r-sub">Rodrigues 3D Rotation Active: {isAlignEnabled ? 'YES' : 'NO'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. 3D Gravity Alignment & Normalization Factors */}
      <div className="ml-card">
        <div className="radar-header">
          <span className="card-title">3D GRAVITY ALIGNMENT & NORMALIZATION</span>
          <span className="state-meta">1s Steps: #{metrics.stepCount || 0}</span>
        </div>

        {/* 3D Alignment Readout Grid */}
        <div className="ekf-telemetry-grid">
          <div className="readout-box">
            <span className="r-label">RAW GYROSCOPE [Gx, Gy, Gz]</span>
            <span className="r-val highlight-cyan" style={{ fontSize: '11px' }}>
              {(metrics.rawGyr[0] || 0).toFixed(2)}, {(metrics.rawGyr[1] || 0).toFixed(2)}, {(metrics.rawGyr[2] || 0).toFixed(2)} rad/s
            </span>
            <span className="r-sub">Raw phone body frame</span>
          </div>
          <div className="readout-box">
            <span className="r-label">ALIGNED GYROSCOPE [Gx, Gy, Gz]</span>
            <span className="r-val highlight-amber" style={{ fontSize: '11px' }}>
              {(metrics.alignedGyr[0] || 0).toFixed(2)}, {(metrics.alignedGyr[1] || 0).toFixed(2)}, {(metrics.alignedGyr[2] || 0).toFixed(2)} rad/s
            </span>
            <span className="r-sub">Rotated into dataset dashboard frame</span>
          </div>
        </div>

        {/* Normalization Factors Table */}
        <div className="scalers-table-box" style={{ marginTop: '6px' }}>
          <table className="scalers-table">
            <thead>
              <tr>
                <th>Field</th>
                <th>Mean (μ)</th>
                <th>Std (σ)</th>
              </tr>
            </thead>
            <tbody>
              {featScalers.names.map((name, idx) => (
                <tr key={name}>
                  <td><strong>{name}</strong></td>
                  <td>{(featScalers.mean[idx] || 0).toFixed(4)}</td>
                  <td>{(featScalers.std[idx] || 1).toFixed(4)}</td>
                </tr>
              ))}
              {targetScalers.names.map((name, idx) => (
                <tr key={name} className="target-scaler-row">
                  <td><strong style={{ color: 'var(--accent-cyan)' }}>{name} (1s Target)</strong></td>
                  <td>{(targetScalers.mean[idx] || 0).toFixed(4)}</td>
                  <td>{(targetScalers.std[idx] || 1).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
