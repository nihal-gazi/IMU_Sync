import React, { useRef, useEffect } from 'react';

export default function MLVectorRadar({
  motionState,
  hiddenStateRef,
  scalers,
  ekfMetrics,
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
          ctx.fillText('N (+Py)', cx, cy - radius - 2);
          ctx.textBaseline = 'top';
          ctx.fillText('S (-Py)', cx, cy + radius + 2);
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText('E (+Px)', cx + radius + 3, cy);
          ctx.textAlign = 'right';
          ctx.fillText('W (-Px)', cx - radius - 3, cy);

          // 4. Vector Arrow POINTING TOWARDS (p.x, p.y)
          const motion = motionState?.current || { posX: 0, posY: 0, vx: 0, vy: 0, speed: 0 };
          const px = motion.posX || 0;
          const py = motion.posY || 0;
          const posDist = Math.hypot(px, py);

          let dirX = 0;
          let dirY = -1; // Default North at origin
          let arrowLen = radius * 0.4;

          if (posDist > 0.001) {
            dirX = px / posDist;
            dirY = -py / posDist; // Math +Py is canvas -Y (North)
            const maxPos = 50.0;
            const magRatio = Math.min(posDist / maxPos, 1.0);
            arrowLen = Math.max(radius * 0.35, radius * magRatio);
          }

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
  const hidden = hiddenStateRef?.current || new Float32Array(32);
  const ekf = ekfMetrics || {
    lastPredDx: 0,
    lastPredDy: 0,
    lastDeltaPx: 0,
    lastDeltaPy: 0,
    kalmanGain: 0.67,
    correctionCount: 0,
    timeSinceSec: 0.1
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
          <span className="badge">{isONNXReady ? 'TLIO TRANSFORMER ACTIVE' : 'INITIALIZING'}</span>
        </div>
        <div className="radar-body">
          <div className="radar-canvas-box">
            <canvas ref={radarCanvasRef} />
          </div>
          <div className="vector-readouts">
            <div className="readout-box">
              <span className="r-label">PARTICLE POSITION (Px, Py)</span>
              <span className="r-val highlight-cyan">
                Px: {(motion.posX || 0).toFixed(2)} m, Py: {(motion.posY || 0).toFixed(2)} m
              </span>
              <span className="r-sub">Default North at origin (0, 0)</span>
            </div>
            <div className="readout-box">
              <span className="r-label">KINEMATIC VELOCITY (Vx, Vy)</span>
              <span className="r-val highlight-amber">
                Vx: {(motion.vx || 0).toFixed(2)}, Vy: {(motion.vy || 0).toFixed(2)} m/s
              </span>
              <span className="r-sub">Speed: {(motion.speedKmh || 0).toFixed(1)} km/h</span>
            </div>
            <div className="readout-box">
              <span className="r-label">TLIO SYSTEM MODE</span>
              <span className="r-val">High-Freq Kinematics + 1s Transformer EKF</span>
              <span className="r-sub">Tight-coupling drift elimination</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. 1-Second TLIO Transformer & EKF Correction Metrics */}
      <div className="ml-card">
        <div className="radar-header">
          <span className="card-title">1-SEC TRANSFORMER & EKF CORRECTION STATUS</span>
          <span className="state-meta">EKF Updates: #{ekf.correctionCount || 0}</span>
        </div>

        {/* EKF Telemetry Readout Grid */}
        <div className="ekf-telemetry-grid">
          <div className="readout-box">
            <span className="r-label">1s TRANSFORMER DISPLACEMENT (Δx, Δy)</span>
            <span className="r-val highlight-cyan">
              Δx: {(ekf.lastPredDx || 0).toFixed(2)}m, Δy: {(ekf.lastPredDy || 0).toFixed(2)}m
            </span>
            <span className="r-sub">Regressed from 10-sample IMU attention window</span>
          </div>
          <div className="readout-box">
            <span className="r-label">1s EKF CORRECTION DELTA (δPx, δPy)</span>
            <span className="r-val highlight-amber">
              δPx: {(ekf.lastDeltaPx >= 0 ? '+' : '') + (ekf.lastDeltaPx || 0).toFixed(3)}m, δPy: {(ekf.lastDeltaPy >= 0 ? '+' : '') + (ekf.lastDeltaPy || 0).toFixed(3)}m
            </span>
            <span className="r-sub">Kalman Gain K_pos: {(ekf.kalmanGain || 0.67).toFixed(2)}</span>
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
