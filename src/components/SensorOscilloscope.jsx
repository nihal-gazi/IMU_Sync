import React, { useRef, useEffect } from 'react';

export default function SensorOscilloscope({
  accelDataRef,
  gyroDataRef,
  currentImu
}) {
  const accelCanvasRef = useRef(null);
  const gyroCanvasRef = useRef(null);

  useEffect(() => {
    let animId;

    const renderScope = (canvas, dataRef, channels, yMinDefault, yMaxDefault) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.parentElement.getBoundingClientRect();

      const targetW = Math.floor(rect.width);
      const targetH = Math.floor(rect.height);
      if (targetW <= 0 || targetH <= 0) return;

      if (canvas.width !== targetW * dpr || canvas.height !== targetH * dpr) {
        canvas.width = targetW * dpr;
        canvas.height = targetH * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, targetW, targetH);

      // Grid Lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i <= 3; i++) {
        const y = (targetH / 4) * i;
        ctx.moveTo(0, y);
        ctx.lineTo(targetW, y);
      }
      for (let i = 1; i <= 5; i++) {
        const x = (targetW / 6) * i;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, targetH);
      }
      ctx.stroke();

      const history = dataRef.current || [[], [], []];
      const count = history[0] ? history[0].length : 0;

      if (count >= 2) {
        // Compute min/max for auto-scaling
        let minVal = yMinDefault;
        let maxVal = yMaxDefault;
        for (let c = 0; c < channels.length; c++) {
          if (!history[c]) continue;
          for (let i = 0; i < count; i++) {
            const v = history[c][i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
          }
        }
        const margin = Math.max(Math.abs(maxVal - minVal) * 0.15, 0.5);
        const yMin = minVal - margin;
        const yMax = maxVal + margin;
        const range = yMax - yMin || 1;

        // Zero-line
        const zeroY = targetH - ((0 - yMin) / range) * targetH;
        if (zeroY >= 0 && zeroY <= targetH) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, zeroY);
          ctx.lineTo(targetW, zeroY);
          ctx.stroke();
        }

        // Draw waveform channels
        const dx = targetW / (count - 1);
        for (let c = 0; c < channels.length; c++) {
          if (!history[c] || history[c].length === 0) continue;
          ctx.strokeStyle = channels[c].color;
          ctx.lineWidth = 2.0;
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          ctx.beginPath();

          for (let i = 0; i < count; i++) {
            const val = history[c][i] !== undefined ? history[c][i] : 0;
            const x = i * dx;
            const y = targetH - ((val - yMin) / range) * targetH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      ctx.restore();
    };

    const loop = () => {
      renderScope(
        accelCanvasRef.current,
        accelDataRef,
        [
          { color: '#ff4757' }, // Ax (Red)
          { color: '#2ed573' }, // Ay (Green)
          { color: '#1e90ff' }  // Az (Blue)
        ],
        -12, 12
      );

      renderScope(
        gyroCanvasRef.current,
        gyroDataRef,
        [
          { color: '#ffa502' }, // Gx (Orange)
          { color: '#00d2d3' }, // Gy (Cyan)
          { color: '#ff4757' }  // Gz (Magenta)
        ],
        -2.5, 2.5
      );

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [accelDataRef, gyroDataRef]);

  const imu = currentImu || [0, 0, 9.81, 0, 0, 0];
  const [ax, ay, az, gx, gy, gz] = imu;

  return (
    <div className="sensor-graphs-grid">
      {/* Accelerometer Oscilloscope */}
      <div className="graph-card">
        <div className="graph-header">
          <div className="graph-title-group">
            <span className="graph-title">ACCELEROMETER</span>
            <span className="graph-unit">[m/s²]</span>
          </div>
          <div className="graph-legend">
            <span className="legend-item"><span className="legend-color dot-ax"></span>Ax: <b>{(ax >= 0 ? '+' : '') + (ax || 0).toFixed(2)}</b></span>
            <span className="legend-item"><span className="legend-color dot-ay"></span>Ay: <b>{(ay >= 0 ? '+' : '') + (ay || 0).toFixed(2)}</b></span>
            <span className="legend-item"><span className="legend-color dot-az"></span>Az: <b>{(az >= 0 ? '+' : '') + (az || 0).toFixed(2)}</b></span>
          </div>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={accelCanvasRef} />
        </div>
      </div>

      {/* Gyroscope Oscilloscope */}
      <div className="graph-card">
        <div className="graph-header">
          <div className="graph-title-group">
            <span className="graph-title">GYROSCOPE</span>
            <span className="graph-unit">[rad/s]</span>
          </div>
          <div className="graph-legend">
            <span className="legend-item"><span className="legend-color dot-gx"></span>Gx: <b>{(gx >= 0 ? '+' : '') + (gx || 0).toFixed(3)}</b></span>
            <span className="legend-item"><span className="legend-color dot-gy"></span>Gy: <b>{(gy >= 0 ? '+' : '') + (gy || 0).toFixed(3)}</b></span>
            <span className="legend-item"><span className="legend-color dot-gz"></span>Gz: <b>{(gz >= 0 ? '+' : '') + (gz || 0).toFixed(3)}</b></span>
          </div>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={gyroCanvasRef} />
        </div>
      </div>
    </div>
  );
}
