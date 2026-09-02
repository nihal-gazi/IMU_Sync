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
      const w = canvas.width / (window.devicePixelRatio || 1);
      const h = canvas.height / (window.devicePixelRatio || 1);
      if (!w || !h) return;

      ctx.clearRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i <= 3; i++) {
        const y = (h / 4) * i;
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
      }
      for (let i = 1; i <= 5; i++) {
        const x = (w / 6) * i;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
      }
      ctx.stroke();

      const history = dataRef.current;
      const count = history[0].length;
      if (count < 2) return;

      // Dynamic Auto-scaling
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let c = 0; c < channels.length; c++) {
        for (let i = 0; i < count; i++) {
          const v = history[c][i];
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
      }
      const margin = Math.max(Math.abs(maxVal - minVal) * 0.2, 1.0);
      const yMin = minVal - margin;
      const yMax = maxVal + margin;
      const range = yMax - yMin || 1;

      // Zero-line
      const zeroY = h - ((0 - yMin) / range) * h;
      if (zeroY >= 0 && zeroY <= h) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, zeroY);
        ctx.lineTo(w, zeroY);
        ctx.stroke();
      }

      // Draw waveforms
      const dx = w / (count - 1);
      for (let c = 0; c < channels.length; c++) {
        ctx.strokeStyle = channels[c].color;
        ctx.lineWidth = 1.8;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();

        for (let i = 0; i < count; i++) {
          const val = history[c][i];
          const x = i * dx;
          const y = h - ((val - yMin) / range) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    };

    const loop = () => {
      const dpr = window.devicePixelRatio || 1;

      // Resize if needed
      [accelCanvasRef.current, gyroCanvasRef.current].forEach(canvas => {
        if (!canvas) return;
        const rect = canvas.parentElement.getBoundingClientRect();
        if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
          canvas.width = rect.width * dpr;
          canvas.height = rect.height * dpr;
          canvas.getContext('2d').scale(dpr, dpr);
        }
      });

      renderScope(
        accelCanvasRef.current,
        accelDataRef,
        [{ color: '#ff4757' }, { color: '#2ed573' }, { color: '#1e90ff' }],
        -15, 15
      );

      renderScope(
        gyroCanvasRef.current,
        gyroDataRef,
        [{ color: '#ffa502' }, { color: '#00d2d3' }, { color: '#ff4757' }],
        -3, 3
      );

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [accelDataRef, gyroDataRef]);

  const [ax, ay, az, gx, gy, gz] = currentImu;

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
            <span className="legend-item"><span className="legend-color dot-ax"></span>Ax: <b>{(ax >= 0 ? '+' : '') + ax.toFixed(2)}</b></span>
            <span className="legend-item"><span className="legend-color dot-ay"></span>Ay: <b>{(ay >= 0 ? '+' : '') + ay.toFixed(2)}</b></span>
            <span className="legend-item"><span className="legend-color dot-az"></span>Az: <b>{(az >= 0 ? '+' : '') + az.toFixed(2)}</b></span>
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
            <span className="legend-item"><span className="legend-color dot-gx"></span>Gx: <b>{(gx >= 0 ? '+' : '') + gx.toFixed(3)}</b></span>
            <span className="legend-item"><span className="legend-color dot-gy"></span>Gy: <b>{(gy >= 0 ? '+' : '') + gy.toFixed(3)}</b></span>
            <span className="legend-item"><span className="legend-color dot-gz"></span>Gz: <b>{(gz >= 0 ? '+' : '') + gz.toFixed(3)}</b></span>
          </div>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={gyroCanvasRef} />
        </div>
      </div>
    </div>
  );
}
