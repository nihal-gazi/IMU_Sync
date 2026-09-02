import React, { useRef, useEffect } from 'react';

export default function SensorOscilloscope({
  accelDataRef,
  gyroDataRef,
  currentImuRef
}) {
  const accelCanvasRef = useRef(null);
  const gyroCanvasRef = useRef(null);

  // Digital readout elements for 60fps DOM updates without triggering React re-renders
  const axValRef = useRef(null);
  const ayValRef = useRef(null);
  const azValRef = useRef(null);
  const gxValRef = useRef(null);
  const gyValRef = useRef(null);
  const gzValRef = useRef(null);

  useEffect(() => {
    let animId;

    const renderScope = (canvas, dataRef, channels, yMinDefault, yMaxDefault) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;

      // Get accurate dimensions
      const rect = canvas.getBoundingClientRect();
      let targetW = Math.floor(rect.width);
      let targetH = Math.floor(rect.height);

      // Fallback if container is temporarily 0
      if (targetW <= 10) targetW = canvas.parentElement?.clientWidth || 300;
      if (targetH <= 10) targetH = canvas.parentElement?.clientHeight || 150;
      if (targetW <= 10 || targetH <= 10) return;

      if (canvas.width !== targetW * dpr || canvas.height !== targetH * dpr) {
        canvas.width = targetW * dpr;
        canvas.height = targetH * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, targetW, targetH);

      // 1. Grid Background Lines
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

      const history = dataRef.current;
      const count = (history && history[0]) ? history[0].length : 0;

      if (count >= 2) {
        // Dynamic Range Compute
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

        const span = Math.max(Math.abs(maxVal - minVal), 1.0);
        const margin = span * 0.15;
        const yMin = minVal - margin;
        const yMax = maxVal + margin;
        const range = yMax - yMin || 1;

        // Zero reference line
        const zeroY = targetH - ((0 - yMin) / range) * targetH;
        if (zeroY >= 0 && zeroY <= targetH) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, zeroY);
          ctx.lineTo(targetW, zeroY);
          ctx.stroke();
        }

        // Draw Channels Waveforms
        const dx = targetW / (count - 1);
        for (let c = 0; c < channels.length; c++) {
          if (!history[c] || history[c].length === 0) continue;
          ctx.strokeStyle = channels[c].color;
          ctx.lineWidth = 2.2;
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          ctx.beginPath();

          for (let i = 0; i < count; i++) {
            const val = history[c][i] !== undefined ? history[c][i] : 0;
            const x = i * dx;
            const y = Math.max(0, Math.min(targetH, targetH - ((val - yMin) / range) * targetH));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      ctx.restore();
    };

    const loop = () => {
      // 1. Render Accel Oscilloscope
      renderScope(
        accelCanvasRef.current,
        accelDataRef,
        [
          { color: '#ff4757' }, // Ax (Red)
          { color: '#2ed573' }, // Ay (Green)
          { color: '#1e90ff' }  // Az (Blue)
        ],
        -10, 15
      );

      // 2. Render Gyro Oscilloscope
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

      // 3. Update digital readout badges directly at 60fps
      if (currentImuRef && currentImuRef.current) {
        const [ax, ay, az, gx, gy, gz] = currentImuRef.current;
        if (axValRef.current) axValRef.current.textContent = (ax >= 0 ? '+' : '') + ax.toFixed(2);
        if (ayValRef.current) ayValRef.current.textContent = (ay >= 0 ? '+' : '') + ay.toFixed(2);
        if (azValRef.current) azValRef.current.textContent = (az >= 0 ? '+' : '') + az.toFixed(2);
        if (gxValRef.current) gxValRef.current.textContent = (gx >= 0 ? '+' : '') + gx.toFixed(3);
        if (gyValRef.current) gyValRef.current.textContent = (gy >= 0 ? '+' : '') + gy.toFixed(3);
        if (gzValRef.current) gzValRef.current.textContent = (gz >= 0 ? '+' : '') + gz.toFixed(3);
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [accelDataRef, gyroDataRef, currentImuRef]);

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
            <span className="legend-item"><span className="legend-color dot-ax"></span>Ax: <b ref={axValRef}>+0.00</b></span>
            <span className="legend-item"><span className="legend-color dot-ay"></span>Ay: <b ref={ayValRef}>+0.00</b></span>
            <span className="legend-item"><span className="legend-color dot-az"></span>Az: <b ref={azValRef}>+9.81</b></span>
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
            <span className="legend-item"><span className="legend-color dot-gx"></span>Gx: <b ref={gxValRef}>+0.000</b></span>
            <span className="legend-item"><span className="legend-color dot-gy"></span>Gy: <b ref={gyValRef}>+0.000</b></span>
            <span className="legend-item"><span className="legend-color dot-gz"></span>Gz: <b ref={gzValRef}>+0.000</b></span>
          </div>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={gyroCanvasRef} />
        </div>
      </div>
    </div>
  );
}
