import React, { useRef, useEffect } from 'react';

export default function MLVectorRadar({
  motionState,
  hiddenStateRef,
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
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
      }

      const w = rect.width;
      const h = rect.height;
      if (!w || !h) return;

      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(cx, cy) - 10;

      // 1. Concentric Circles
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

      // 4. ML Vector Arrow (pointing at every millisecond)
      const motion = motionState.current;
      const angleRad = motion.headingRad;
      const maxSpeed = 30.0;
      const magRatio = Math.min(motion.speed / maxSpeed, 1.0);
      const arrowLen = Math.max(radius * 0.25, radius * magRatio);

      const tipX = cx + Math.sin(angleRad) * arrowLen;
      const tipY = cy - Math.cos(angleRad) * arrowLen;

      // Glowing Radial Sector
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

      // Center pivot
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fill();

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [motionState]);

  const motion = motionState.current;
  const hidden = hiddenStateRef.current || new Float32Array(32);
  let normSq = 0;
  for (let i = 0; i < hidden.length; i++) normSq += hidden[i] * hidden[i];
  const hiddenNorm = Math.sqrt(normSq);

  return (
    <div className="ml-view-grid">
      {/* Polar Vector Radar */}
      <div className="ml-card">
        <div className="radar-header">
          <span className="card-title">ML OUTPUT VECTOR // POINTING AT EVERY MS</span>
          <span className="badge">{isONNXReady ? 'ONNX LIVE' : 'INITIALIZING'}</span>
        </div>
        <div className="radar-body">
          <div className="radar-canvas-box">
            <canvas ref={radarCanvasRef} />
          </div>
          <div className="vector-readouts">
            <div className="readout-box">
              <span className="r-label">PREDICTED HEADING (θ)</span>
              <span className="r-val highlight-cyan">{motion.headingDeg.toFixed(2)}°</span>
              <span className="r-sub">{motion.headingRad.toFixed(3)} rad</span>
            </div>
            <div className="readout-box">
              <span className="r-label">VECTOR MAGNITUDE |v|</span>
              <span className="r-val highlight-amber">{motion.speed.toFixed(2)} m/s</span>
              <span className="r-sub">{motion.speedKmh.toFixed(1)} km/h</span>
            </div>
            <div className="readout-box">
              <span className="r-label">STEP DISPLACEMENT (dx, dy)</span>
              <span className="r-val">dx: {motion.dx.toFixed(3)}m, dy: {motion.dy.toFixed(3)}m</span>
              <span className="r-sub">dt = {(motion.dt * 1000).toFixed(0)}ms</span>
            </div>
          </div>
        </div>
      </div>

      {/* RNN Hidden State Representation */}
      <div className="ml-card">
        <div className="radar-header">
          <span className="card-title">RNN HIDDEN LAYER REPRESENTATION (h_t ∈ ℝ³²)</span>
          <span className="state-meta">||h_t|| = {hiddenNorm.toFixed(2)}</span>
        </div>
        <div className="hidden-state-body">
          <div className="neurons-bar-grid">
            {Array.from({ length: 32 }).map((_, i) => {
              const val = hidden[i] || 0;
              const heightPct = Math.min(Math.max(Math.abs(val) * 100, 5), 100);
              const bgColor = val >= 0
                ? `rgba(0, 240, 255, ${0.4 + Math.abs(val) * 0.6})`
                : `rgba(255, 0, 127, ${0.4 + Math.abs(val) * 0.6})`;
              return (
                <div key={i} className="neuron-bar-col">
                  <div
                    className="neuron-bar"
                    style={{ height: `${heightPct}%`, backgroundColor: bgColor }}
                  />
                </div>
              );
            })}
          </div>
          <div className="model-explanation">
            <p><strong>Recurrent Node Equation:</strong> <code>h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)</code></p>
            <p><strong>Vector Mapping:</strong> <code>[v_x, v_y, |v|, cos θ, sin θ] = W_out · h_t + b_out</code></p>
          </div>
        </div>
      </div>
    </div>
  );
}
