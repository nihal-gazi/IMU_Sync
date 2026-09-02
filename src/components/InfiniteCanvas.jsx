import React, { useRef, useEffect, useCallback } from 'react';
import { Plus, Minus, Maximize2 } from 'lucide-react';

export default function InfiniteCanvas({
  motionState,
  trailRef,
  onRecenterRef
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const viewState = useRef({
    zoom: 1.0,
    minZoom: 0.1,
    maxZoom: 8.0,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    initialPanX: 0,
    initialPanY: 0,
    cameraLocked: true,
    width: 0,
    height: 0
  });

  const recenter = useCallback(() => {
    const v = viewState.current;
    const { posX, posY } = motionState.current;
    const ppm = 20 * v.zoom;
    v.panX = v.width / 2 - posX * ppm;
    v.panY = v.height / 2 + posY * ppm;
    v.cameraLocked = true;
  }, [motionState]);

  useEffect(() => {
    if (onRecenterRef) onRecenterRef.current = recenter;
  }, [recenter, onRecenterRef]);

  const handleResize = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    viewState.current.width = rect.width;
    viewState.current.height = rect.height;

    if (viewState.current.cameraLocked) {
      recenter();
    }
  }, [recenter]);

  const zoomAt = (screenX, screenY, factor) => {
    const v = viewState.current;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = screenX - rect.left;
    const mouseY = screenY - rect.top;

    const newZoom = Math.min(Math.max(v.zoom * factor, v.minZoom), v.maxZoom);
    if (newZoom === v.zoom) return;

    v.panX = mouseX - (mouseX - v.panX) * (newZoom / v.zoom);
    v.panY = mouseY - (mouseY - v.panY) * (newZoom / v.zoom);
    v.zoom = newZoom;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    handleResize();
    window.addEventListener('resize', handleResize);

    const onMouseDown = (e) => {
      const v = viewState.current;
      v.isDragging = true;
      v.cameraLocked = false;
      v.dragStartX = e.clientX;
      v.dragStartY = e.clientY;
      v.initialPanX = v.panX;
      v.initialPanY = v.panY;
    };

    const onMouseMove = (e) => {
      const v = viewState.current;
      if (!v.isDragging) return;
      v.panX = v.initialPanX + (e.clientX - v.dragStartX);
      v.panY = v.initialPanY + (e.clientY - v.dragStartY);
    };

    const onMouseUp = () => {
      viewState.current.isDragging = false;
    };

    const onWheel = (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.85;
      zoomAt(e.clientX, e.clientY, factor);
    };

    const onTouchStart = (e) => {
      if (e.touches.length === 1) {
        const v = viewState.current;
        v.isDragging = true;
        v.cameraLocked = false;
        v.dragStartX = e.touches[0].clientX;
        v.dragStartY = e.touches[0].clientY;
        v.initialPanX = v.panX;
        v.initialPanY = v.panY;
      }
    };

    const onTouchMove = (e) => {
      const v = viewState.current;
      if (!v.isDragging || e.touches.length !== 1) return;
      v.panX = v.initialPanX + (e.touches[0].clientX - v.dragStartX);
      v.panY = v.initialPanY + (e.touches[0].clientY - v.dragStartY);
    };

    const onTouchEnd = () => {
      viewState.current.isDragging = false;
    };

    canvas.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchmove', onTouchMove, { passive: true });
    window.addEventListener('touchend', onTouchEnd);

    return () => {
      window.removeEventListener('resize', handleResize);
      canvas.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
    };
  }, [handleResize]);

  // Main 60 FPS Render Loop
  useEffect(() => {
    let animId;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const v = viewState.current;
      const motion = motionState.current;
      const trail = trailRef.current;

      const w = v.width;
      const h = v.height;
      if (!w || !h) return;

      if (v.cameraLocked) {
        const ppm = 20 * v.zoom;
        v.panX = w / 2 - motion.posX * ppm;
        v.panY = h / 2 + motion.posY * ppm;
      }

      // 1. Clear Background
      ctx.fillStyle = '#060609';
      ctx.fillRect(0, 0, w, h);

      const ppm = 20 * v.zoom;
      const worldToScreen = (wx, wy) => ({
        sx: v.panX + wx * ppm,
        sy: v.panY - wy * ppm
      });

      // 2. Draw Infinite Coordinate Grid
      let gridStepMeters = 10;
      if (v.zoom > 3.0) gridStepMeters = 2;
      else if (v.zoom > 1.5) gridStepMeters = 5;
      else if (v.zoom < 0.4) gridStepMeters = 50;
      else if (v.zoom < 0.2) gridStepMeters = 100;

      const gridStepPx = gridStepMeters * ppm;
      const startX = Math.floor(-v.panX / gridStepPx) * gridStepMeters;
      const endX = Math.ceil((w - v.panX) / gridStepPx) * gridStepMeters;
      const startY = Math.floor((v.panY - h) / gridStepPx) * gridStepMeters;
      const endY = Math.ceil(v.panY / gridStepPx) * gridStepMeters;

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = startX; x <= endX; x += gridStepMeters) {
        const { sx } = worldToScreen(x, 0);
        ctx.moveTo(sx, 0);
        ctx.lineTo(sx, h);
      }
      for (let y = startY; y <= endY; y += gridStepMeters) {
        const { sy } = worldToScreen(0, y);
        ctx.moveTo(0, sy);
        ctx.lineTo(w, sy);
      }
      ctx.stroke();

      // Major Origin Axes
      const origin = worldToScreen(0, 0);
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.22)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(origin.sx, 0);
      ctx.lineTo(origin.sx, h);
      ctx.moveTo(0, origin.sy);
      ctx.lineTo(w, origin.sy);
      ctx.stroke();

      // Axis Numbers
      ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';

      for (let x = startX; x <= endX; x += gridStepMeters * 2) {
        if (x === 0) continue;
        const { sx } = worldToScreen(x, 0);
        if (sx >= 0 && sx <= w - 35) {
          ctx.fillText(`${x}m`, sx + 4, Math.min(Math.max(origin.sy + 4, 10), h - 20));
        }
      }

      // 3. Draw Trajectory Path Trail
      if (trail.length >= 2) {
        ctx.lineWidth = Math.max(1.5, 2.5 * v.zoom);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        for (let i = 1; i < trail.length; i++) {
          const p0 = worldToScreen(trail[i - 1].x, trail[i - 1].y);
          const p1 = worldToScreen(trail[i].x, trail[i].y);
          const spd = trail[i].speed || 0;

          if (spd < 5) ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
          else if (spd < 20) ctx.strokeStyle = 'rgba(255, 184, 0, 0.6)';
          else ctx.strokeStyle = 'rgba(255, 0, 127, 0.7)';

          ctx.beginPath();
          ctx.moveTo(p0.sx, p0.sy);
          ctx.lineTo(p1.sx, p1.sy);
          ctx.stroke();
        }
      }

      // 4. Draw Center Particle & Velocity Vector Arrow
      const { sx, sy } = worldToScreen(motion.posX, motion.posY);

      // Radar Pulse
      const pulseTime = (Date.now() % 2000) / 2000;
      ctx.strokeStyle = `rgba(0, 240, 255, ${0.4 * (1 - pulseTime)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(sx, sy, 14 + pulseTime * 24, 0, 2 * Math.PI);
      ctx.stroke();

      // Outer Ring
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.65)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx, sy, 14, 0, 2 * Math.PI);
      ctx.stroke();

      // Center Core Point
      ctx.fillStyle = '#00f0ff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(sx, sy, 4.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Small Line with Arrow Pointing along (Vx, Vy) — default North (0, -1) when at rest
      const vx = motion.vx || 0;
      const vy = motion.vy || 0;
      const speed = Math.hypot(vx, vy);

      let dirX = 0;
      let dirY = -1; // Default North
      let arrowLen = 32;

      if (speed > 0.01) {
        dirX = vx / speed;
        dirY = -vy / speed; // Math +Vy is canvas -Y
        arrowLen = Math.min(Math.max(28 + speed * 6, 26), 65);
      }

      const tipX = sx + dirX * arrowLen;
      const tipY = sy + dirY * arrowLen;

      // Stem Line
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();

      // Arrowhead Wings
      const headLen = 8;
      const headAngle = Math.PI / 6;
      const stemAngle = Math.atan2(tipY - sy, tipX - sx);

      ctx.fillStyle = '#00f0ff';
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(
        tipX - headLen * Math.cos(stemAngle - headAngle),
        tipY - headLen * Math.sin(stemAngle - headAngle)
      );
      ctx.lineTo(
        tipX - headLen * 0.5 * Math.cos(stemAngle),
        tipY - headLen * 0.5 * Math.sin(stemAngle)
      );
      ctx.lineTo(
        tipX - headLen * Math.cos(stemAngle + headAngle),
        tipY - headLen * Math.sin(stemAngle + headAngle)
      );
      ctx.closePath();
      ctx.fill();

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [motionState, trailRef]);

  return (
    <main className="canvas-container" ref={containerRef}>
      <canvas ref={canvasRef} className="grid-canvas" />

      {/* Crosshair Overlay */}
      <div className="canvas-overlay-tl">
        <div className="coordinate-crosshair">
          <span className="coord-tag">ORIGIN [0, 0]</span>
          <span className="zoom-tag">{viewState.current.zoom.toFixed(1)}x</span>
        </div>
      </div>

      {/* Zoom Controls */}
      <div className="canvas-overlay-br">
        <div className="zoom-controls">
          <button className="zoom-btn" onClick={() => zoomAt(viewState.current.width / 2, viewState.current.height / 2, 1.25)} title="Zoom In">
            <Plus size={14} />
          </button>
          <button className="zoom-btn" onClick={() => zoomAt(viewState.current.width / 2, viewState.current.height / 2, 0.8)} title="Zoom Out">
            <Minus size={14} />
          </button>
          <button className="zoom-btn" onClick={() => { viewState.current.zoom = 1.0; recenter(); }} title="Reset Zoom">
            <Maximize2 size={12} />
          </button>
        </div>
      </div>
    </main>
  );
}
