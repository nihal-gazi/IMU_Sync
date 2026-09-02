/**
 * Infinite Draggable Canvas Grid & Compass Tracker
 * Features:
 * - Infinite coordinate grid with dynamic LOD and coordinate numbers.
 * - Draggable/Pannable & Zoomable with smooth rendering.
 * - Center Point with Compass Arrow pointing as per ML model output vector.
 * - Glowing trajectory trail showing dead-reckoned path.
 */

class CanvasGrid {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
    // Viewport State
    this.zoom = 1.0;
    this.minZoom = 0.1;
    this.maxZoom = 10.0;
    this.panX = 0; // Screen offset
    this.panY = 0;
    
    // Agent / Compass Tracker State (World Coordinates in meters)
    this.posX = 0.0;
    this.posY = 0.0;
    this.headingDeg = 0.0;
    this.headingRad = 0.0;
    this.vectorMag = 0.0;
    this.speedKmh = 0.0;
    
    // Trajectory History: Array of {x, y, speed}
    this.trail = [];
    this.maxTrailLength = 5000;
    
    // Drag Interaction State
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.initialPanX = 0;
    this.initialPanY = 0;
    this.cameraLocked = true; // Auto-follow center compass
    
    this.initCanvasSize();
    this.initEventListeners();
    this.startRenderLoop();
  }

  initCanvasSize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayWidth = rect.width;
    this.displayHeight = rect.height;
    
    if (this.cameraLocked) {
      this.recenter();
    }
  }

  initEventListeners() {
    window.addEventListener('resize', () => this.initCanvasSize());

    // Mouse Dragging
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.cameraLocked = false;
      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;
      this.initialPanX = this.panX;
      this.initialPanY = this.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.dragStartX;
      const dy = e.clientY - this.dragStartY;
      this.panX = this.initialPanX + dx;
      this.panY = this.initialPanY + dy;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Touch Dragging
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.cameraLocked = false;
        this.dragStartX = e.touches[0].clientX;
        this.dragStartY = e.touches[0].clientY;
        this.initialPanX = this.panX;
        this.initialPanY = this.panY;
      }
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
      if (!this.isDragging || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - this.dragStartX;
      const dy = e.touches[0].clientY - this.dragStartY;
      this.panX = this.initialPanX + dx;
      this.panY = this.initialPanY + dy;
    }, { passive: true });

    window.addEventListener('touchend', () => {
      this.isDragging = false;
    });

    // Mouse Wheel Zooming
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      this.zoomAt(e.clientX, e.clientY, zoomFactor);
    }, { passive: false });
  }

  zoomAt(screenX, screenY, factor) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = screenX - rect.left;
    const mouseY = screenY - rect.top;

    const newZoom = Math.min(Math.max(this.zoom * factor, this.minZoom), this.maxZoom);
    if (newZoom === this.zoom) return;

    // Adjust pan to zoom into mouse cursor position
    this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
    this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
    this.zoom = newZoom;

    const zoomTag = document.getElementById('zoomLevel');
    if (zoomTag) zoomTag.textContent = `${this.zoom.toFixed(1)}x`;
  }

  recenter() {
    // Center viewport directly on tracked compass point
    const pixelsPerMeter = 20 * this.zoom;
    this.panX = this.displayWidth / 2 - this.posX * pixelsPerMeter;
    this.panY = this.displayHeight / 2 + this.posY * pixelsPerMeter; // Canvas Y is inverted
    this.cameraLocked = true;
  }

  setZoom(newZoom) {
    this.zoom = Math.min(Math.max(newZoom, this.minZoom), this.maxZoom);
    const zoomTag = document.getElementById('zoomLevel');
    if (zoomTag) zoomTag.textContent = `${this.zoom.toFixed(1)}x`;
    if (this.cameraLocked) this.recenter();
  }

  clearTrail() {
    this.trail = [];
    this.posX = 0;
    this.posY = 0;
    this.recenter();
  }

  /**
   * Updates tracked agent motion using predicted displacement vector (dx, dy)
   * and direction heading.
   */
  updateMotion(dx, dy, headingDeg, headingRad, speedKmh) {
    this.posX += dx;
    this.posY += dy;
    this.headingDeg = headingDeg;
    this.headingRad = headingRad;
    this.speedKmh = speedKmh;
    this.vectorMag = Math.sqrt(dx * dx + dy * dy);

    // Append to trail
    this.trail.push({ x: this.posX, y: this.posY, speed: speedKmh });
    if (this.trail.length > this.maxTrailLength) {
      this.trail.shift();
    }

    if (this.cameraLocked) {
      const pixelsPerMeter = 20 * this.zoom;
      this.panX = this.displayWidth / 2 - this.posX * pixelsPerMeter;
      this.panY = this.displayHeight / 2 + this.posY * pixelsPerMeter;
    }
  }

  /**
   * Converts world coordinates (meters) to screen canvas pixels
   */
  worldToScreen(wx, wy) {
    const ppm = 20 * this.zoom; // 20 pixels = 1 meter
    const sx = this.panX + wx * ppm;
    const sy = this.panY - wy * ppm; // Math North (+Y) is Canvas Up (-Y)
    return { sx, sy };
  }

  /**
   * Main Render Loop (60 FPS)
   */
  startRenderLoop() {
    const render = () => {
      this.draw();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  draw() {
    const ctx = this.ctx;
    const width = this.displayWidth;
    const height = this.displayHeight;

    // 1. Clear background (Deep Black Cyberpunk)
    ctx.fillStyle = '#060609';
    ctx.fillRect(0, 0, width, height);

    // 2. Draw Infinite Draggable Grid
    this.drawInfiniteGrid(ctx, width, height);

    // 3. Draw Historical Trajectory Trail
    this.drawTrail(ctx);

    // 4. Draw Center Compass Point with Directional Vector Arrow
    this.drawCompassEntity(ctx);
  }

  drawInfiniteGrid(ctx, width, height) {
    const ppm = 20 * this.zoom; // Pixels per meter
    
    // Choose grid step based on zoom level (LOD)
    let gridStepMeters = 10; // 10 meters default
    if (this.zoom > 3.0) gridStepMeters = 2;
    else if (this.zoom > 1.5) gridStepMeters = 5;
    else if (this.zoom < 0.4) gridStepMeters = 50;
    else if (this.zoom < 0.2) gridStepMeters = 100;

    const gridStepPx = gridStepMeters * ppm;

    // Calculate visible coordinate ranges
    const startX = Math.floor(-this.panX / gridStepPx) * gridStepMeters;
    const endX = Math.ceil((width - this.panX) / gridStepPx) * gridStepMeters;
    const startY = Math.floor((this.panY - height) / gridStepPx) * gridStepMeters;
    const endY = Math.ceil(this.panY / gridStepPx) * gridStepMeters;

    // Minor Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = startX; x <= endX; x += gridStepMeters) {
      const { sx } = this.worldToScreen(x, 0);
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, height);
    }
    for (let y = startY; y <= endY; y += gridStepMeters) {
      const { sy } = this.worldToScreen(0, y);
      ctx.moveTo(0, sy);
      ctx.lineTo(width, sy);
    }
    ctx.stroke();

    // Major Axes (X = 0 and Y = 0) with subtle glow
    const origin = this.worldToScreen(0, 0);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // Y-Axis (Vertical line at X=0)
    ctx.moveTo(origin.sx, 0);
    ctx.lineTo(origin.sx, height);
    // X-Axis (Horizontal line at Y=0)
    ctx.moveTo(0, origin.sy);
    ctx.lineTo(width, origin.sy);
    ctx.stroke();

    // Grid Numbers / Coordinates Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    for (let x = startX; x <= endX; x += gridStepMeters * 2) {
      if (x === 0) continue;
      const { sx } = this.worldToScreen(x, 0);
      if (sx >= 0 && sx <= width - 40) {
        ctx.fillText(`${x}m`, sx + 4, Math.min(Math.max(origin.sy + 4, 10), height - 20));
      }
    }

    for (let y = startY; y <= endY; y += gridStepMeters * 2) {
      if (y === 0) continue;
      const { sy } = this.worldToScreen(0, y);
      if (sy >= 0 && sy <= height - 20) {
        ctx.fillText(`${y}m`, Math.min(Math.max(origin.sx + 4, 10), width - 40), sy + 4);
      }
    }
  }

  drawTrail(ctx) {
    if (this.trail.length < 2) return;

    ctx.lineWidth = Math.max(1.5, 2.5 * this.zoom);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let i = 1; i < this.trail.length; i++) {
      const p0 = this.worldToScreen(this.trail[i - 1].x, this.trail[i - 1].y);
      const p1 = this.worldToScreen(this.trail[i].x, this.trail[i].y);
      const speed = this.trail[i].speed;

      // Color coding: Cyan (slow) -> Amber (medium) -> Magenta (fast)
      if (speed < 10) {
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
      } else if (speed < 40) {
        ctx.strokeStyle = 'rgba(255, 184, 0, 0.6)';
      } else {
        ctx.strokeStyle = 'rgba(255, 0, 127, 0.7)';
      }

      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(p1.sx, p1.sy);
      ctx.stroke();
    }
  }

  /**
   * Draws the center compass point and the directional arrow line
   * pointing as per the ML model's output vector.
   */
  drawCompassEntity(ctx) {
    const { sx, sy } = this.worldToScreen(this.posX, this.posY);

    // 1. Radar Pulse Ripple Effect
    const pulseTime = (Date.now() % 2000) / 2000;
    const pulseRadius = 14 + pulseTime * 24;
    ctx.strokeStyle = `rgba(0, 240, 255, ${0.4 * (1 - pulseTime)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(sx, sy, pulseRadius, 0, 2 * Math.PI);
    ctx.stroke();

    // 2. Outer Compass Ring
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, 14, 0, 2 * Math.PI);
    ctx.stroke();

    // 3. Center Solid Core Point
    ctx.fillStyle = '#00f0ff';
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(sx, sy, 4.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0; // Reset blur

    // 4. Directional Arrow Line (ML Model Vector Output)
    // In our coordinate system: 0 rad is North (-Y on screen), pi/2 is East (+X on screen)
    const angleRad = this.headingRad;
    const arrowLen = Math.min(Math.max(28 + this.vectorMag * 30, 26), 65);
    
    // Arrow tip endpoint (dx = sin(θ), dy = -cos(θ) for canvas)
    const tipX = sx + Math.sin(angleRad) * arrowLen;
    const tipY = sy - Math.cos(angleRad) * arrowLen;

    // Arrow Stem Line
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Arrow Head Triangular Wings
    const headLen = 8;
    const headAngle = Math.PI / 6; // 30 deg
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

    // 5. Mini Heading Tag next to Compass
    ctx.fillStyle = 'rgba(0, 240, 255, 0.9)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${this.headingDeg.toFixed(0)}°`, tipX + 8, tipY - 4);
  }
}

window.CanvasGrid = CanvasGrid;
