/**
 * ML Vector Polar Radar & RNN Hidden State Visualizer
 * Visualizes:
 * 1. Vector pointing at every millisecond (Polar Radar with magnitude & angle).
 * 2. RNN Hidden State Neurons Activation Bars (32 hidden neurons).
 */

class MLRadar {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
    this.headingDeg = 0;
    this.headingRad = 0;
    this.magnitude = 0;
    this.speedKmh = 0;
    
    this.initCanvasSize();
    this.initNeuronBars();
    window.addEventListener('resize', () => this.initCanvasSize());
  }

  initCanvasSize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.ctx.scale(dpr, dpr);
      this.displayWidth = rect.width;
      this.displayHeight = rect.height;
    }
  }

  initNeuronBars() {
    const container = document.getElementById('neuronsGrid');
    if (!container) return;
    container.innerHTML = '';
    
    this.neuronBars = [];
    for (let i = 0; i < 32; i++) {
      const col = document.createElement('div');
      col.className = 'neuron-bar-col';
      
      const bar = document.createElement('div');
      bar.className = 'neuron-bar';
      bar.style.height = '10%';
      
      col.appendChild(bar);
      container.appendChild(col);
      this.neuronBars.push(bar);
    }
  }

  /**
   * Updates radar with ML model prediction
   */
  update(predResult) {
    this.headingDeg = predResult.headingDeg;
    this.headingRad = predResult.headingRad;
    this.magnitude = predResult.speed;
    this.speedKmh = predResult.speedKmh;

    // Update Text DOM Elements
    const degEl = document.getElementById('mlHeadingDeg');
    if (degEl) degEl.textContent = `${this.headingDeg.toFixed(2)}°`;

    const radEl = document.getElementById('mlHeadingRad');
    if (radEl) radEl.textContent = `${this.headingRad.toFixed(3)} rad`;

    const magEl = document.getElementById('mlMagnitude');
    if (magEl) magEl.textContent = `${this.magnitude.toFixed(2)} m/s`;

    const kmhEl = document.getElementById('mlSpeedKmh');
    if (kmhEl) kmhEl.textContent = `${this.speedKmh.toFixed(1)} km/h`;

    const dispEl = document.getElementById('mlDisplacement');
    if (dispEl) dispEl.textContent = `dx: ${predResult.dx.toFixed(3)}m, dy: ${predResult.dy.toFixed(3)}m`;

    // Update RNN Hidden State Visualizer
    if (predResult.hiddenState && this.neuronBars) {
      const h = predResult.hiddenState;
      let normSq = 0;
      for (let i = 0; i < Math.min(h.length, this.neuronBars.length); i++) {
        const val = h[i]; // in range [-1, 1] due to tanh
        normSq += val * val;
        const heightPct = Math.min(Math.max(Math.abs(val) * 100, 5), 100);
        this.neuronBars[i].style.height = `${heightPct}%`;
        
        // Color: Cyan for positive, Magenta for negative
        if (val >= 0) {
          this.neuronBars[i].style.backgroundColor = `rgba(0, 240, 255, ${0.4 + Math.abs(val) * 0.6})`;
        } else {
          this.neuronBars[i].style.backgroundColor = `rgba(255, 0, 127, ${0.4 + Math.abs(val) * 0.6})`;
        }
      }
      
      const normEl = document.getElementById('hiddenStateNorm');
      if (normEl) normEl.textContent = `||h_t|| = ${Math.sqrt(normSq).toFixed(2)}`;
    }

    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;
    if (!w || !h) return;

    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 12;

    // 1. Concentric Range Circles
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let r of [0.33, 0.66, 1.0]) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * r, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // 2. Cardinal Crosshairs
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
    ctx.fillText('E (90°)', cx + radius + 4, cy);
    ctx.textAlign = 'right';
    ctx.fillText('W (270°)', cx - radius - 4, cy);

    // 4. ML Pointing Vector Arrow
    const angleRad = this.headingRad;
    const maxSpeed = 30.0; // 30 m/s (~108 km/h) max radius
    const magRatio = Math.min(this.magnitude / maxSpeed, 1.0);
    const arrowLen = Math.max(radius * 0.25, radius * magRatio);

    const tipX = cx + Math.sin(angleRad) * arrowLen;
    const tipY = cy - Math.cos(angleRad) * arrowLen;

    // Glowing Trail/Sector Arc
    const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, radius);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.3)');
    grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, arrowLen, 0, 2 * Math.PI);
    ctx.fill();

    // Arrow Line
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Arrow Head
    const headLen = 10;
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
}

window.MLRadar = MLRadar;
