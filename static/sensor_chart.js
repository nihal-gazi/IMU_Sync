/**
 * High Performance Real-Time 60 FPS Sensor Oscilloscope
 * Renders smooth continuous waveform graphs for Accelerometer and Gyroscope.
 */

class SensorOscilloscope {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
    this.maxSamples = options.maxSamples || 200;
    this.yMin = options.yMin || -15;
    this.yMax = options.yMax || 15;
    this.autoScale = options.autoScale !== undefined ? options.autoScale : true;
    
    // Series channels: Array of { name, color, data: Float32Array }
    this.channels = options.channels || [];
    this.history = this.channels.map(() => new Float32Array(this.maxSamples));
    this.writeIndex = 0;
    this.count = 0;
    
    this.initCanvasSize();
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

  /**
   * Pushes a new sample of readings for all channels
   * @param {Array<number>} values - [v0, v1, v2, ...]
   */
  pushSample(values) {
    for (let c = 0; c < this.channels.length; c++) {
      if (values[c] !== undefined) {
        this.history[c][this.writeIndex] = values[c];
      }
    }
    this.writeIndex = (this.writeIndex + 1) % this.maxSamples;
    if (this.count < this.maxSamples) this.count++;
  }

  /**
   * Renders single frame of the oscilloscope waveform
   */
  render() {
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;
    if (!w || !h) return;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Auto-scale compute
    if (this.autoScale && this.count > 10) {
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let c = 0; c < this.channels.length; c++) {
        for (let i = 0; i < this.count; i++) {
          const val = this.history[c][i];
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      }
      const margin = Math.max(Math.abs(maxVal - minVal) * 0.2, 1.0);
      this.yMin = minVal - margin;
      this.yMax = maxVal + margin;
    }

    // Grid Lines & Zero Axis
    this.drawGrid(ctx, w, h);

    // Waveform Series
    for (let c = 0; c < this.channels.length; c++) {
      this.drawSeries(ctx, c, w, h);
    }
  }

  drawGrid(ctx, w, h) {
    // Background Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;

    // Horizontal Lines (3 lines)
    ctx.beginPath();
    for (let i = 1; i <= 3; i++) {
      const y = (h / 4) * i;
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    // Vertical Lines (5 lines)
    for (let i = 1; i <= 5; i++) {
      const x = (w / 6) * i;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    ctx.stroke();

    // Zero-line
    const zeroY = this.valToY(0, h);
    if (zeroY >= 0 && zeroY <= h) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, zeroY);
      ctx.lineTo(w, zeroY);
      ctx.stroke();
    }
  }

  valToY(val, h) {
    const range = this.yMax - this.yMin || 1;
    return h - ((val - this.yMin) / range) * h;
  }

  drawSeries(ctx, channelIdx, w, h) {
    const channel = this.channels[channelIdx];
    const data = this.history[channelIdx];
    const count = this.count;
    if (count < 2) return;

    ctx.strokeStyle = channel.color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const startIndex = (this.writeIndex - count + this.maxSamples) % this.maxSamples;
    const dx = w / (this.maxSamples - 1);

    for (let i = 0; i < count; i++) {
      const bufIdx = (startIndex + i) % this.maxSamples;
      const val = data[bufIdx];
      const x = i * dx;
      const y = this.valToY(val, h);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
}

window.SensorOscilloscope = SensorOscilloscope;
