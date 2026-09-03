// Verification test for StepDetector logic in Node.js
class StepDetector {
  constructor() {
    this.strideLength = 0.65;
    this.stepCount = 0;
    this.lastStepTime = 0;
    this.minStepIntervalMs = 260;
    this.lastStepIntervalMs = 0;
    this.maWindowSize = 5;
    this.zBuffer = [];
    this.windowSize = 15;
    this.smoothedZHistory = [];
    this.dynamicThreshold = 1.5;
    this.pocketZuptWindowSize = 30;
    this.accelNormHistory = [];
    this.quietDurationMs = 0;
    this.lastSampleTime = 0;
    this.isPocketZupt = false;
    this.frozenHeading = null;
  }

  processSample(ax, ay, az, currentHeadingDeg, timestamp) {
    const dt = this.lastSampleTime > 0 ? Math.min(100, timestamp - this.lastSampleTime) : 16.6;
    this.lastSampleTime = timestamp;

    const accelNorm = Math.sqrt(ax * ax + ay * ay + az * az);
    this.accelNormHistory.push(accelNorm);
    if (this.accelNormHistory.length > this.pocketZuptWindowSize) {
      this.accelNormHistory.shift();
    }

    let variance = 0;
    if (this.accelNormHistory.length >= 10) {
      const n = this.accelNormHistory.length;
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = this.accelNormHistory[i];
        sum += v;
        sumSq += v * v;
      }
      const mean = sum / n;
      variance = Math.max(0, sumSq / n - mean * mean);
    }

    if (variance < 0.20 && this.accelNormHistory.length >= 15) {
      this.quietDurationMs += dt;
      if (this.quietDurationMs >= 500) {
        if (!this.isPocketZupt) {
          this.isPocketZupt = true;
          this.frozenHeading = currentHeadingDeg;
        }
      }
    } else {
      this.quietDurationMs = 0;
      this.isPocketZupt = false;
      this.frozenHeading = null;
    }

    const effectiveHeading = this.isPocketZupt && this.frozenHeading !== null
      ? this.frozenHeading
      : currentHeadingDeg;

    if (this.isPocketZupt) {
      return {
        isStep: false,
        strideMeters: 0,
        isPocketZupt: true,
        effectiveHeadingDeg: effectiveHeading,
        currentVariance: variance,
      };
    }

    this.zBuffer.push(az);
    if (this.zBuffer.length > this.maWindowSize) {
      this.zBuffer.shift();
    }
    const smoothedZ = this.zBuffer.reduce((a, b) => a + b, 0) / this.zBuffer.length;

    this.smoothedZHistory.push(smoothedZ);
    if (this.smoothedZHistory.length > this.windowSize) {
      this.smoothedZHistory.shift();
    }

    if (this.smoothedZHistory.length < 8) {
      return {
        isStep: false,
        strideMeters: 0,
        isPocketZupt: false,
        effectiveHeadingDeg: effectiveHeading,
        currentVariance: variance,
      };
    }

    let localMin = Infinity;
    let localMax = -Infinity;
    const len = this.smoothedZHistory.length;

    for (let i = 0; i < len; i++) {
      const val = this.smoothedZHistory[i];
      if (val < localMin) localMin = val;
      if (val > localMax) localMax = val;
    }

    const swing = localMax - localMin;
    const timeSinceLastStep = timestamp - this.lastStepTime;

    const midIdx = len - 2;
    const isPeak =
      midIdx > 0 &&
      this.smoothedZHistory[midIdx] >= this.smoothedZHistory[midIdx - 1] &&
      this.smoothedZHistory[midIdx] > this.smoothedZHistory[midIdx + 1];

    let isStep = false;
    if (isPeak && swing > this.dynamicThreshold && timeSinceLastStep > this.minStepIntervalMs) {
      this.stepCount++;
      this.lastStepIntervalMs = timeSinceLastStep;
      this.lastStepTime = timestamp;
      isStep = true;
    }

    return {
      isStep,
      strideMeters: isStep ? this.strideLength : 0,
      isPocketZupt: false,
      effectiveHeadingDeg: effectiveHeading,
      currentVariance: variance,
    };
  }
}

const detector = new StepDetector();
let x = 0, y = 0;
const heading = 45; // 45 degrees
console.log('--- TEST 1: Walking Footsteps (1.5Hz step frequency, 2.0 m/s² vibration swing) ---');
let stepsDetected = 0;

for (let t = 0; t < 180; t++) {
  const timeMs = t * (1000 / 60);
  const az = 9.81 + 2.0 * Math.sin((2 * Math.PI * 1.5 * t) / 60);
  const ax = 0.5 * Math.sin(t);
  const ay = 0.5 * Math.cos(t);

  const res = detector.processSample(ax, ay, az, heading, timeMs);
  if (res.isStep) {
    stepsDetected++;
    const thetaRad = (heading * Math.PI) / 180;
    x += 0.65 * Math.sin(thetaRad);
    y += 0.65 * Math.cos(thetaRad);
    console.log(`  Step #${stepsDetected} @ ${timeMs.toFixed(0)}ms: Stride=${res.strideMeters}m -> Pos=(${x.toFixed(2)}, ${y.toFixed(2)})`);
  }
}

console.log('\n--- TEST 2: Pocket ZUPT (Stationary for > 0.5s, variance < 0.2 m/s²) ---');
let zuptFired = false;
for (let t = 180; t < 270; t++) {
  const timeMs = t * (1000 / 60);
  const az = 9.81 + (Math.random() - 0.5) * 0.05;
  const ax = (Math.random() - 0.5) * 0.05;
  const ay = (Math.random() - 0.5) * 0.05;

  const res = detector.processSample(ax, ay, az, 90, timeMs);
  if (res.isPocketZupt) {
    zuptFired = true;
    console.log(`  Pocket ZUPT Active @ ${timeMs.toFixed(0)}ms! Heading frozen at: ${res.effectiveHeadingDeg} deg (Var: ${res.currentVariance.toFixed(4)} < 0.20)`);
    break;
  }
}

console.log('\nVerification Summary:');
console.log(`Total Steps Detected in 3s: ${stepsDetected}`);
console.log(`Final Odometry Position: (${x.toFixed(3)}, ${y.toFixed(3)}) meters`);
console.log(`Pocket ZUPT Correctly Triggered: ${zuptFired}`);
