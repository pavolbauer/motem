// Pose → egocentric features and sliding temporal windows.
// Analogous to VAME's egocentric alignment: hip-centered, torso-scaled.

export const NUM_LM = 33;
export const FEAT_DIM = NUM_LM * 2;   // x,y per landmark (z is too noisy in-browser)
export const WIN = 30;                // temporal window length (~2 s at 15 fps)
export const SAMPLE_MS = 1000 / 15;   // feature sampling rate

export function poseToFeatures(lm) {
  const hipX = (lm[23].x + lm[24].x) / 2, hipY = (lm[23].y + lm[24].y) / 2;
  const shX = (lm[11].x + lm[12].x) / 2, shY = (lm[11].y + lm[12].y) / 2;
  const torso = Math.hypot(shX - hipX, shY - hipY) || 1e-4; // scale reference
  const f = new Float32Array(FEAT_DIM);
  for (let i = 0; i < NUM_LM; i++) {
    f[i * 2]     = clamp((lm[i].x - hipX) / torso, -4, 4);
    f[i * 2 + 1] = clamp((lm[i].y - hipY) / torso, -4, 4);
  }
  return f;
}

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export class WindowBuffer {
  constructor(win = WIN) {
    this.win = win;
    this.frames = [];
    this.lastT = 0;
  }
  push(feat, t) {
    if (this.lastT && t - this.lastT > 500) this.frames.length = 0; // tracking gap → reset
    this.lastT = t;
    this.frames.push(feat);
    if (this.frames.length > this.win) this.frames.shift();
  }
  full() { return this.frames.length === this.win; }
  // copy as [win][dim] plain arrays (safe to keep / feed tf.tensor3d)
  snapshot() { return this.frames.map(f => Array.from(f)); }
  reset() { this.frames.length = 0; this.lastT = 0; }
}
