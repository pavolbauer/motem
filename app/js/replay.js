import { POSE_CONNECTIONS } from './pose.js';
import { NUM_LM } from './features.js';

// Animates a decoded motion window (hip-centered, torso-scaled features)
// as a stick figure — the generative direction of the pipeline.

export class Replayer {
  constructor(wrap, canvas, label) {
    this.wrap = wrap;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.label = label;
    this.raf = null;
    wrap.addEventListener('click', () => this.stop());
  }

  play(frames, { fps = 15, loops = 4, label = '' } = {}) {
    this.stop();
    this.wrap.classList.add('show');
    this.label.textContent = label;
    const t0 = performance.now();
    const total = frames.length * loops;
    const step = (t) => {
      const i = Math.floor((t - t0) / (1000 / fps));
      if (i >= total) { this.stop(); return; }
      this.drawFrame(frames[i % frames.length]);
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
    this.wrap.classList.remove('show');
  }

  drawFrame(feat) {
    const c = this.canvas;
    const W = c.width = c.clientWidth * devicePixelRatio;
    const H = c.height = c.clientHeight * devicePixelRatio;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, W, H);
    const S = Math.min(W, H) / 4.2;           // torso length in px
    const cx = W / 2, cy = H * 0.45;          // hip anchor
    const pt = (i) => [cx + feat[i * 2] * S, cy + feat[i * 2 + 1] * S];
    ctx.strokeStyle = 'rgba(244,114,182,.95)';
    ctx.lineWidth = 3 * devicePixelRatio;
    ctx.lineCap = 'round';
    ctx.shadowColor = '#f472b6';
    ctx.shadowBlur = 12 * devicePixelRatio;
    ctx.beginPath();
    POSE_CONNECTIONS.forEach(([a, b]) => {
      const pa = pt(a), pb = pt(b);
      ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#5eead4';
    for (let i = 0; i < NUM_LM; i++) {
      if (i > 0 && i < 11) continue;
      const [x, y] = pt(i);
      ctx.beginPath(); ctx.arc(x, y, 3.5 * devicePixelRatio, 0, 7); ctx.fill();
    }
  }
}
