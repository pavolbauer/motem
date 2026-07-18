// Embedding scatter view. Coordinate space is fixed to [-RANGE, RANGE]
// (the projector emits ~[-1,1]), so live points, saved sessions and the
// compare overlay all share one frame.

export const RANGE = 1.15;

export class EmbedView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.layer = document.createElement('canvas'); // accumulated points
    this.lctx = this.layer.getContext('2d');
    this.points = [];           // {x, y, t}
    this.current = null;        // live cursor [x, y]
    this.startT = 0;
    this.clickCb = null;
    canvas.addEventListener('click', (e) => {
      if (!this.clickCb) return;
      const r = canvas.getBoundingClientRect();
      const nx = ((e.clientX - r.left) / r.width * 2 - 1) * RANGE;
      const ny = ((e.clientY - r.top) / r.height * 2 - 1) * RANGE;
      this.clickCb(nx, ny);
    });
  }

  resize() {
    const W = this.canvas.clientWidth * devicePixelRatio;
    const H = this.canvas.clientHeight * devicePixelRatio;
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W; this.canvas.height = H;
      this.layer.width = W; this.layer.height = H;
      this.restamp();
    }
  }

  toPx(x, y) {
    return [
      (x / RANGE + 1) / 2 * this.canvas.width,
      (y / RANGE + 1) / 2 * this.canvas.height,
    ];
  }

  colorFor(t) { // teal → magenta over 4 minutes of recording
    const h = 175 + Math.min(1, (t - this.startT) / 240000) * 140;
    return `hsla(${h},85%,65%,.85)`;
  }

  reset() {
    this.points.length = 0;
    this.startT = performance.now();
    this.lctx.clearRect(0, 0, this.layer.width, this.layer.height);
  }

  add(x, y, t) {
    this.points.push({ x, y, t });
    this.stamp(x, y, t);
  }

  stamp(x, y, t) {
    const [px, py] = this.toPx(x, y);
    this.lctx.fillStyle = this.colorFor(t);
    this.lctx.beginPath();
    this.lctx.arc(px, py, 2.2 * devicePixelRatio, 0, 7);
    this.lctx.fill();
  }

  restamp() {
    this.lctx.clearRect(0, 0, this.layer.width, this.layer.height);
    this.points.forEach(p => this.stamp(p.x, p.y, p.t));
  }

  render() {
    this.resize();
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    // grid
    ctx.strokeStyle = 'rgba(38,48,79,.5)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
      const gx = W * i / 6, gy = H * i / 6;
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }
    ctx.drawImage(this.layer, 0, 0);
    if (this.current) {
      const [px, py] = this.toPx(this.current[0], this.current[1]);
      const pulse = 1 + 0.25 * Math.sin(performance.now() / 180);
      ctx.strokeStyle = 'rgba(94,234,212,.9)';
      ctx.lineWidth = 2 * devicePixelRatio;
      ctx.beginPath(); ctx.arc(px, py, 7 * devicePixelRatio * pulse, 0, 7); ctx.stroke();
      ctx.fillStyle = '#5eead4';
      ctx.beginPath(); ctx.arc(px, py, 3 * devicePixelRatio, 0, 7); ctx.fill();
    }
  }

  // nearest recorded point to normalized coords, within maxDist
  nearest(nx, ny, maxDist = 0.1) {
    let best = -1, bd = maxDist * maxDist;
    this.points.forEach((p, i) => {
      const d = (p.x - nx) ** 2 + (p.y - ny) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }
}
