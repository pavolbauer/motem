import { RANGE } from './scatter.js';

// Overlap / novelty analysis between two sessions on an occupancy grid
// over the shared embedding frame.

const GRID = 40;
const COL_A = '79,156,255', COL_B = '255,140,66', COL_BOTH = '61,220,132';

function occupancy(points) {
  const set = new Set();
  points.forEach(p => {
    const gx = Math.floor((p.x / RANGE + 1) / 2 * GRID);
    const gy = Math.floor((p.y / RANGE + 1) / 2 * GRID);
    if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) set.add(gy * GRID + gx);
  });
  return set;
}

export function analyze(sessA, sessB) {
  const a = occupancy(sessA.points), b = occupancy(sessB.points);
  const both = new Set([...a].filter(c => b.has(c)));
  const onlyA = new Set([...a].filter(c => !b.has(c)));
  const onlyB = new Set([...b].filter(c => !a.has(c)));
  const union = a.size + b.size - both.size;
  return {
    a, b, both, onlyA, onlyB,
    jaccard: union ? both.size / union : 0,
    novelB: b.size ? onlyB.size / b.size : 0,
    novelA: a.size ? onlyA.size / a.size : 0,
  };
}

export function renderCompare(view, sessA, sessB, an) {
  view.resize();
  const { ctx, canvas } = view;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const cw = W / GRID, ch = H / GRID;
  const cell = (idx, rgb, alpha) => {
    ctx.fillStyle = `rgba(${rgb},${alpha})`;
    ctx.fillRect((idx % GRID) * cw, Math.floor(idx / GRID) * ch, cw, ch);
  };
  an.onlyA.forEach(c => cell(c, COL_A, 0.22));
  an.onlyB.forEach(c => cell(c, COL_B, 0.22));
  an.both.forEach(c => cell(c, COL_BOTH, 0.30));

  const dot = (p, rgb) => {
    const [px, py] = view.toPx(p.x, p.y);
    ctx.fillStyle = `rgba(${rgb},.75)`;
    ctx.beginPath(); ctx.arc(px, py, 1.6 * devicePixelRatio, 0, 7); ctx.fill();
  };
  sessA.points.forEach(p => dot(p, COL_A));
  sessB.points.forEach(p => dot(p, COL_B));
}

// nearest point across both sessions → {sess, index} for decoder replay
export function nearestAcross(sessA, sessB, nx, ny, maxDist = 0.12) {
  let best = null, bd = maxDist * maxDist;
  const scan = (sess, tag) => sess.points.forEach((p, i) => {
    const d = (p.x - nx) ** 2 + (p.y - ny) ** 2;
    if (d < bd) { bd = d; best = { sess, tag, index: i }; }
  });
  scan(sessA, 'A'); scan(sessB, 'B');
  return best;
}
