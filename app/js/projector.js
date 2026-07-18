// Parametric UMAP head: fit classic UMAP on the calibration latents once,
// then train a small MLP to reproduce that projection. The MLP is the
// parametric embedder — every future latent maps into the SAME frozen 2D
// space, which is what makes sessions comparable.

const UMAP_EPOCHS = 250;

export class Projector {
  constructor(zDim = 16) {
    this.z = zDim;
    this.mlp = null;
    this.norm = null; // {cx, cy, sx, sy} → map UMAP coords to [-1, 1]
  }

  async fit(latents, { onUmap, onMlp } = {}) {
    const UMAPClass = (window.UMAP && window.UMAP.UMAP) || window.UMAP;
    const nn = Math.max(5, Math.min(30, Math.floor(Math.sqrt(latents.length))));
    const umap = new UMAPClass({
      nComponents: 2, nNeighbors: nn, minDist: 0.1, nEpochs: UMAP_EPOCHS,
    });
    const coords = await umap.fitAsync(latents, (ep) => {
      onUmap?.(ep, UMAP_EPOCHS);
      return true;
    });

    // normalize the map into [-1,1] with margin; this frame is frozen forever
    let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    coords.forEach(([x, y]) => {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    });
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const sx = (maxX - minX) / 2 / 0.85 || 1, sy = (maxY - minY) / 2 / 0.85 || 1;
    this.norm = { cx, cy, sx, sy };
    const target = coords.map(([x, y]) => [(x - cx) / sx, (y - cy) / sy]);

    this.mlp = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [this.z], units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 2 }),
      ],
    });
    this.mlp.compile({ optimizer: tf.train.adam(2e-3), loss: 'meanSquaredError' });
    const xsT = tf.tensor2d(latents), ysT = tf.tensor2d(target);
    const EP = 150;
    await this.mlp.fit(xsT, ysT, {
      epochs: EP, batchSize: 32, shuffle: true, verbose: 0,
      callbacks: { onEpochEnd: async (e) => { onMlp?.(e + 1, EP); await tf.nextFrame(); } },
    });
    xsT.dispose(); ysT.dispose();
    return target; // normalized calibration cloud (handy for debugging)
  }

  project(latent) {
    return tf.tidy(() => {
      const p = this.mlp.predict(tf.tensor2d([latent], [1, this.z]));
      const d = p.dataSync();
      return [d[0], d[1]];
    });
  }

  async save() {
    await this.mlp.save('indexeddb://motem-proj');
    localStorage.setItem('motem_proj_norm', JSON.stringify(this.norm));
  }

  static async fromUrl(base, norm, zDim) {
    try {
      const p = new Projector(zDim);
      p.mlp = await tf.loadLayersModel(base + 'motem-proj.json');
      p.norm = norm;
      return p;
    } catch (e) { return null; }
  }

  static async load(zDim) {
    const norm = JSON.parse(localStorage.getItem('motem_proj_norm') || 'null');
    if (!norm) return null;
    try {
      const p = new Projector(zDim);
      p.mlp = await tf.loadLayersModel('indexeddb://motem-proj');
      p.norm = norm;
      return p;
    } catch (e) { return null; }
  }
}
