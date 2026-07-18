// Motion VAE trained fully in-browser with TF.js.
// Two architectures over the same [WIN, FEAT_DIM] window:
//  - 'conv': temporal Conv1D encoder + dense decoder (fast, phone-friendly)
//  - 'gru' : bidirectional GRU encoder + GRU decoder (VAME-style)
// Both use the VAE lambda: hidden → mu / logvar, reparameterized sample,
// MSE reconstruction + KL with linear annealing (as in VAME's rnn_vae.py).

export class MotionVAE {
  constructor({ win = 30, dim = 66, zDim = 16, arch = 'conv' } = {}) {
    this.win = win; this.dim = dim; this.z = zDim; this.arch = arch;
    this.enc = null; this.dec = null;
  }

  build() {
    const inp = tf.input({ shape: [this.win, this.dim] });
    let h;
    if (this.arch === 'gru') {
      h = tf.layers.bidirectional({
        layer: tf.layers.gru({ units: 48, returnSequences: false }),
        mergeMode: 'concat',
      }).apply(inp);
    } else {
      h = tf.layers.conv1d({ filters: 48, kernelSize: 5, strides: 2, activation: 'relu', padding: 'same' }).apply(inp);
      h = tf.layers.conv1d({ filters: 64, kernelSize: 5, strides: 2, activation: 'relu', padding: 'same' }).apply(h);
      h = tf.layers.flatten().apply(h);
    }
    h = tf.layers.dense({ units: 96, activation: 'relu' }).apply(h);
    const mu = tf.layers.dense({ units: this.z, name: 'mu' }).apply(h);
    const lv = tf.layers.dense({ units: this.z, name: 'logvar' }).apply(h);
    this.enc = tf.model({ inputs: inp, outputs: [mu, lv] });

    const zin = tf.input({ shape: [this.z] });
    let d;
    if (this.arch === 'gru') {
      d = tf.layers.repeatVector({ n: this.win }).apply(zin);
      d = tf.layers.gru({ units: 96, returnSequences: true }).apply(d);
      d = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: this.dim }) }).apply(d);
    } else {
      d = tf.layers.dense({ units: 96, activation: 'relu' }).apply(zin);
      d = tf.layers.dense({ units: 256, activation: 'relu' }).apply(d);
      d = tf.layers.dense({ units: this.win * this.dim }).apply(d);
      d = tf.layers.reshape({ targetShape: [this.win, this.dim] }).apply(d);
    }
    this.dec = tf.model({ inputs: zin, outputs: d });
  }

  async train(windows, { epochs, batch = 32, beta = 0.005, klStart = 6, anneal = 10, onProgress } = {}) {
    epochs = epochs ?? (this.arch === 'gru' ? 25 : 45);
    const n = windows.length;
    const xs = tf.tensor3d(windows);
    const opt = tf.train.adam(1e-3);
    let lastLoss = 0;
    for (let e = 0; e < epochs; e++) {
      const perm = tf.util.createShuffledIndices(n);
      const klW = e <= klStart ? 0 : Math.min(1, (e - klStart) / anneal);
      let sum = 0, steps = 0;
      for (let i = 0; i < n; i += batch) {
        const idx = tf.tensor1d(new Int32Array(perm.slice(i, i + batch)), 'int32');
        const xb = tf.gather(xs, idx);
        idx.dispose();
        const cost = opt.minimize(() => tf.tidy(() => {
          const [mu, lv] = this.enc.apply(xb);
          const eps = tf.randomNormal(mu.shape);
          const z = mu.add(lv.mul(0.5).exp().mul(eps));
          const rec = this.dec.apply(z);
          const mse = tf.losses.meanSquaredError(xb, rec);
          const kl = tf.scalar(1).add(lv).sub(mu.square()).sub(lv.exp())
            .mean().mul(-0.5);
          return mse.add(kl.mul(beta * klW));
        }), true);
        sum += (await cost.data())[0];
        cost.dispose(); xb.dispose();
        steps++;
      }
      lastLoss = sum / steps;
      onProgress?.(e + 1, epochs, lastLoss);
      await tf.nextFrame();
    }
    xs.dispose(); opt.dispose();
    return lastLoss;
  }

  // deterministic embedding = mu (as VAME does at inference)
  encode(window) {
    return tf.tidy(() => {
      const [mu] = this.enc.predict(tf.tensor3d([window]));
      return Array.from(mu.dataSync());
    });
  }

  encodeAll(windows, batch = 128) {
    const out = [];
    for (let i = 0; i < windows.length; i += batch) {
      const part = windows.slice(i, i + batch);
      tf.tidy(() => {
        const [mu] = this.enc.predict(tf.tensor3d(part));
        const d = mu.dataSync();
        for (let j = 0; j < part.length; j++) out.push(Array.from(d.slice(j * this.z, (j + 1) * this.z)));
      });
    }
    return out;
  }

  // the network run backwards: latent → motion window [win][dim]
  decode(latent) {
    return tf.tidy(() => {
      const w = this.dec.predict(tf.tensor2d([latent], [1, this.z]));
      return w.squeeze().arraySync();
    });
  }

  async save() {
    await this.enc.save('indexeddb://motem-enc');
    await this.dec.save('indexeddb://motem-dec');
    localStorage.setItem('motem_vae_meta', JSON.stringify(
      { win: this.win, dim: this.dim, z: this.z, arch: this.arch }));
  }

  static async load() {
    const meta = JSON.parse(localStorage.getItem('motem_vae_meta') || 'null');
    if (!meta) return null;
    try {
      const vae = new MotionVAE({ win: meta.win, dim: meta.dim, zDim: meta.z, arch: meta.arch });
      vae.enc = await tf.loadLayersModel('indexeddb://motem-enc');
      vae.dec = await tf.loadLayersModel('indexeddb://motem-dec');
      return vae;
    } catch (e) { return null; }
  }

  // pretrained weights served next to the app (see EXPORT WEIGHTS)
  static async loadFromUrl(base) {
    try {
      const res = await fetch(base + 'motem-meta.json', { cache: 'no-store' });
      if (!res.ok) return null;
      const meta = await res.json();
      const v = meta.vae;
      const vae = new MotionVAE({ win: v.win, dim: v.dim, zDim: v.z, arch: v.arch });
      vae.enc = await tf.loadLayersModel(base + 'motem-enc.json');
      vae.dec = await tf.loadLayersModel(base + 'motem-dec.json');
      return { vae, meta };
    } catch (e) { return null; }
  }
}

// Deterministic weight seeding for the no-training demo space: same seed →
// identical weights on every device, so demo embeddings stay comparable.
// Kernels get seeded uniform values (±gain/√fanIn), biases zero.
export function seedWeights(model, seed, gain = 1.6) {
  let s = seed >>> 0;
  const rnd = () => {                    // mulberry32
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const ws = model.getWeights().map(w => {
    if (w.shape.length === 1) return tf.zeros(w.shape);
    const fanIn = w.shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const lim = gain / Math.sqrt(fanIn);
    const vals = new Float32Array(w.size);
    for (let i = 0; i < vals.length; i++) vals[i] = (rnd() * 2 - 1) * lim;
    return tf.tensor(vals, w.shape);
  });
  model.setWeights(ws);
  ws.forEach(w => w.dispose());
}
