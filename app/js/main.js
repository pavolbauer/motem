import { PoseEngine } from './pose.js';
import { poseToFeatures, WindowBuffer, WIN, FEAT_DIM, SAMPLE_MS } from './features.js';
import { MotionVAE } from './vae.js';
import { Projector } from './projector.js';
import { EmbedView } from './scatter.js';
import { Replayer } from './replay.js';
import * as db from './sessions.js';
import { analyze, renderCompare, nearestAcross } from './compare.js';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const setStatus = (t) => statusEl.textContent = t;
const toast = (t) => {
  const el = $('toast');
  el.textContent = t; el.classList.add('show');
  clearTimeout(toast._t); toast._t = setTimeout(() => el.classList.remove('show'), 2600);
};

const state = {
  mode: 'live',
  arch: 'conv',
  modelReady: false,
  modelId: null,
  collecting: false,        // calibration data recording
  recording: false,         // session recording
  calibWindows: [],
  sessLatents: [],          // parallel to view.points while recording
  compare: null,            // {a, b, an} when a comparison is shown
  training: false,
};

let vae = null, proj = null;
const pose = new PoseEngine($('cam'), $('poseCanvas'));
const view = new EmbedView($('embedCanvas'));
const replayer = new Replayer($('replayWrap'), $('replayCanvas'), $('replayLabel'));
const buf = new WindowBuffer(WIN);

/* ---------------- pose stream → features → embedding ---------------- */

let lastSample = 0, embedSkip = 0, calibSkip = 0;

pose.onPose = (lm, t) => {
  if (state.mode !== 'compare') view.render();
  if (!lm) return;
  if (t - lastSample < SAMPLE_MS - 5) return;
  lastSample = t;
  buf.push(poseToFeatures(lm), t);
  if (!buf.full()) return;

  if (state.collecting && ++calibSkip >= 3) {   // ~5 windows/s while calibrating
    calibSkip = 0;
    state.calibWindows.push(buf.snapshot());
    $('calibCount').textContent = state.calibWindows.length;
    $('trainBtn').disabled = state.calibWindows.length < 60 || state.training;
  }

  if (state.modelReady && !state.training && ++embedSkip >= 4) { // ~3.7 pts/s
    embedSkip = 0;
    const z = vae.encode(buf.snapshot());
    const [x, y] = proj.project(z);
    view.current = [x, y];
    if (state.recording) {
      view.add(x, y, performance.now());
      state.sessLatents.push(z);
    }
  }
};

/* ---------------- embedding tile clicks → decoder replay ---------------- */

view.clickCb = (nx, ny) => {
  if (state.mode === 'compare' && state.compare) {
    const hit = nearestAcross(state.compare.a, state.compare.b, nx, ny);
    if (!hit) return;
    playLatent(hit.sess.latents[hit.index],
      `${hit.tag} · ${hit.sess.name} · decoded from latent`);
  } else if (view.points.length) {
    const i = view.nearest(nx, ny);
    if (i < 0) return;
    playLatent(state.sessLatents[i], 'this session · decoded from latent');
  }
};

function playLatent(latent, label) {
  if (!vae || !latent) return;
  replayer.play(vae.decode(latent), { label });
}

/* ---------------- modes ---------------- */

document.querySelectorAll('.modeBtn').forEach(btn => {
  btn.onclick = () => setMode(btn.dataset.mode);
});

function setMode(m) {
  state.mode = m;
  document.querySelectorAll('.modeBtn').forEach(b =>
    b.classList.toggle('on', b.dataset.mode === m));
  $('calibPanel').classList.toggle('show', m === 'calibrate');
  $('comparePanel').classList.toggle('show', m === 'compare');
  $('embedUI').style.display = m === 'live' ? 'flex' : 'none';
  replayer.stop();
  if (m === 'compare') refreshSessionUI();
  if (m !== 'compare') state.compare = null;
  if (m !== 'calibrate' && state.collecting) toggleCalibRec();
  updateStatus();
}

function updateStatus() {
  if (state.training) return; // training progress owns the status line
  if (!state.modelReady) setStatus('no model yet — CALIBRATE first');
  else if (state.recording) setStatus(`recording · ${view.points.length} pts`);
  else setStatus(`model ready (${vae.arch}, z=${vae.z}) — REC to record a session`);
  $('recBtn').disabled = !state.modelReady;
  $('saveBtn').disabled = !(state.modelReady && !state.recording && view.points.length > 0);
}

/* ---------------- live: record & save sessions ---------------- */

$('recBtn').onclick = () => {
  state.recording = !state.recording;
  if (state.recording) {
    view.reset();
    state.sessLatents = [];
    $('recBtn').textContent = '■ STOP';
    $('recBtn').classList.add('rec');
  } else {
    $('recBtn').textContent = '● REC';
    $('recBtn').classList.remove('rec');
  }
  updateStatus();
};

$('saveBtn').onclick = async () => {
  const name = $('sessName').value.trim() ||
    `session ${new Date().toLocaleTimeString()}`;
  await db.saveSession({
    id: Date.now(),
    name,
    createdAt: Date.now(),
    modelId: state.modelId,
    points: view.points.map(p => ({ x: p.x, y: p.y, t: p.t })),
    latents: state.sessLatents,
  });
  $('sessName').value = '';
  toast(`saved “${name}” (${view.points.length} pts)`);
};

/* ---------------- calibrate & train ---------------- */

$('archPick').querySelectorAll('button').forEach(b => {
  b.onclick = () => {
    state.arch = b.dataset.arch;
    $('archPick').querySelectorAll('button').forEach(x =>
      x.classList.toggle('on', x === b));
  };
});

$('calibRecBtn').onclick = toggleCalibRec;
function toggleCalibRec() {
  state.collecting = !state.collecting;
  $('calibRecBtn').textContent = state.collecting ? '■ STOP' : '● RECORD DATA';
  $('calibMsg').textContent = state.collecting
    ? 'Move! Cover the full movement vocabulary you want mapped.' : '';
}

const setProg = (id, v) => $(id).style.width = `${Math.round(v * 100)}%`;

$('trainBtn').onclick = async () => {
  if (state.training) return;
  if (state.collecting) toggleCalibRec();
  state.training = true;
  $('trainBtn').disabled = true;
  ['p1', 'p2', 'p3'].forEach(p => setProg(p, 0));
  try {
    const windows = state.calibWindows;
    $('calibMsg').textContent = `training on ${windows.length} windows…`;

    vae = new MotionVAE({ win: WIN, dim: FEAT_DIM, zDim: 16, arch: state.arch });
    vae.build();
    await vae.train(windows, {
      onProgress: (e, tot, loss) => {
        setProg('p1', e / tot);
        setStatus(`VAE ${e}/${tot} · loss ${loss.toFixed(4)}`);
      },
    });

    const latents = vae.encodeAll(windows);
    proj = new Projector(vae.z);
    await proj.fit(latents, {
      onUmap: (e, tot) => { setProg('p2', e / tot); setStatus(`UMAP ${e}/${tot}`); },
      onMlp: (e, tot) => { setProg('p3', e / tot); setStatus(`projector ${e}/${tot}`); },
    });

    await vae.save();
    await proj.save();
    state.modelId = Date.now();
    localStorage.setItem('motem_model_id', state.modelId);
    state.modelReady = true;
    state.calibWindows = [];
    $('calibCount').textContent = '0';
    $('calibMsg').textContent = 'Done — space is frozen. Go LIVE and record sessions.';
    toast('model trained & saved');
    view.reset();
    setMode('live');
  } catch (e) {
    console.error(e);
    $('calibMsg').textContent = 'Training failed: ' + e.message;
  } finally {
    state.training = false;
    updateStatus();
  }
};

/* ---------------- compare ---------------- */

async function refreshSessionUI() {
  const sessions = await db.listSessions();
  const opts = sessions.map(s =>
    `<option value="${s.id}">${s.name} (${s.points.length})</option>`).join('');
  $('selA').innerHTML = opts;
  $('selB').innerHTML = opts;
  if (sessions.length > 1) $('selB').selectedIndex = 1;
  $('sessList').innerHTML = sessions.length
    ? '<b>saved sessions</b><br>' + sessions.map(s =>
        `${s.name} — ${s.points.length} pts, ${new Date(s.createdAt).toLocaleString()}` +
        `<span class="del" data-id="${s.id}">✕</span>`).join('<br>')
    : 'no saved sessions yet — record some in LIVE mode';
  $('sessList').querySelectorAll('.del').forEach(el => {
    el.onclick = async () => { await db.deleteSession(+el.dataset.id); refreshSessionUI(); };
  });
}

$('cmpBtn').onclick = async () => {
  const a = await db.getSession(+$('selA').value);
  const b = await db.getSession(+$('selB').value);
  if (!a || !b) { toast('pick two sessions'); return; }
  const an = analyze(a, b);
  state.compare = { a, b, an };
  const warn = (a.modelId !== state.modelId || b.modelId !== state.modelId)
    ? '<br>⚠ recorded with a different model — space may not match' : '';
  $('cmpStats').innerHTML =
    `<span class="both">■ overlap ${(an.jaccard * 100).toFixed(0)}%</span> (Jaccard)<br>` +
    `<span class="a">■ only in ${a.name}: ${(an.novelA * 100).toFixed(0)}%</span><br>` +
    `<span class="b">■ only in ${b.name}: ${(an.novelB * 100).toFixed(0)}%</span>` + warn;
  $('comparePanel').classList.remove('show');   // reveal the map
  renderCompare(view, a, b, an);
  toast('tap the map to replay motions · COMPARE button returns to setup');
};

// re-render compare on resize/orientation change
window.addEventListener('resize', () => {
  if (state.mode === 'compare' && state.compare)
    setTimeout(() => renderCompare(view, state.compare.a, state.compare.b, state.compare.an), 120);
});

/* ---------------- boot ---------------- */

$('flipBtn').onclick = () => pose.flip().catch(e => toast(e.message));

$('startBtn').onclick = async () => {
  const bs = $('bootStatus'), be = $('bootErr');
  be.textContent = '';
  try {
    bs.textContent = 'starting TensorFlow.js…';
    await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
    await tf.ready();

    bs.textContent = 'opening camera…';
    await pose.openCamera();

    await pose.initLandmarker((t) => bs.textContent = t);

    bs.textContent = 'loading saved model…';
    vae = await MotionVAE.load();
    if (vae) proj = await Projector.load(vae.z);
    if (vae && proj) {
      state.modelReady = true;
      state.modelId = +localStorage.getItem('motem_model_id') || Date.now();
    }

    try { await navigator.wakeLock.request('screen'); } catch (e) {}
    pose.start();
    $('curtain').style.display = 'none';
    setMode(state.modelReady ? 'live' : 'calibrate');
    if (!state.modelReady) toast('first run: record calibration data, then train');
  } catch (e) {
    bs.textContent = '';
    be.textContent = 'Could not start: ' + e.message +
      ' — camera needs https:// or localhost.';
  }
};
