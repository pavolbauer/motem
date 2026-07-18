import { FilesetResolver, PoseLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// BlazePose 33-landmark skeleton (body only, face kept to nose dot)
export const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],          // shoulders + arms
  [15,17],[15,19],[15,21],[16,18],[16,20],[16,22],  // hands
  [11,23],[12,24],[23,24],                          // torso
  [23,25],[25,27],[24,26],[26,28],                  // legs
  [27,29],[29,31],[27,31],[28,30],[30,32],[28,32],  // feet
];

export class PoseEngine {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.facing = 'user';
    this.mirror = true;
    this.stream = null;
    this.landmarker = null;
    this.onPose = null;      // (landmarks|null, tMs) => void
    this.running = false;
  }

  async openCamera() {
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: this.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    this.mirror = (this.facing === 'user');
    this.video.style.transform = this.mirror ? 'scaleX(-1)' : 'none';
    await this.video.play();
  }

  async flip() {
    this.facing = this.facing === 'user' ? 'environment' : 'user';
    try { await this.openCamera(); }
    catch (e) { this.facing = this.facing === 'user' ? 'environment' : 'user'; await this.openCamera(); }
  }

  async initLandmarker(onStatus) {
    onStatus?.('loading pose model…');
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
    const make = (delegate) => PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: "VIDEO", numPoses: 1,
      minPoseDetectionConfidence: 0.4, minTrackingConfidence: 0.4,
    });
    try { this.landmarker = await make('GPU'); }
    catch (e) { this.landmarker = await make('CPU'); }
  }

  start() {
    this.running = true;
    let lastVideoTime = -1, lastFrame = 0;
    const loop = (t) => {
      if (!this.running) return;
      requestAnimationFrame(loop);
      if (t - lastFrame < 30) return;
      lastFrame = t;
      if (this.video.readyState < 2) return;
      if (this.video.currentTime === lastVideoTime) return;
      lastVideoTime = this.video.currentTime;
      let lm = null;
      try {
        const res = this.landmarker.detectForVideo(this.video, performance.now());
        lm = (res.landmarks && res.landmarks[0]) || null;
      } catch (e) { /* skip frame */ }
      this.draw(lm);
      this.onPose?.(lm, performance.now());
    };
    requestAnimationFrame(loop);
  }

  // map normalized landmark coords onto the cover-fitted video
  mapPoint(p, W, H) {
    const vw = this.video.videoWidth || 1280, vh = this.video.videoHeight || 720;
    const s = Math.max(W / vw, H / vh);
    const dw = vw * s, dh = vh * s;
    let x = (W - dw) / 2 + p.x * dw;
    const y = (H - dh) / 2 + p.y * dh;
    if (this.mirror) x = W - x;
    return [x, y];
  }

  draw(lm) {
    const c = this.canvas;
    const W = c.width = c.clientWidth * devicePixelRatio;
    const H = c.height = c.clientHeight * devicePixelRatio;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, W, H);
    if (!lm) return;
    ctx.strokeStyle = 'rgba(94,234,212,.9)';
    ctx.lineWidth = 2.5 * devicePixelRatio;
    ctx.beginPath();
    POSE_CONNECTIONS.forEach(([a, b]) => {
      const pa = this.mapPoint(lm[a], W, H), pb = this.mapPoint(lm[b], W, H);
      ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
    });
    ctx.stroke();
    ctx.fillStyle = '#f472b6';
    lm.forEach((p, i) => {
      if (i > 0 && i < 11) return; // skip face except nose
      const [x, y] = this.mapPoint(p, W, H);
      ctx.beginPath(); ctx.arc(x, y, 3 * devicePixelRatio, 0, 7); ctx.fill();
    });
  }
}
