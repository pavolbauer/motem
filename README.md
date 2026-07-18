# MOTEM — whole-body motion embedding in the browser

Live pose → motion-VAE latent → parametric UMAP map. Everything (inference
**and training**) runs in the browser; nothing leaves the device.

Pipeline (a browser analogue of VAME's `rnn_vae.py`):

```
camera ──► MediaPipe PoseLandmarker (33 landmarks)
       ──► egocentric features (hip-centered, torso-scaled x,y = 66 dims)
       ──► sliding window (30 frames ≈ 2 s @ 15 fps)
       ──► motion VAE (TF.js; Conv1D fast / bidirectional-GRU VAME-style)
       ──► z = mu (16-dim latent)
       ──► parametric UMAP head (UMAP fit once + MLP that reproduces it)
       ──► fixed 2D map shared by all sessions
```

Clicking the map runs the network the other way: latent → VAE decoder →
animated stick figure.

## Run

```sh
cd app
python3 -m http.server 8123
# open http://localhost:8123
```

Camera requires a secure context: `localhost` works; for a phone use HTTPS
(e.g. `ngrok http 8123`, Tailscale serve, or `npx local-ssl-proxy`).

## Use

1. **CALIBRATE** — record 1–3 min covering your movement vocabulary, pick
   CONV (fast) or GRU (VAME-style), hit TRAIN. VAE + UMAP + projector train
   in-browser; weights persist in IndexedDB. This freezes the space — later
   sessions are all embedded by the same weights, so they are comparable.
   - **SKIP — DEMO SPACE**: start instantly without training. Uses a
     deterministically seeded random encoder + tanh projection (same weights
     on every device → embeddings still comparable), but decoded replays are
     meaningless until a real model is trained.
   - **EXPORT WEIGHTS**: downloads 7 files (enc/dec/proj json+bin +
     `motem-meta.json`). Commit them to a `model/` folder at the repo root
     (also copy into `app/model/` for local dev) and every visitor auto-loads
     your pretrained space — no calibration needed.
2. **LIVE** — one tile shows the tracked pose, the other the map. The whole
   trace since going live stays visible (time-gradient teal→magenta) plus the
   pulsing current point. ⟲ RESET clears the trace; SAVE stores it as a
   session (points + latents) in IndexedDB. Tap any point to replay its
   decoded motion.
3. **COMPARE** — pick two sessions → occupancy-grid overlap analysis:
   green = shared movement regions, blue/orange = movement unique to each
   session, with Jaccard overlap and novelty percentages. Tap a region to
   decode and replay the nearest latent.

Re-running CALIBRATE retrains and replaces the space; sessions saved under an
older model are flagged in COMPARE.

## Pose tracking options

Currently: **MediaPipe PoseLandmarker (BlazePose GHUM), 33 landmarks**
(x, y, z + visibility each); features use x,y of all 33 → 66 dims. The model
variant is set by the URL in `js/pose.js` (`pose_landmarker_lite`) — swap for
`pose_landmarker_full` or `pose_landmarker_heavy` for better accuracy at the
same 33 keypoints, slower inference. Alternatives if the tradeoff should
change: MoveNet (17 COCO keypoints, fastest, TF.js-native), MediaPipe
Holistic (33 pose + 2×21 hands + 468 face) or adding HandLandmarker
(21/hand, Air-Hendrix style) if hand detail matters. Feature-level knobs in
`js/features.js`: drop the 10 face points (→ 25 kpts / 50 dims, less noise),
add z, change window length.

Tiles sit side-by-side in landscape, stacked in portrait (CSS
`orientation` media query).
