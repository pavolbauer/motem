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
2. **LIVE** — one tile shows the tracked pose, the other the map. ● REC
   accumulates the trace (all points since recording start + pulsing current
   point). SAVE stores the session (points + latents) in IndexedDB.
   Tap any point to replay its decoded motion.
3. **COMPARE** — pick two sessions → occupancy-grid overlap analysis:
   green = shared movement regions, blue/orange = movement unique to each
   session, with Jaccard overlap and novelty percentages. Tap a region to
   decode and replay the nearest latent.

Re-running CALIBRATE retrains and replaces the space; sessions saved under an
older model are flagged in COMPARE.

Tiles sit side-by-side in landscape, stacked in portrait (CSS
`orientation` media query).
