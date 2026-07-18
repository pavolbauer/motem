#!/usr/bin/env python3
"""Bundle the modular app into a single self-contained dist/index.html.

Concatenates the ES modules in dependency order, strips inter-module
imports/exports (they all share one module scope after bundling), hoists the
MediaPipe CDN import, and inlines the result into index.html.
"""
import re
from pathlib import Path

APP = Path(__file__).parent
ORDER = ['pose.js', 'features.js', 'vae.js', 'projector.js', 'scatter.js',
         'sessions.js', 'replay.js', 'compare.js', 'main.js']

CDN_IMPORT = ('import { FilesetResolver, PoseLandmarker } from\n'
              '  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";\n')

parts = [CDN_IMPORT]
for name in ORDER:
    src = (APP / 'js' / name).read_text()
    # drop local imports (including multi-line ones) and the CDN import
    src = re.sub(r"^import[^;]*?from\s*['\"][^'\"]*['\"];\s*$", '', src,
                 flags=re.M | re.S)
    src = re.sub(r'^export\s+', '', src, flags=re.M)
    parts.append(f'/* ================= {name} ================= */\n{src}')
    if name == 'sessions.js':
        parts.append('const db = { saveSession, deleteSession, listSessions, getSession };\n')

bundle = '\n'.join(parts)

html = (APP / 'index.html').read_text()
tag = '<script type="module" src="js/main.js"></script>'
assert tag in html, 'main.js script tag not found in index.html'
html = html.replace(tag, f'<script type="module">\n{bundle}</script>')

out = APP / 'dist'
out.mkdir(exist_ok=True)
(out / 'index.html').write_text(html)
# repo-root copy is what GitHub Pages serves
(APP.parent / 'index.html').write_text(html)
print(f'wrote {out / "index.html"} and {APP.parent / "index.html"} ({len(html) // 1024} kB)')
