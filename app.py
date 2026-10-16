import base64
import html
import io
import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(page_title="Manga page reader", page_icon="📖", layout="wide")

# ---- where the data lives (override with environment variables if needed) ----
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "dataset"))                       # holds sequences.json + page images
PRED_FILE = Path(os.environ.get("PRED_FILE", "predictions/test_predictions.jsonl"))  # the model's test output

SPEAKER_COLORS = ["#35C2D6", "#E5487F", "#F2B138", "#8B6CF0", "#4CC38A", "#FF8A5B", "#5B9BFF", "#C7D94B"]
UNKNOWN_COLOR = "#9AA0B8"

# held-out scores (10 development sequences, task's score.py); filled from the training runs
SCORES = [
    {"Model": "Qwen2.5-VL-7B, no tuning", "Text recovery": 0.404, "Speaker + text": 0.157},
    {"Model": "Qwen2.5-VL-7B + LoRA (v1)", "Text recovery": 0.530, "Speaker + text": 0.250},
    {"Model": "Qwen2.5-VL-7B + LoRA, higher resolution (v2)", "Text recovery": 0.562, "Speaker + text": 0.250},
    {"Model": "Qwen3-VL-8B, no tuning (returned no text on 8 of 10)", "Text recovery": 0.139, "Speaker + text": 0.038},
    {"Model": "Qwen3-VL-8B + LoRA (v3)", "Text recovery": 0.816, "Speaker + text": 0.463},
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Zen+Kaku+Gothic+New:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Zen Kaku Gothic New', system-ui, sans-serif; }
.stApp {
  background-color: #14152B;
  background-image: radial-gradient(rgba(237,239,247,.07) 1px, transparent 1.3px);
  background-size: 14px 14px;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
.mr-title { font-family: 'Dela Gothic One', 'Zen Kaku Gothic New', sans-serif; font-size: clamp(2rem, 4.4vw, 3.4rem);
            line-height: 1.05; margin: 0 0 .5rem; color: #EDEFF7; letter-spacing: .01em; }
.mr-sub { color: #B9BDD3; max-width: 60ch; margin: 0 0 1.4rem; font-size: 1.05rem; line-height: 1.55; }
.mr-row { display: flex; margin: 0 0 20px; }
.mr-row.right { justify-content: flex-end; }
.mr-bubble { --c: #35C2D6; max-width: 84%; background: #F4F6FA; color: #14152B; border: 3px solid var(--c);
             border-radius: 20px; padding: 11px 16px 12px; position: relative; font-size: 1.02rem; line-height: 1.5; }
.mr-bubble::after { content: ""; position: absolute; bottom: -10px; left: 24px; width: 15px; height: 15px; background: #F4F6FA;
                    border-right: 3px solid var(--c); border-bottom: 3px solid var(--c); transform: rotate(45deg); }
.mr-row.right .mr-bubble::after { left: auto; right: 24px; }
.mr-who { display: inline-block; background: var(--c); color: #14152B; font-weight: 700; font-size: .78rem;
          padding: 1px 9px; border-radius: 999px; margin-bottom: 5px; }
.mr-caption { width: 100%; border: 2px solid #EDEFF7; color: #EDEFF7; padding: 10px 14px; margin: 0 0 20px;
              box-shadow: 5px 5px 0 rgba(237,239,247,.18); font-size: 1.0rem; line-height: 1.5; }
.mr-caption .mr-who { background: #EDEFF7; }
.mr-empty { border: 2px dashed rgba(237,239,247,.35); color: #B9BDD3; padding: 22px; border-radius: 14px; line-height: 1.55; }
.mr-meta { color: #B9BDD3; font-size: .92rem; margin: 0 0 .8rem; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
"""

SCENE_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%; background: transparent; overflow: hidden;
               font-family: 'Zen Kaku Gothic New', system-ui, sans-serif; color: #EDEFF7; }
  #stage { position: absolute; inset: 0; }
  #stage canvas { display: block; }
  #label { position: absolute; left: 14px; bottom: 12px; font-size: 14px; color: #B9BDD3; pointer-events: none; }
  #flat { display: flex; gap: 10px; padding: 10px; height: 100%; box-sizing: border-box; align-items: center; justify-content: center; }
  #flat img { max-height: 92%; max-width: 32%; border: 2px solid #EDEFF7; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body>
<div id="stage"></div><div id="label"></div>
<script>
const pages = __PAGES__;
const focus = __FOCUS__;
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const stage = document.getElementById('stage');
document.getElementById('label').textContent = 'Page ' + (focus + 1) + ' of ' + pages.length + '  (page 1 sits on the right, as in manga)';

function flatFallback() {
  stage.innerHTML = '<div id="flat">' + pages.map(p => p ? '<img src="' + p.uri + '">' : '').join('') + '</div>';
}

try {
  if (typeof THREE === 'undefined') throw new Error('three.js not loaded');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0, 13);
  const group = new THREE.Group();
  scene.add(group);

  const H = 4.4;
  const aniso = renderer.capabilities.getMaxAnisotropy();

  function placeholder(n) {
    const c = document.createElement('canvas'); c.width = 512; c.height = 760;
    const g = c.getContext('2d');
    g.fillStyle = '#F4F6FA'; g.fillRect(0, 0, 512, 760);
    g.strokeStyle = '#14152B'; g.lineWidth = 6; g.strokeRect(24, 24, 464, 712);
    g.fillStyle = '#14152B'; g.font = 'bold 54px sans-serif'; g.textAlign = 'center';
    g.fillText('Page ' + n, 256, 400);
    const t = new THREE.CanvasTexture(c); t.encoding = THREE.sRGBEncoding; return t;
  }

  const meshes = pages.map((p, i) => {
    const aspect = p ? p.w / p.h : 0.68;
    const geo = new THREE.BoxGeometry(H * aspect, H, 0.05);
    const edge = new THREE.MeshBasicMaterial({ color: 0xdfe3ee });
    const face = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const back = new THREE.MeshBasicMaterial({ color: 0xe8ebf3 });
    if (p) {
      new THREE.TextureLoader().load(p.uri, tex => {
        tex.encoding = THREE.sRGBEncoding; tex.anisotropy = aniso;
        face.map = tex; face.needsUpdate = true;
      });
    } else {
      face.map = placeholder(i + 1); face.needsUpdate = true;
    }
    const mesh = new THREE.Mesh(geo, [edge, edge, edge, edge, face, back]);
    // start as one tight stack; the fan-out on load is the only unprompted motion
    mesh.position.set(0, 0, -1 - i * 0.05);
    mesh.scale.setScalar(0.8);
    group.add(mesh);
    return mesh;
  });

  let spacing = 2.9;
  function targetFor(i) {
    const off = focus - i;                       // earlier pages sit to the right
    const near = off === 0;
    return {
      x: off * spacing,
      z: near ? 1.0 : -Math.abs(off) * 1.2,
      ry: -off * 0.42,
      s: near ? 1.1 : 0.8,
    };
  }

  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    spacing = Math.min(2.9, camera.aspect * 2.3);
  }
  window.addEventListener('resize', resize); resize();

  let px = 0, py = 0;
  window.addEventListener('pointermove', e => {
    const r = stage.getBoundingClientRect();
    px = ((e.clientX - r.left) / r.width - 0.5) * 2;
    py = ((e.clientY - r.top) / r.height - 0.5) * 2;
  });

  const k = reduce ? 1 : 0.075;
  const clock = new THREE.Clock();
  (function tick() {
    const t = clock.getElapsedTime();
    meshes.forEach((m, i) => {
      const tg = targetFor(i);
      m.position.x += (tg.x - m.position.x) * k;
      m.position.z += (tg.z - m.position.z) * k;
      m.rotation.y += (tg.ry - m.rotation.y) * k;
      const s = m.scale.x + (tg.s - m.scale.x) * k; m.scale.setScalar(s);
    });
    const ty = reduce ? 0 : px * 0.28, tx = reduce ? 0 : py * 0.12;
    group.rotation.y += (ty - group.rotation.y) * 0.06;
    group.rotation.x += (tx - group.rotation.x) * 0.06;
    group.position.y = reduce ? 0 : Math.sin(t * 0.9) * 0.05;
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  })();
} catch (err) {
  flatFallback();
}
</script></body></html>
"""


@st.cache_data
def load_sequences():
    f = DATASET_DIR / "sequences.json"
    if not f.exists():
        return None
    entries = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(entries, dict):
        entries = entries.get("sequences", list(entries.values()))
    out = {}
    for e in entries:
        key = next((k for k in ("images", "image_paths", "paths") if k in e), None)
        if key:
            out[e["sequence_id"]] = [p if os.path.isabs(p) else str(DATASET_DIR / p) for p in e[key]]
    return out


@st.cache_data
def load_predictions():
    if not PRED_FILE.exists():
        return None
    rows = [json.loads(l) for l in PRED_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["sequence_id"]: r["pages"] for r in rows}


@st.cache_data(show_spinner=False)
def page_payload(path, max_side=1100):
    p = Path(path)
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return {"uri": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "w": im.width, "h": im.height}


def speaker_colors(pages):
    """One colour per speaker, assigned by first appearance across all three pages."""
    colors, side, n = {}, {}, 0
    for page in pages:
        for line in page:
            s = line.get("speaker", "UNKNOWN")
            if s in colors or s == "NARRATION":
                continue
            if s == "UNKNOWN":
                colors[s], side[s] = UNKNOWN_COLOR, n % 2
            else:
                colors[s], side[s] = SPEAKER_COLORS[n % len(SPEAKER_COLORS)], n % 2
            n += 1
    return colors, side


def render_page(lines, colors, side):
    if not lines:
        return ('<div class="mr-empty">No story text on this page. The model found only art or sound effects here.</div>')
    out = []
    for line in lines:
        who = html.escape(str(line.get("speaker", "UNKNOWN")))
        text = html.escape(str(line.get("text", "")))
        if line.get("speaker") == "NARRATION":
            out.append(f'<div class="mr-caption"><span class="mr-who">Narration</span><br>{text}</div>')
        else:
            c = colors.get(line.get("speaker", "UNKNOWN"), UNKNOWN_COLOR)
            cls = "right" if side.get(line.get("speaker", "UNKNOWN"), 0) else ""
            out.append(f'<div class="mr-row {cls}"><div class="mr-bubble" style="--c:{c}">'
                       f'<span class="mr-who">{who}</span><br>{text}</div></div>')
    return "".join(out)


st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<h1 class="mr-title">Manga page reader</h1>'
            '<p class="mr-sub">Pick a sequence. The model reads three consecutive pages and returns who said what, '
            'in reading order, with the same label for the same character on every page.</p>',
            unsafe_allow_html=True)

seqs, preds = load_sequences(), load_predictions()
if seqs is None or preds is None:
    st.error("Data not found. Place these next to app.py, then reload:\n\n"
             f"- `{DATASET_DIR}/sequences.json` and the page images it points to\n"
             f"- `{PRED_FILE}` (your test_predictions.jsonl)")
    st.stop()

ids = [i for i in preds if i in seqs] or list(preds)
with st.sidebar:
    st.markdown("### Sequence")
    idx = st.selectbox("Choose a sequence", range(len(ids)),
                       format_func=lambda i: f"Sequence {i + 1}  ({ids[i][-6:]})", label_visibility="collapsed")
    st.download_button("Download all predictions", PRED_FILE.read_bytes(), file_name="test_predictions.jsonl")
    with st.expander("How well does it score?"):
        st.table(SCORES)
        st.caption("Scored on 10 development sequences the model never trained on, using the task's score.py. "
                   "Ten sequences is a small sample, so treat the numbers as rough.")

sid = ids[idx]
pages = preds[sid]
payloads = [page_payload(p) for p in seqs.get(sid, [])][:3]
payloads += [None] * (3 - len(payloads))
if not any(payloads):
    st.sidebar.info("Page images were not found, so the 3D view shows blank pages. The transcript still works.")

page_no = st.radio("Page", [1, 2, 3], horizontal=True, format_func=lambda n: f"Page {n}", label_visibility="collapsed")
colors, side = speaker_colors(pages)

left, right = st.columns([1.15, 1], gap="large")
with left:
    scene = SCENE_HTML.replace("__PAGES__", json.dumps(payloads)).replace("__FOCUS__", str(page_no - 1))
    if hasattr(st, "iframe"):            # newer Streamlit
        st.iframe(scene, height=560)
    else:                                # older Streamlit
        components.html(scene, height=560)
with right:
    lines = pages[page_no - 1] if page_no - 1 < len(pages) else []
    n_speakers = len({l.get("speaker") for l in lines if l.get("speaker") != "NARRATION"})
    st.markdown(f'<p class="mr-meta">Page {page_no}: {len(lines)} lines, {n_speakers} speakers</p>', unsafe_allow_html=True)
    with st.container(height=500, border=False):
        st.markdown(render_page(lines, colors, side), unsafe_allow_html=True)
