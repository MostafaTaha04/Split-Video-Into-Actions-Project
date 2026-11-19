"""
viewer.py
=========
A video player with the cut timeline underneath it.

Open ``viewer.html`` in a browser: the video plays at the top; under it a
timeline shows every step as a coloured block, with a line at each cut and a
playhead that moves while the video plays. Click a block (or a step card below)
to jump to that step; the card for the step on screen is highlighted. Each step
is also saved as its own clip.

    python viewer.py --video data/videos/pc_assembly_01.mp4 \
        --segments data/annotations/pc_assembly_01.json

    # optional: show the annotated steps as a second lane for comparison
    python viewer.py --video ... --segments ... --ground-truth data/annotations/pc_assembly_01.json

Writes into outputs/<video name>/ (or into --out-dir):
    viewer.html        the player page (open it in a browser)
    preview.mp4        a browser-friendly H.264 copy of the video (<= 960 px);
                       made once and reused on later runs (it depends only on
                       the video); --fresh makes it again
    clips/step_NN.mp4  one H.264 clip per step, cut again on every run

The page and the clips are rebuilt from the segments file on every run, so a
change to that file always shows up.

Browsers cannot play every video format (OpenCV's default MPEG-4 output, for
example, plays nowhere), so the preview and the clips are re-encoded to H.264
with ffmpeg. ffmpeg comes from the system PATH or, if absent, from the
``imageio-ffmpeg`` package (``pip install imageio-ffmpeg``), which ships its
own copy on Windows, macOS and Linux. Without ffmpeg the page still works if
the original video is H.264, playing it directly; clips are then skipped.

The segments file is a JSON list of ``{"start": s, "end": s, "label": "..."}``
(or an object holding that list under "segments"). Steps labelled with one of
the activities in labels.py get that activity's colour, and their clips are
named after it (``step_03_SCREW.mp4``).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil
import subprocess
from typing import List, Optional

PREVIEW_MAX_SIDE = 960
PREVIEW_MAX_FPS = 30

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756",
           "#72B7B2", "#EECA3B", "#9D755D", "#FF9DA6", "#79706E"]

try:  # the project's activity colours, when run inside the project
    from labels import LABEL_COLORS
except ImportError:  # pragma: no cover
    LABEL_COLORS = {}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_segments(path: str) -> List[dict]:
    """[{start, end, label, direction?}] from either supported format."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("segments", data.get("steps", [])) if isinstance(data, dict) else data
    out = []
    for s in raw:
        start = float(s.get("start_time", s.get("start", 0.0)))
        end = float(s.get("end_time", s.get("end", start)))
        label = s.get("label") or s.get("activity") or s.get("activity_description") or ""
        out.append({"start": round(start, 3), "end": round(end, 3), "label": str(label),
                    "direction": s.get("direction")})
    out.sort(key=lambda s: s["start"])
    return [s for s in out if s["end"] > s["start"]]


def video_info(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    info = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": n / fps if fps else 0.0,
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC))}
    cap.release()
    return info


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _is_h264(info: dict) -> bool:
    code = "".join(chr((info["fourcc"] >> (8 * i)) & 0xFF) for i in range(4)).lower()
    return code in ("avc1", "h264", "x264")


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def _scale_filter(info: dict) -> str:
    w, h = info["width"], info["height"]
    long_side = max(w, h)
    if long_side <= PREVIEW_MAX_SIDE:
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if w >= h:
        return f"scale={PREVIEW_MAX_SIDE}:-2"
    return f"scale=-2:{PREVIEW_MAX_SIDE}"


def make_preview(ffmpeg: str, video: str, out: str, info: dict,
                 until: Optional[float] = None) -> None:
    # -hwaccel auto: use the graphics card to decode the 4K video when it can (falls back to the CPU);
    # -stats: print a progress line while it works.
    cmd = [ffmpeg, "-y", "-v", "error", "-stats", "-hwaccel", "auto", "-i", video]
    if until:
        cmd += ["-t", f"{until + 0.5:.3f}"]
    vf = _scale_filter(info)
    if info["fps"] > PREVIEW_MAX_FPS + 1:
        vf += f",fps={PREVIEW_MAX_FPS}"
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


def remove_file(path: str, tries: int = 10) -> None:
    """Delete a file. On Windows another program (OneDrive syncing, File Explorer, a video
    player) can hold a file for a moment, so retry for a few seconds before giving up."""
    import stat
    import time
    for i in range(tries):
        try:
            os.chmod(path, stat.S_IWRITE)   # clear a read-only flag, if any
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if i == tries - 1:
                raise PermissionError(
                    f"Windows will not let this file be deleted: {path}\n"
                    "It is probably open somewhere: close the player page or clip in the browser, "
                    "any video player, and File Explorer windows showing that folder, then run again.")
            time.sleep(0.5)


def clear_folder(folder: str) -> None:
    """Delete every file under folder. The folders themselves are kept: Windows and OneDrive
    often refuse to remove a folder that was just emptied, and an empty folder does no harm."""
    for root, _dirs, files in os.walk(folder):
        for f in files:
            remove_file(os.path.join(root, f))


def cut_clips(ffmpeg: str, preview: str, segments: List[dict], clips_dir: str) -> List[str]:
    os.makedirs(clips_dir, exist_ok=True)
    clear_folder(clips_dir)   # clips of an earlier run, e.g. of a step that no longer exists
    names = []
    width = max(2, len(str(len(segments))))
    for i, s in enumerate(segments, 1):
        tag = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in s["label"])[:40]
        name = f"step_{i:0{width}d}" + (f"_{tag}" if tag else "") + ".mp4"
        subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", f"{s['start']:.3f}", "-i", preview,
                        "-t", f"{s['end'] - s['start']:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                        "-movflags", "+faststart", os.path.join(clips_dir, name)], check=True)
        names.append(name)
    return names


def thumbnails(video: str, segments: List[dict], width: int = 192) -> List[str]:
    """One small JPEG (data URI) per step, from the middle of the step."""
    import cv2
    cap = cv2.VideoCapture(video)
    out = []
    for s in segments:
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * (s["start"] + s["end"]) / 2)
        ok, frame = cap.read()
        if not ok:
            out.append("")
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else "")
    cap.release()
    return out


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def colour_map(segments: List[dict]) -> dict:
    """The project's colour for each known activity; otherwise one colour per
    distinct label (in order of first appearance). With no labels at all,
    steps alternate between two blues so neighbours differ."""
    labels = [s["label"] for s in segments]
    distinct = list(dict.fromkeys(l for l in labels if l))
    if distinct and all(l in LABEL_COLORS for l in distinct):
        return {l: LABEL_COLORS[l] for l in distinct}
    if len(distinct) >= 2:
        return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(distinct)}
    return {}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
 :root { --ink:#1f2328; --muted:#646c76; --line:#d8dde3; --bg:#f6f7f9; --card:#fff; --accent:#0b62d6; }
 * { box-sizing: border-box; }
 body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); background:var(--bg); }
 main { max-width: 1100px; margin: 0 auto; padding: 20px 20px 48px; }
 h1 { font-size: 20px; margin: 0 0 2px; }
 .sub { color: var(--muted); margin-bottom: 14px; }
 .player { background:#000; border-radius:10px; overflow:hidden; display:flex; justify-content:center; }
 video { display:block; max-width:100%; max-height:62vh; }
 .panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px 10px; margin-top:12px; }
 .now { display:flex; align-items:center; gap:10px; min-height:28px; flex-wrap:wrap; }
 .now .badge { font-weight:600; padding:2px 9px; border-radius:999px; color:#fff; }
 .now .label { font-weight:600; }
 .now .time { color:var(--muted); font-variant-numeric: tabular-nums; }
 .controls { margin-left:auto; display:flex; gap:6px; }
 button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:7px; padding:3px 10px; cursor:pointer; }
 button:hover { border-color:#aab3bd; } button[aria-pressed="true"] { background:#e8f0fe; border-color:var(--accent); color:var(--accent); }
 .lane-name { font-size:12px; color:var(--muted); margin:8px 0 3px; }
 .lane { position:relative; height:46px; border-radius:6px; background:#eef1f4; cursor:pointer; user-select:none; }
 .lane.gt { height:22px; }
 .seg { position:absolute; top:0; bottom:0; border-right:2px solid #fff; overflow:hidden;
        color:#fff; font-weight:600; font-size:12px; display:flex; align-items:center; justify-content:center;
        opacity:.78; transition:opacity .12s; white-space:nowrap; }
 .seg span { padding:0 4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100%; }
 .seg:hover { opacity:.95; } .seg.active { opacity:1; box-shadow: inset 0 0 0 2px rgba(0,0,0,.35); }
 .lane.gt .seg { opacity:.55; font-weight:500; font-size:10px; }
 .playhead { position:absolute; top:-4px; bottom:-4px; width:2px; background:#111; pointer-events:none; }
 .playhead::before { content:""; position:absolute; top:-5px; left:-5px; border:6px solid transparent; border-top-color:#111; }
 .axis { position:relative; height:18px; font-size:11px; color:var(--muted); font-variant-numeric: tabular-nums; }
 .axis span { position:absolute; transform:translateX(-50%); top:2px; }
 .axis span::before { content:""; position:absolute; left:50%; top:-4px; height:4px; border-left:1px solid #b9c0c8; }
 .tip { position:fixed; pointer-events:none; background:#1f2328; color:#fff; font-size:12px; padding:5px 8px; border-radius:6px; display:none; z-index:9; white-space:nowrap; }
 .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px; font-size:12px; color:var(--muted); }
 .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
 h2 { font-size:16px; margin:22px 0 8px; }
 .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap:10px; }
 .card { background:var(--card); border:1px solid var(--line); border-radius:9px; overflow:hidden; cursor:pointer; transition: box-shadow .12s, border-color .12s; }
 .card:hover { border-color:#aab3bd; } .card.active { border-color:var(--accent); box-shadow:0 0 0 2px rgba(11,98,214,.25); }
 .card .thumb { aspect-ratio: 16/10; background:#111 center/contain no-repeat; border-top:4px solid var(--c); }
 .card .body { padding:7px 9px 9px; }
 .card .t { font-weight:600; } .card .m { color:var(--muted); font-size:12px; font-variant-numeric: tabular-nums; }
 .card a { font-size:12px; color:var(--accent); text-decoration:none; } .card a:hover { text-decoration:underline; }
 .topnav { margin-bottom:8px; } .topnav a { color:var(--accent); text-decoration:none; font-size:13px; } .topnav a:hover { text-decoration:underline; }
 .keys { color:var(--muted); font-size:12px; margin-top:6px; }
 kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; font-size:11px; background:#fff; }
</style></head>
<body><main>
__TOPNAV__<h1>__TITLE__</h1>
<div class="sub">__SUB__</div>
<div class="player"><video id="v" src="__SRC__" controls preload="metadata" playsinline></video></div>
<div class="panel">
  <div class="now"><span class="badge" id="nb">Step –</span><span class="label" id="nl"></span><span class="time" id="nt"></span>
    <span class="controls"><button id="prev" title="Previous step (←)">◀ Prev</button><button id="loop" aria-pressed="false" title="Repeat the current step (L)">Loop step</button><button id="next" title="Next step (→)">Next ▶</button></span></div>
  <div class="lane-name">__LANE__ · each line is a cut</div>
  <div class="lane" id="lane"></div>
  __GTLANE__
  <div class="axis" id="axis"></div>
  <div class="legend" id="legend"></div>
  <div class="keys"><kbd>Space</kbd> play/pause · <kbd>←</kbd>/<kbd>→</kbd> previous/next step · <kbd>L</kbd> loop step · click the timeline to jump</div>
</div>
<h2>Clips</h2>
<div class="grid" id="grid"></div>
<div class="tip" id="tip"></div>
</main>
<script>
const SEG = __SEGMENTS__;
const GT = __GT__;
const COLORS = __COLORS__;
const PALETTE = __PALETTE__;
const DURATION = __DURATION__;
const v = document.getElementById('v');
const lane = document.getElementById('lane'), axis = document.getElementById('axis');
const tip = document.getElementById('tip'), grid = document.getElementById('grid');
let loop = false, current = -1;
const total = () => (isFinite(v.duration) && v.duration > 0) ? v.duration : DURATION;
const fmt = t => { t = Math.max(0, t); const m = Math.floor(t / 60), s = t - 60 * m; return m + ':' + s.toFixed(1).padStart(4, '0'); };
const colour = (s, i) => COLORS[s.label] || PALETTE[i % 2 ? 1 : 0];
function textOn(hex) { const n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) > 165 ? '#1f2328' : '#fff'; }
function name(s, i) { return s.label ? s.label : 'Step ' + (i + 1); }

function drawLane(el, list, isGt) {
  el.querySelectorAll('.seg, .playhead').forEach(n => n.remove());
  const T = total();
  list.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'seg'; d.style.left = (100 * s.start / T) + '%'; d.style.width = (100 * (s.end - s.start) / T) + '%';
    d.style.background = isGt ? '#6b7680' : colour(s, i);
    if (!isGt) d.style.color = textOn(colour(s, i));
    const txt = document.createElement('span'); txt.textContent = isGt ? (s.label || '') : (s.label ? s.label : (i + 1)); d.appendChild(txt);
    d.dataset.i = i;
    d.addEventListener('mousemove', e => { tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
      tip.textContent = (isGt ? 'Annotated: ' : 'Step ' + (i + 1) + ': ') + (s.label || '') + '  ' + fmt(s.start) + '–' + fmt(s.end) + ' (' + (s.end - s.start).toFixed(1) + ' s)' + (s.direction && s.direction !== 'unclear' && !isGt ? ' · ' + s.direction : ''); });
    d.addEventListener('mouseleave', () => tip.style.display = 'none');
    el.appendChild(d);
  });
  if (!isGt) { const ph = document.createElement('div'); ph.className = 'playhead'; ph.id = 'ph'; el.appendChild(ph); }
}
function drawAxis() {
  axis.innerHTML = ''; const T = total();
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]; const step = steps.find(x => T / x <= 12) || 600;
  for (let t = 0; t <= T + 1e-6; t += step) { const s = document.createElement('span'); s.style.left = (100 * t / T) + '%'; s.textContent = fmt(t).replace(/\.0$/, ''); axis.appendChild(s); }
}
function laneClick(e) { const r = lane.getBoundingClientRect(); const t = total() * (e.clientX - r.left) / r.width; v.currentTime = Math.max(0, Math.min(total(), t)); }
lane.addEventListener('click', laneClick);
const gtEl = document.getElementById('gtlane'); if (gtEl) gtEl.addEventListener('click', e => { const r = gtEl.getBoundingClientRect(); v.currentTime = total() * (e.clientX - r.left) / r.width; });

function buildCards() {
  SEG.forEach((s, i) => {
    const c = document.createElement('div'); c.className = 'card'; c.dataset.i = i; c.style.setProperty('--c', colour(s, i));
    c.innerHTML = '<div class="thumb" style="background-image:url(' + (s.thumb || '') + ')"></div><div class="body"><div class="t">Step ' + (i + 1) + (s.label ? ' · ' + s.label.replace(/</g, '&lt;') : '') + '</div>' +
      '<div class="m">' + fmt(s.start) + ' – ' + fmt(s.end) + ' · ' + (s.end - s.start).toFixed(1) + ' s' + (s.direction && s.direction !== 'unclear' ? ' · ' + s.direction : '') + '</div>' +
      (s.clip ? '<a href="' + s.clip + '" target="_blank" onclick="event.stopPropagation()">Open clip file</a>' : '') + '</div>';
    c.addEventListener('click', () => playStep(i));
    grid.appendChild(c);
  });
}
function idxAt(t) { for (let i = 0; i < SEG.length; i++) if (t >= SEG[i].start && t < SEG[i].end) return i; return t >= (SEG.at(-1)?.end ?? 0) ? SEG.length - 1 : -1; }
function playStep(i) { if (i < 0 || i >= SEG.length) return; v.currentTime = SEG[i].start + 0.01; current = i; v.play(); update(); }
function update() {
  const t = v.currentTime, T = total();
  const ph = document.getElementById('ph'); if (ph) ph.style.left = (100 * t / T) + '%';
  let i = idxAt(t);
  if (loop && current >= 0 && t >= SEG[current].end - 0.03) { v.currentTime = SEG[current].start; i = current; }
  if (i !== current && !(loop && current >= 0)) current = i;
  lane.querySelectorAll('.seg').forEach(n => n.classList.toggle('active', +n.dataset.i === current));
  grid.querySelectorAll('.card').forEach(n => n.classList.toggle('active', +n.dataset.i === current));
  const nb = document.getElementById('nb'), nl = document.getElementById('nl'), nt = document.getElementById('nt');
  if (current >= 0) { const s = SEG[current]; nb.textContent = 'Step ' + (current + 1) + ' of ' + SEG.length; nb.style.background = colour(s, current);
    nl.textContent = s.label || ''; nt.textContent = fmt(t) + '  ·  step ' + fmt(s.start) + '–' + fmt(s.end); }
  else { nb.textContent = 'Before step 1'; nb.style.background = '#8a939c'; nl.textContent = ''; nt.textContent = fmt(t); }
}
function tick() { update(); if (!v.paused) requestAnimationFrame(tick); }
v.addEventListener('play', () => requestAnimationFrame(tick));
v.addEventListener('seeked', update); v.addEventListener('timeupdate', update);
v.addEventListener('loadedmetadata', () => { drawLane(lane, SEG, false); if (gtEl) drawLane(gtEl, GT, true); drawAxis(); update(); });
document.getElementById('prev').onclick = () => playStep(Math.max(0, (current < 0 ? 0 : current - 1)));
document.getElementById('next').onclick = () => playStep(Math.min(SEG.length - 1, current + 1));
const lb = document.getElementById('loop'); lb.onclick = () => { loop = !loop; lb.setAttribute('aria-pressed', loop); if (loop && current < 0) current = idxAt(v.currentTime); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') { e.preventDefault(); document.getElementById('next').click(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); document.getElementById('prev').click(); }
  else if (e.key === ' ') { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === 'l' || e.key === 'L') lb.click();
});
const legend = document.getElementById('legend');
Object.entries(COLORS).forEach(([k, c]) => { const s = document.createElement('span'); s.innerHTML = '<i style="background:' + c + '"></i>'; s.appendChild(document.createTextNode(k)); legend.appendChild(s); });
buildCards(); drawLane(lane, SEG, false); if (gtEl) drawLane(gtEl, GT, true); drawAxis(); update();
</script></body></html>
"""


def build_page(segments: List[dict], src: str, title: str, subtitle: str, duration: float,
               gt: Optional[List[dict]] = None, lane_title: str = "Detected activities",
               back_href: Optional[str] = None) -> str:
    colours = colour_map(segments)
    gt_lane = ('<div class="lane-name">Annotated steps (ground truth)</div><div class="lane gt" id="gtlane"></div>'
               if gt else "")
    rep = {
        "__TITLE__": html.escape(title), "__SUB__": html.escape(subtitle), "__SRC__": html.escape(src),
        "__SEGMENTS__": json.dumps(segments), "__GT__": json.dumps(gt or []),
        "__COLORS__": json.dumps(colours), "__PALETTE__": json.dumps(["#4C78A8", "#72A0C8"]),
        "__DURATION__": f"{duration:.3f}", "__GTLANE__": gt_lane,
        "__LANE__": html.escape(lane_title),
        "__TOPNAV__": (f'<div class="topnav"><a href="{html.escape(back_href)}">&larr; All videos</a></div>'
                       if back_href else ""),
    }
    page = PAGE
    for k, v in rep.items():
        page = page.replace(k, v)
    return page


def open_in_browser(path: str) -> None:
    """Open a local page in the default web browser (works with non-English folder names)."""
    import pathlib
    import webbrowser
    webbrowser.open(pathlib.Path(path).resolve().as_uri())


def build_viewer(video: str, segments_path: str, out_dir: Optional[str] = None,
                 ground_truth: Optional[str] = None, make_clips: bool = True,
                 lane_title: str = "Detected activities", back_href: Optional[str] = None,
                 fresh: bool = False) -> str:
    """Build the player. fresh=True deletes everything made for this video before (preview
    included) and makes it all again from the original video."""
    if not out_dir:
        stem = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", stem)
    if fresh and os.path.isdir(out_dir):
        print("Starting fresh: deleting the earlier output in", out_dir)
        clear_folder(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    segments = load_segments(segments_path)
    if not segments:
        raise ValueError(f"no steps in {segments_path}")
    info = video_info(video)
    until = max(info["duration"], segments[-1]["end"])
    ffmpeg = find_ffmpeg()

    if ffmpeg:
        preview = os.path.join(out_dir, "preview.mp4")
        if os.path.exists(preview) and os.path.getmtime(preview) > os.path.getmtime(video):
            print("Reusing the browser copy of the video made earlier (add --fresh to make it again).")
            print("Timeline and clips are rebuilt from", os.path.basename(segments_path))
        else:
            print(f"Making a browser-friendly copy of the video ({int(until // 60)}:{int(until % 60):02d} long). "
                  "For a 4K video this takes a few minutes; later runs reuse it...")
            make_preview(ffmpeg, video, preview + ".part.mp4", info)
            os.replace(preview + ".part.mp4", preview)
        src, play_file = "preview.mp4", preview
        if make_clips:
            names = cut_clips(ffmpeg, preview, segments, os.path.join(out_dir, "clips"))
            for s, n in zip(segments, names):
                s["clip"] = "clips/" + n
    else:
        if not _is_h264(info):
            print("WARNING: ffmpeg not found and the video is not H.264, so browsers may not play it. "
                  "Install it with:  pip install imageio-ffmpeg")
        src = os.path.relpath(os.path.abspath(video), os.path.abspath(out_dir)).replace(os.sep, "/")
        play_file = video
        print("NOTE: ffmpeg not found - playing the original video; step clips not written.\n"
              "      A 4K video may not play in the browser. Fix it with:  pip install -r requirements.txt")

    for s, t in zip(segments, thumbnails(play_file, segments)):
        s["thumb"] = t

    gt = load_segments(ground_truth) if ground_truth else None
    n = len(segments)
    title = os.path.basename(video)
    subtitle = (f"{n} steps · {n - 1} cuts · {until:.1f} s"
                + (f" · compared with {os.path.basename(ground_truth)}" if gt else ""))
    page = build_page(segments, src, title, subtitle, until, gt, lane_title, back_href)
    out = os.path.join(out_dir, "viewer.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out


def main():
    ap = argparse.ArgumentParser(description="Video player with the cut timeline underneath.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments", required=True, help="JSON list of {start, end, label} steps")
    ap.add_argument("--ground-truth", help="the annotated steps (same format) to show as a second lane")
    ap.add_argument("--out-dir", help="where to write viewer.html (default: outputs/<video name>/)")
    ap.add_argument("--no-clips", action="store_true", help="do not write one clip per step")
    ap.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
    ap.add_argument("--fresh", action="store_true",
                    help="delete this video's earlier output and make everything again, preview included")
    a = ap.parse_args()
    path = build_viewer(a.video, a.segments, a.out_dir, a.ground_truth, not a.no_clips, fresh=a.fresh)
    print(f"Player page: {path}")
    if not a.no_open:
        open_in_browser(path)


if __name__ == "__main__":
    main()
