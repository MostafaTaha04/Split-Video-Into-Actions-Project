"""
show_all.py
===========
One command to see every video with its timeline and its clips.

    python show_all.py                       # every video that has an annotation file
    python show_all.py pc_assembly_02        # only the videos you name
    python show_all.py --no-clips            # faster: skip the per-step clips
    python show_all.py --fresh               # delete everything made before and redo it all

For each video in data/videos/ that has an annotation file in
data/annotations/ with the same name, this builds the player page with the
timeline under the video and one clip per step (see viewer.py). It writes them
into outputs/<video name>/. It then writes outputs/index.html, a home page
that lists all the videos, each with its own timeline. That page opens in the
browser; click a video to open its player.

The first run makes a browser-friendly copy of each 4K video, which takes a few
minutes per video. Later runs reuse those copies, because they depend only on
the video. The timelines, pages and clips are rebuilt from the annotation files
on every run, so after you edit an annotation file, running this again shows the
change and only takes seconds. --fresh deletes the outputs and makes everything
again from the original videos (a few minutes per video), e.g. for a demo.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import sys
import traceback

from labels import LABEL_COLORS
from viewer import build_viewer, load_segments, open_in_browser, video_info

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(ROOT, "data", "videos")
ANNOTATIONS = os.path.join(ROOT, "data", "annotations")
OUTPUTS = os.path.join(ROOT, "outputs")


def poster(video: str, t: float, width: int = 320) -> str:
    """One frame of the video as a small JPEG data URI (for the home page)."""
    import cv2
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * t)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return ""
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def review_status(path: str) -> str:
    """'checked' or 'draft ...' from an annotation file's provenance: its review_status, or
    'checked' when the method says who checked it (e.g. "checked_by Mostafa Taha,Osama Najjar")."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ""
    prov = data.get("provenance", {}) if isinstance(data, dict) else {}
    status = str(prov.get("review_status", "")).strip()
    if not status and "checked" in str(prov.get("method", "")).lower():
        status = "checked"
    return status


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>All videos</title>
<style>
 :root { --ink:#1f2328; --muted:#646c76; --line:#d8dde3; --bg:#f6f7f9; --card:#fff; --accent:#0b62d6; }
 * { box-sizing:border-box; }
 body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); background:var(--bg); }
 main { max-width:1100px; margin:0 auto; padding:20px 20px 48px; }
 h1 { font-size:20px; margin:0 0 2px; } .sub { color:var(--muted); margin-bottom:14px; }
 .legend { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:12px; color:var(--muted); margin-bottom:14px; }
 .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
 a.row { display:flex; gap:14px; align-items:center; background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:10px; margin-bottom:10px; color:inherit; text-decoration:none; transition:border-color .12s, box-shadow .12s; }
 a.row:hover { border-color:var(--accent); box-shadow:0 0 0 2px rgba(11,98,214,.18); }
 .poster { width:200px; flex:none; aspect-ratio:16/9; border-radius:6px; background:#111 center/cover no-repeat; }
 .info { flex:1; min-width:0; }
 .name { font-weight:600; font-size:15px; } .meta { color:var(--muted); font-size:12px; margin:1px 0 7px; }
 .badge { display:inline-block; font-size:11px; padding:0 7px; border-radius:999px; margin-left:6px; vertical-align:1px; }
 .badge.draft { background:#fff4d6; color:#7a5200; } .badge.checked { background:#dcf3e3; color:#1e6b34; }
 .lane { position:relative; height:26px; border-radius:5px; background:#eef1f4; overflow:hidden; }
 .seg { position:absolute; top:0; bottom:0; border-right:1px solid #fff; }
 .open { color:var(--accent); font-size:12px; margin-top:6px; }
 .missing { color:var(--muted); font-size:13px; margin-top:18px; }
 .sub a { color:var(--accent); text-decoration:none; }
</style></head><body><main>
<h1>All videos</h1>
<div class="sub">__SUB__</div>
<div class="legend">__LEGEND__</div>
__ROWS__
__MISSING__
</main></body></html>
"""


def write_index(items: list, missing: list) -> str:
    rows = []
    used = []
    for it in items:
        T = max(it["duration"], 1e-6)
        segs = []
        for s in it["segments"]:
            c = LABEL_COLORS.get(s["label"], "#8a939c")
            if s["label"] not in used:
                used.append(s["label"])
            tip = f'{s["label"]}  {fmt(s["start"])}-{fmt(s["end"])}'
            segs.append(f'<div class="seg" title="{html.escape(tip)}" style="left:{100 * s["start"] / T:.3f}%;'
                        f'width:{100 * (s["end"] - s["start"]) / T:.3f}%;background:{c}"></div>')
        status = it["status"] or "draft"
        cls = "checked" if status.lower().startswith("checked") else "draft"
        rows.append(
            f'<a class="row" href="{html.escape(it["href"])}">'
            f'<div class="poster" style="background-image:url({it["poster"]})"></div>'
            f'<div class="info"><div class="name">{html.escape(it["name"])}'
            f'<span class="badge {cls}">{html.escape(status)}</span></div>'
            f'<div class="meta">{fmt(it["duration"])} · {len(it["segments"])} activities'
            f'{" · " + str(it["clips"]) + " clips" if it["clips"] else ""}</div>'
            f'<div class="lane">{"".join(segs)}</div><div class="open">Open player &rarr;</div></div></a>')
    legend = "".join(f'<span><i style="background:{LABEL_COLORS.get(l, "#8a939c")}"></i>{html.escape(l)}</span>'
                     for l in used)
    miss = ("<div class=\"missing\">No annotation file yet: " + ", ".join(html.escape(m) for m in missing) + "</div>"
            if missing else "")
    sub = f"{len(items)} videos · click one to open its player and clips"
    if os.path.exists(os.path.join(OUTPUTS, "results.html")):
        sub += ' · <a href="results.html">Model results &rarr;</a>'
    if os.path.exists(os.path.join(OUTPUTS, "final_results.html")):
        sub += ' · <a href="final_results.html">Final test &rarr;</a>'
    page = (INDEX.replace("__SUB__", sub)
            .replace("__LEGEND__", legend).replace("__ROWS__", "\n".join(rows)).replace("__MISSING__", miss))
    os.makedirs(OUTPUTS, exist_ok=True)
    out = os.path.join(OUTPUTS, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out


def index_item(name: str, video: str, ann: str):
    """What the home page shows for one video, or None if its player has not been built yet."""
    out_dir = os.path.join(OUTPUTS, name)
    if not os.path.exists(os.path.join(out_dir, "viewer.html")):
        return None
    segments = load_segments(ann)
    if not segments:
        return None
    preview = os.path.join(out_dir, "preview.mp4")
    play = preview if os.path.exists(preview) else video
    duration = max(video_info(play)["duration"], segments[-1]["end"])
    clips = len(glob.glob(os.path.join(out_dir, "clips", "*.mp4")))
    return {"name": name, "href": f"{name}/viewer.html", "segments": segments, "duration": duration,
            "status": review_status(ann), "clips": clips, "poster": poster(play, duration * 0.45)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the player (timeline + clips) for every annotated video.")
    ap.add_argument("names", nargs="*", help="only these videos (name without .mp4); default: all")
    ap.add_argument("--no-clips", action="store_true", help="skip the per-step clips (faster)")
    ap.add_argument("--no-open", action="store_true", help="do not open the home page in the browser")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the earlier outputs and make everything again, the video copies included")
    a = ap.parse_args()

    all_videos = sorted(glob.glob(os.path.join(VIDEOS, "*.mp4")))
    videos = all_videos
    if a.names:
        wanted = {n[:-4] if n.endswith(".mp4") else n for n in a.names}
        videos = [v for v in all_videos if os.path.splitext(os.path.basename(v))[0] in wanted]
        unknown = wanted - {os.path.splitext(os.path.basename(v))[0] for v in videos}
        for n in sorted(unknown):
            print(f"No video called {n}.mp4 in {VIDEOS}")
    if not videos:
        print(f"No videos to build in {VIDEOS}")
        return 1

    for i, video in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        ann = os.path.join(ANNOTATIONS, name + ".json")
        if not os.path.exists(ann):
            print(f"[{i}/{len(videos)}] {name}: no annotation file - skipped")
            continue
        print(f"[{i}/{len(videos)}] {name}")
        status = review_status(ann)
        try:
            build_viewer(video, ann, out_dir=os.path.join(OUTPUTS, name), make_clips=not a.no_clips,
                         lane_title=f"Annotated activities ({status or 'draft'})", back_href="../index.html",
                         fresh=a.fresh)
        except Exception:
            traceback.print_exc()
            print(f"    {name}: FAILED - see the error above; continuing with the next video")

    # The home page lists every video whose player exists, not only the ones built in this run.
    items, missing = [], []
    for video in all_videos:
        name = os.path.splitext(os.path.basename(video))[0]
        ann = os.path.join(ANNOTATIONS, name + ".json")
        if not os.path.exists(ann):
            missing.append(name + ".mp4")
            continue
        it = index_item(name, video, ann)
        if it:
            items.append(it)
    if not items:
        print("Nothing to show.")
        return 1
    index = write_index(items, missing)
    print(f"\nHome page ({len(items)} videos): {index}")
    if not a.no_open:
        open_in_browser(index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
