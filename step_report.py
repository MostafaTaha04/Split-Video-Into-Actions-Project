"""
step_report.py
==============
Turn a segmentation run into a self-contained, shareable HTML page: the
"structured, step-by-step procedural breakdown" the project brief asks for.

For every detected step the page shows its time span, three keyframes (start,
middle, end) and -- where hands were tracked -- the palm positions during the
step drawn on the middle keyframe, i.e. *where on the device the work
happened*. If ground truth is available, each step also shows the annotated
step it overlaps most, so the segmentation can be checked by eye.

No action labels are shown: the brief requires segmentation, not recognition,
and the rule-based labeller is a documented negative result (README). Each
step does show whether it looked like assembly or disassembly, when the run
recorded it (action_direction.py), with the strength of that evidence.

main.py writes this automatically (disable with --no-report). For a saved run:

    python step_report.py --run results/results_cpu_final \\
        --video split-video-data/CPUplacement.mp4 \\
        --ground-truth ground_truth/ground_truth_cpuplacement.json
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from typing import Optional

import cv2
import numpy as np

THUMB_W = 260


def _b64_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _b64_png_file(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _frame_at(cap, fps: float, t: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return cv2.resize(frame, (THUMB_W, max(1, int(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)


def _draw_points(img: np.ndarray, points) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for x, y, grip in points:
        c = (int(x * w), int(y * h))
        if grip:
            cv2.circle(out, c, 4, (0, 140, 255), -1)          # gripping: filled orange
        else:
            cv2.circle(out, c, 4, (255, 220, 0), 1)           # open hand: cyan ring
    return out


def _load_gt(gt_path: Optional[str]):
    if not gt_path or not os.path.exists(gt_path):
        return []
    with open(gt_path, encoding="utf-8") as fh:
        steps = json.load(fh).get("steps", [])
    return [(float(s["start"]), float(s["end"]), s.get("label", "")) for s in steps]


def _best_overlap(a0, a1, gt):
    best, lab = 0.0, ""
    for g0, g1, label in gt:
        ov = max(0.0, min(a1, g1) - max(a0, g0))
        if ov > best:
            best, lab = ov, label
    return lab, (best / (a1 - a0) if a1 > a0 else 0.0)


def build_report(run_dir: str, video_path: Optional[str] = None,
                 gt_path: Optional[str] = None, out: Optional[str] = None) -> str:
    with open(os.path.join(run_dir, "segmentation_results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    video_path = video_path or res.get("video_info", {}).get("path")
    pts_path = os.path.join(run_dir, "interaction_points.json")
    points = {}
    if os.path.exists(pts_path):
        with open(pts_path, encoding="utf-8") as fh:
            points = json.load(fh)
    gt = _load_gt(gt_path)

    cap = cv2.VideoCapture(video_path) if video_path else None
    fps = (cap.get(cv2.CAP_PROP_FPS) if cap is not None and cap.isOpened() else 0) or 0

    cards = []
    for s in res["segments"]:
        t0, t1 = float(s["start_time"]), float(s["end_time"])
        pad = min(0.3, (t1 - t0) / 4)
        thumbs = []
        if fps:
            for k, t in enumerate((t0 + pad, (t0 + t1) / 2, t1 - pad)):
                img = _frame_at(cap, fps, t)
                if img is None:
                    continue
                if k == 1:
                    img = _draw_points(img, points.get(str(s["id"]), {}).get("points", []))
                thumbs.append(f'<figure><img src="{_b64_jpeg(img)}"><figcaption>'
                              f'{["start", "middle", "end"][k]} &middot; {t:.1f}s</figcaption></figure>')
        p = points.get(str(s["id"]), {})
        hand_note = (f'{p["n_hand_detections"]} hand detections in {p["n_frames"]} frames'
                     if p else "hand positions not recorded for this run")
        gt_html = ""
        if gt:
            lab, frac = _best_overlap(t0, t1, gt)
            gt_html = (f'<div class="gt">annotated step: <b>{html.escape(lab) or "&ndash;"}</b> '
                       f'({frac:.0%} of this step)</div>')
        dir_html = ""
        if "direction" in s:
            p_rm = s.get("direction_p_remove")
            ev = "no motion evidence" if p_rm is None else f"P(removing) = {p_rm:.2f}"
            dir_html = (f'<div class="dir dir-{html.escape(s["direction"])}">direction: '
                        f'<b>{html.escape(s["direction"])}</b> ({ev}; '
                        f'{html.escape(str(s.get("direction_source", "")))})</div>')
        cards.append(
            f'<section class="step"><header><span class="n">Step {s["id"] + 1}</span>'
            f'<span class="t">{t0:.2f}s &ndash; {t1:.2f}s &middot; {t1 - t0:.1f}s</span></header>'
            f'{gt_html}{dir_html}<div class="thumbs">{"".join(thumbs)}</div>'
            f'<div class="meta">{hand_note}</div></section>')
    if cap is not None:
        cap.release()

    bounds = "".join(
        f"<tr><td>{b['timestamp']:.2f}s</td><td>{b['confidence']:.2f}</td>"
        f"<td>{html.escape(str(b.get('reason', '')).replace('|', ', '))}</td></tr>"
        for b in res["boundaries"])
    tl = os.path.join(run_dir, "timeline.png")
    timeline = f'<img class="timeline" src="{_b64_png_file(tl)}">' if os.path.exists(tl) else ""
    ev = os.path.join(run_dir, "evaluation_report.txt")
    evaluation = ""
    if os.path.exists(ev):
        with open(ev, encoding="utf-8") as fh:
            evaluation = (f"<details open><summary>Evaluation against ground truth</summary>"
                          f"<pre>{html.escape(fh.read())}</pre></details>")
    cfg = res.get("run_config", {})
    cfg_txt = " &middot; ".join(f"{k}: {html.escape(str(v))}" for k, v in cfg.items())
    name = os.path.basename(video_path or run_dir)
    ds = res.get("direction_summary")
    dir_sum = ("" if not ds else
               f"<br>direction: {ds['n_assembly']} assembly, {ds['n_disassembly']} disassembly, "
               f"{ds['n_unclear']} unclear &rarr; <b>{html.escape(ds['verdict'])}</b>")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step breakdown &ndash; {html.escape(name)}</title>
<style>
 body {{ font: 15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; background: #f6f6f4; color: #1d1d1b; }}
 main {{ max-width: 1000px; margin: 0 auto; padding: 28px 20px 60px; }}
 h1 {{ font-size: 24px; margin: 0 0 4px; }}
 .sub {{ color: #5b5b57; margin-bottom: 18px; font-size: 13px; }}
 .timeline {{ width: 100%; border-radius: 6px; margin: 8px 0 18px; }}
 .step {{ background: #fff; border: 1px solid #e3e3df; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
 .step header {{ display: flex; gap: 14px; align-items: baseline; }}
 .n {{ font-weight: 650; font-size: 17px; }}
 .t {{ color: #5b5b57; font-variant-numeric: tabular-nums; }}
 .gt {{ margin-top: 4px; font-size: 13px; color: #2f5d3a; }}
 .dir {{ margin-top: 4px; font-size: 13px; color: #5b5b57; }}
 .dir-assembly b {{ color: #2f5d3a; }} .dir-disassembly b {{ color: #9a3412; }}
 .thumbs {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
 figure {{ margin: 0; }} figure img {{ display: block; border-radius: 4px; max-width: {THUMB_W}px; }}
 figcaption {{ font-size: 12px; color: #6b6b66; margin-top: 3px; }}
 .meta {{ font-size: 12px; color: #6b6b66; margin-top: 6px; }}
 table {{ border-collapse: collapse; font-size: 13px; background: #fff; }}
 td, th {{ border-bottom: 1px solid #eee; padding: 4px 10px; text-align: left; }}
 pre {{ background: #fff; border: 1px solid #e3e3df; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; }}
 .legend {{ font-size: 12px; color: #6b6b66; }}
</style></head><body><main>
<h1>Step-by-step breakdown</h1>
<div class="sub">{html.escape(name)} &middot; {len(res["segments"])} steps, {len(res["boundaries"])} boundaries<br>{cfg_txt}{dir_sum}</div>
{timeline}
<p class="legend">Dots on the middle keyframe mark where the hands were during the step
(filled orange = gripping, ring = open hand).</p>
{"".join(cards)}
<h2>Boundaries</h2>
<table><tr><th>time</th><th>score</th><th>cues active at the boundary</th></tr>{bounds}</table>
{evaluation}
</main></body></html>"""
    out = out or os.path.join(run_dir, "step_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out


def main():
    ap = argparse.ArgumentParser(description="Build step_report.html for a run.")
    ap.add_argument("--run", required=True, help="results folder of a run")
    ap.add_argument("--video", help="source video (default: path recorded in the run)")
    ap.add_argument("--ground-truth", help="ground-truth JSON (optional)")
    ap.add_argument("--out", help="output path (default: <run>/step_report.html)")
    a = ap.parse_args()
    print(build_report(a.run, a.video, a.ground_truth, a.out))


if __name__ == "__main__":
    main()
