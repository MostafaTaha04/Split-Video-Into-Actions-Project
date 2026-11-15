"""
edit_cuts.py
============
Find hard edit cuts (an instantaneous change of shot) in a video.

Why this exists
---------------
The YouTube build clips are edited. When an editor cuts from one step to the
next, the step boundary and the cut fall on the same frame, and any detector
that reacts to a sudden change of picture gets that boundary for free. To know
how much of a score on edited footage comes from finding cuts rather than from
following the hands, the evaluation needs the cut times themselves. With them,
evaluate_extended.py reports, for the AI-annotated clips:

* the fraction of ground-truth boundaries that sit on a cut,
* a cut-only baseline (predict exactly the cuts, nothing else),
* each scorer's recall on boundaries AT a cut and NOT at a cut separately.

Detector (no hand tracking, no learned model)
---------------------------------------------
Every source frame -> grayscale 64x36 thumbnail -> mean absolute difference to
the previous frame (0-255). A frame starts a new shot when that difference is
above 30, at least 4x the median difference of the surrounding +-15 frames,
and the largest in that window. Cuts closer than 1.5 s are merged, and cuts in
the first or last 0.5 s are ignored. The pipeline's own scene_change_score is
not used: it is smoothed and a hard cut barely raises it.

Usage
-----
    python edit_cuts.py --src .                       # every registered clip
    python edit_cuts.py --src . --only "YT build C10"
    python edit_cuts.py --video clip.mp4              # print the cuts only

Writes <results_dir>/edit_cuts.json: {"video", "fps", "duration_s", "cuts": [...]}.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

MIN_DIFF = 30.0
LOCAL_RATIO = 4.0
HALF_WINDOW = 15
MIN_GAP_S = 1.5
EDGE_S = 0.5


def frame_differences(video_path: str):
    """(times, diffs): mean absolute difference of each frame to the previous one."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, times, diffs, i = None, [], [], 0
    while cap.grab():
        ok, frame = cap.retrieve()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            times.append(i / fps)
            diffs.append(float(np.abs(g - prev).mean()))
        prev, i = g, i + 1
    cap.release()
    return np.array(times), np.array(diffs), fps, i / fps


def find_cuts(times: np.ndarray, diffs: np.ndarray, duration: float) -> List[float]:
    cuts: List[float] = []
    for i, v in enumerate(diffs):
        lo, hi = max(0, i - HALF_WINDOW), min(len(diffs), i + HALF_WINDOW + 1)
        around = np.r_[diffs[lo:i], diffs[i + 1:hi]]
        local = float(np.median(around)) if len(around) else 0.0
        if v > MIN_DIFF and v > LOCAL_RATIO * max(local, 2.0) and v == diffs[lo:hi].max():
            if not cuts or times[i] - cuts[-1] >= MIN_GAP_S:
                cuts.append(round(float(times[i]), 2))
    return [c for c in cuts if EDGE_S < c < duration - EDGE_S]


def cuts_for_video(video_path: str) -> dict:
    t, d, fps, duration = frame_differences(video_path)
    return {"video": os.path.basename(video_path), "fps": round(fps, 3),
            "duration_s": round(duration, 2), "cuts": find_cuts(t, d, duration),
            "detector": {"min_diff": MIN_DIFF, "local_ratio": LOCAL_RATIO,
                         "half_window_frames": HALF_WINDOW, "min_gap_s": MIN_GAP_S}}


def load_cuts(results_dir: str):
    """Saved cut times for a run, or None if edit_cuts.py has not been run."""
    p = os.path.join(results_dir, "edit_cuts.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["cuts"]


def main():
    ap = argparse.ArgumentParser(description="Detect hard edit cuts (see module docstring).")
    ap.add_argument("--src", default=".")
    ap.add_argument("--only", help="registered clip name (default: all)")
    ap.add_argument("--video", help="just print the cuts of this video")
    args = ap.parse_args()
    if args.video:
        print(json.dumps(cuts_for_video(args.video), indent=2))
        return
    from clip_registry import load_ai_registry, load_registry, resolve_results, resolve_video
    for c in load_registry(args.src) + load_ai_registry(args.src):
        if args.only and c["name"] != args.only:
            continue
        rd = resolve_results(c["results_dir"], args.src)
        try:
            vp = resolve_video(c["video"], args.src)
        except SystemExit:
            print(f"[skip] {c['name']}: video not found ({c['video']})")
            continue
        res = cuts_for_video(vp)
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "edit_cuts.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"[ok]   {c['name']}: {len(res['cuts'])} cuts")


if __name__ == "__main__":
    main()
