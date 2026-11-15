"""
action_direction.py
===================
Is a step *putting something in* (assembly) or *taking something out*
(disassembly)? A training-free motion cue plus a three-number model, applied
to each segment after segmentation. It never moves a boundary.

Why the segmentation itself is direction-blind
----------------------------------------------
Both boundary scorers look for moments of change (motion starting and stopping,
the scene looking different before and after). Removing a RAM module produces
the same kind of change as inserting one, so the boundaries do not depend on
direction. Only the *names* do: the rule-based labeller used to call every RAM
step "Press RAM module into slot", even in a disassembly video. This module
supplies the missing piece of information.

The cues (all from a separate, cheap pass over the video)
---------------------------------------------------------
Dense optical flow at 5 fps on a 320-pixel-wide copy, with the camera's own
motion removed (the median flow vector is subtracted) and edit cuts zeroed,
averaged onto a 32 x 18 grid. Per segment:

1. ``flux``: net radial flow about the frame centre, normalised by the total
   flow. The hands travel both in and out during any step, so their motion
   largely cancels; the *part* travels one way only. Installing brings it
   towards the work area (net inward, negative); removing takes it away (net
   outward, positive).
2. ``centroid``: where in the segment the motion is concentrated (0 = start,
   1 = end). Installing is bring-then-align (motion early); removing is
   unlatch-then-pull (motion late).
3. ``log_duration``: removing is usually quicker than installing, which needs
   alignment. This is a prior, not a visual cue, and it is wrong for long
   unscrewing steps; it is kept because it helps on average and is reported.

A logistic regression on these three numbers gives ``p_remove``. The weights
are fitted by ``evaluate_direction.py --fit`` on ground-truth step names
("Insert ...", "Remove ...", "Unscrew ...") and saved to
``direction_model.json``; the numbers below are the same fit, used when the
file is missing.

How reliable it is (see evaluate_direction.py / direction_results.json)
-----------------------------------------------------------------------
Weak, and reported as such. Leave-one-source-out on the 30 clips:

* on the annotated steps, about 2 steps in 3 right, about 3 in 4 of the
  steps it is confident about;
* on the segments the system itself finds, about 2 in 3, confident or not;
* per video it is usually "mixed or unclear", and it did NOT recognise the
  two disassembly recordings (input22, input23) as disassembly.

It is therefore used conservatively:

* auto (default): ``p_remove >= HIGH`` -> "disassembly", ``<= LOW`` ->
  "assembly", otherwise "unclear" with a direction-neutral name ("RAM module
  handling") instead of a name that claims installing.
* ``--task assembly|disassembly``, when you know what the video shows: every
  step is named for that direction. This is the reliable way to get correct
  names; ``p_remove`` is still recorded for each step.

Usage
-----
    # add direction to an existing run (segmentation_results.json is updated)
    python action_direction.py --video split-video-data/input22_aug_01.mp4 \\
        --results results/input22 [--task disassembly]

main.py does this automatically after segmentation (--task, --no-direction).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

FLOW_FPS = 5.0           # flow pass sampling rate
FLOW_WIDTH = 320         # flow computed on a copy this wide
GRID_W, GRID_H = 32, 18  # flow averaged onto this grid
CUT_DIFF = 30.0          # mean |diff| of 64x36 grey thumbnails above this = edit cut
MIN_FLOW_FRAMES = 3      # shorter segments get no direction

# Decision band on p_remove (see module docstring).
LOW, HIGH = 0.35, 0.65

ASSEMBLY, DISASSEMBLY, UNCLEAR = "assembly", "disassembly", "unclear"
FEATURES = ("flux", "centroid", "log_duration")

MODEL_FILE = "direction_model.json"
# Fallback = the fit written by `python evaluate_direction.py --fit`.
DEFAULT_MODEL = {
    "features": list(FEATURES),
    "mean": [-0.012427, 0.500699, 1.804789],
    "std": [0.13207, 0.071902, 0.566674],
    "coef": [0.307054, 0.312519, -0.624865],
    "intercept": -0.069715,
}


# --------------------------------------------------------------------------
# flow pass
# --------------------------------------------------------------------------
def extract_flow_grid(video_path: str, fps: float = FLOW_FPS,
                      max_time: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Camera-compensated optical flow on a coarse grid, sampled at ``fps``.

    Returns ``t`` (seconds, one per flow frame), ``grid`` (n, GRID_H, GRID_W, 2;
    pixels per sampled frame at FLOW_WIDTH), ``cut`` (bool per flow frame) and
    ``aspect`` (frame width / height).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    step = vfps / fps
    nxt, i = 0.0, 0
    prev = prev_small = None
    ts, grids, cuts = [], [], []
    while True:
        if max_time is not None and i / vfps > max_time:
            break
        if not cap.grab():
            break
        if i + 1e-6 >= nxt:
            ok, frame = cap.retrieve()
            nxt += step
            if not ok:
                i += 1
                continue
            h, w = frame.shape[:2]
            hh = max(1, int(round(FLOW_WIDTH * h / w)))
            grey = cv2.cvtColor(cv2.resize(frame, (FLOW_WIDTH, hh), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None:
                cut = float(np.abs(small - prev_small).mean()) > CUT_DIFF
                flow = cv2.calcOpticalFlowFarneback(prev, grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow -= np.median(flow.reshape(-1, 2), axis=0)      # remove camera motion
                grid = cv2.resize(flow, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
                if cut:
                    grid[:] = 0.0
                ts.append(i / vfps)
                grids.append(grid.astype(np.float16))
                cuts.append(cut)
            prev, prev_small = grey, small
        i += 1
    cap.release()
    return {
        "t": np.asarray(ts, dtype=np.float64),
        "grid": (np.asarray(grids, dtype=np.float16) if grids
                 else np.zeros((0, GRID_H, GRID_W, 2), np.float16)),
        "cut": np.asarray(cuts, dtype=bool),
        "aspect": np.float64(width / height if height else 16 / 9),
    }


def save_flow(path: str, flow: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **flow)


def load_flow(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _video_id(video_path: str) -> str:
    return f"{os.path.basename(video_path)}:{os.path.getsize(video_path)}"


def flow_for_run(video_path: str, results_dir: str,
                 max_time: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Cached flow grid for a run (``direction_flow.npz`` beside features.csv).

    The cache is reused only if it was made from the same video (name and
    size) and covers ``max_time``; otherwise it is recomputed.
    """
    cache = os.path.join(results_dir, "direction_flow.npz")
    vid = _video_id(video_path) if os.path.exists(video_path) else None
    if os.path.exists(cache):
        flow = load_flow(cache)
        same = vid is None or str(flow.get("video", "")) == vid
        long_enough = max_time is None or (len(flow["t"]) and flow["t"][-1] >= max_time - 1.0)
        if same and long_enough:
            return flow
    flow = extract_flow_grid(video_path, max_time=max_time)
    flow["video"] = np.array(vid or os.path.basename(video_path))
    os.makedirs(results_dir, exist_ok=True)
    save_flow(cache, flow)
    return flow


# --------------------------------------------------------------------------
# per-window features and model
# --------------------------------------------------------------------------
_YY, _XX = np.mgrid[0:GRID_H, 0:GRID_W]
_XX = (_XX + 0.5) / GRID_W - 0.5
_YY = (_YY + 0.5) / GRID_H - 0.5


def window_features(flow: Dict[str, np.ndarray], start: float, end: float) -> Optional[np.ndarray]:
    """[flux, centroid, log_duration] for the window [start, end), or None if
    the window holds fewer than MIN_FLOW_FRAMES flow frames."""
    t = flow["t"]
    m = (t >= start) & (t < end)
    if m.sum() < MIN_FLOW_FRAMES:
        return None
    g = flow["grid"][m].astype(np.float32)
    aspect = float(flow["aspect"])
    # radial unit vectors about the frame centre, in pixel space
    dx, dy = _XX * aspect, _YY
    n = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    rx, ry = dx / n, dy / n
    mag = np.linalg.norm(g, axis=-1)
    total = float(mag.sum())
    if total <= 1e-6:
        return None
    flux = float((g[..., 0] * rx + g[..., 1] * ry).sum() / total)
    energy = mag.sum(axis=(1, 2))
    tn = np.linspace(0.0, 1.0, len(energy))
    centroid = float((energy * tn).sum() / energy.sum())
    return np.array([flux, centroid, np.log(max(end - start, 0.2))], dtype=np.float64)


def load_model(path: Optional[str] = None) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ([path] if path else []) + [os.path.join(here, MODEL_FILE)]:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return dict(DEFAULT_MODEL)


def p_remove(x: np.ndarray, model: dict) -> float:
    z = (np.asarray(x, float) - np.asarray(model["mean"])) / np.asarray(model["std"])
    s = float(np.dot(z, model["coef"]) + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-s)))


def decide(p: Optional[float], task: str = "auto") -> (str, str):
    """(direction, source) for one segment.

    auto: "disassembly" if p >= HIGH, "assembly" if p <= LOW, else "unclear";
    source "motion evidence" (or "none" when unclear). With --task the step
    takes the task's direction (source "task setting"). Letting very strong
    opposite evidence override --task was tried and measured: it changed the
    share of correct steps by less than one point (evaluate_direction.py), so
    the simpler rule is used. p_remove is still recorded for every step.
    """
    if task in (ASSEMBLY, DISASSEMBLY):
        return task, "task setting"
    if p is None or LOW < p < HIGH:
        return UNCLEAR, "none"
    return (DISASSEMBLY if p >= HIGH else ASSEMBLY), "motion evidence"


def summarise(directions: Sequence[str], task: str = "auto") -> dict:
    """Video-level counts and a cautious verdict (auto mode only)."""
    n_a = sum(d == ASSEMBLY for d in directions)
    n_d = sum(d == DISASSEMBLY for d in directions)
    n_u = sum(d == UNCLEAR for d in directions)
    decided = n_a + n_d
    if task in (ASSEMBLY, DISASSEMBLY):
        verdict = f"{task} (set with --task)"
    elif decided >= 3 and n_a >= 0.7 * decided:
        verdict = "mostly assembly"
    elif decided >= 3 and n_d >= 0.7 * decided:
        verdict = "mostly disassembly"
    else:
        verdict = "mixed or unclear"
    return {
        "task": task, "n_assembly": n_a, "n_disassembly": n_d, "n_unclear": n_u,
        "verdict": verdict,
        "note": ("auto mode: a weak motion cue (net in/out motion, timing, duration), about 2 "
                 "steps in 3 right in testing (evaluate_direction.py); pass --task "
                 "assembly|disassembly when you know the video type."),
    }


def analyse(flow: Dict[str, np.ndarray], windows: Iterable[Sequence[float]],
            task: str = "auto", model: Optional[dict] = None) -> List[dict]:
    """One decision per (start, end) window."""
    model = model or load_model()
    out = []
    for start, end in windows:
        x = window_features(flow, float(start), float(end))
        p = None if x is None else p_remove(x, model)
        direction, source = decide(p, task)
        out.append({
            "direction": direction,
            "direction_p_remove": None if p is None else round(p, 3),
            "direction_source": source,
            "direction_cues": (None if x is None
                               else {k: round(float(v), 4) for k, v in zip(FEATURES, x)}),
        })
    return out


# --------------------------------------------------------------------------
# apply to a saved run
# --------------------------------------------------------------------------
def apply_to_results(results_dir: str, video_path: str, task: str = "auto") -> dict:
    """Add direction fields (and direction-aware names) to segmentation_results.json."""
    from activity_recognizer import reword

    path = os.path.join(results_dir, "segmentation_results.json")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    segs = res.get("segments", [])
    max_t = max((s["end_time"] for s in segs), default=None)
    flow = flow_for_run(video_path, results_dir, max_time=max_t)
    decisions = analyse(flow, [(s["start_time"], s["end_time"]) for s in segs], task)
    for s, d in zip(segs, decisions):
        s.update(d)
        key = s.get("activity_key")
        if key:
            s["activity"] = reword(key, d["direction"])
    res["direction_summary"] = summarise([d["direction"] for d in decisions], task)
    res.setdefault("run_config", {})["task"] = task
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--results", required=True, help="run folder with segmentation_results.json")
    ap.add_argument("--task", choices=["auto", ASSEMBLY, DISASSEMBLY], default="auto")
    args = ap.parse_args()
    res = apply_to_results(args.results, args.video, args.task)
    for s in res["segments"]:
        p = s.get("direction_p_remove")
        print(f"  {s['start_time']:7.1f}-{s['end_time']:7.1f}s  {s['direction']:12s} "
              f"p_remove={'  -  ' if p is None else f'{p:.2f}'}  {s.get('activity', '')}")
    print("Summary:", res["direction_summary"]["verdict"])


if __name__ == "__main__":
    main()
