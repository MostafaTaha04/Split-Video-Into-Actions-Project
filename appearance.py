"""
appearance.py
=============
Training-free *appearance-change* signal for step-boundary detection.

Why this exists
---------------
The rule-based transition score is a maximum over ~15 binary cues. On four of
the five evaluation clips at least one cue fires on 73-98% of frames, so the
score sits on a plateau and peak-picking inside that plateau produces a
boundary every ``min_duration`` seconds (systematic over-segmentation; see
README, "Over-segmentation"). That score answers "is something happening?".
A step boundary is better described as "the scene now looks different from
how it looked a moment ago": a part has been placed, a cover removed, the
hands have moved to a different region of the device.

This module measures exactly that, without any training or downloaded model:

1. Each processed frame is reduced to a compact appearance descriptor -- a
   low-resolution grayscale thumbnail (layout / where things are) plus a
   coarse HSV colour histogram (what is visible), both normalised.
2. The *novelty* at time t is the cosine distance between the mean descriptor
   of the window just before t and the window just after t. It is high when
   the before- and after-states differ, and low both during steady work and
   during continuous motion that does not change the scene.

It is the same before/after-similarity idea as training-free action-boundary
detection on deep features (e.g. ABD, Du et al., CVPR 2022), with hand-crafted
descriptors in place of a CNN so that it runs anywhere with OpenCV alone.

Usage
-----
    # pre-compute descriptors for every registered clip (needs the videos)
    python appearance.py --src .

    # one video, aligned to a saved run's frames
    python appearance.py --video split-video-data/CPUplacement.mp4 \\
        --results results/results_cpu_final

Each run gets an ``appearance.npz`` (frame_idx, descriptors) beside its
``features.csv``; evaluate_extended.py picks it up automatically.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable, Optional

import cv2
import numpy as np

THUMB_LONG = 32          # thumbnail long side (short side keeps aspect)
HIST_BINS = (8, 4, 4)    # H, S, V bins
THUMB_WEIGHT = 0.5       # relative weight of layout vs colour


def describe(frame: np.ndarray) -> np.ndarray:
    """Appearance descriptor of one BGR frame (1-D float32, unit L2 norm)."""
    h, w = frame.shape[:2]
    if w >= h:
        size = (THUMB_LONG, max(1, round(THUMB_LONG * h / w)))
    else:
        size = (max(1, round(THUMB_LONG * w / h)), THUMB_LONG)
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    gray -= gray.mean()
    gray /= (np.linalg.norm(gray) + 1e-6)

    hsv = cv2.cvtColor(cv2.resize(frame, (w // 4 or 1, h // 4 or 1),
                                  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS),
                        [0, 180, 0, 256, 0, 256]).astype(np.float32).ravel()
    hist = np.sqrt(hist / (hist.sum() + 1e-6))          # Hellinger embedding
    hist -= hist.mean()
    hist /= (np.linalg.norm(hist) + 1e-6)

    d = np.concatenate([THUMB_WEIGHT * gray, (1.0 - THUMB_WEIGHT) * hist])
    return (d / (np.linalg.norm(d) + 1e-6)).astype(np.float32)


def novelty(desc: np.ndarray, fps: float, window_s: float = 2.0) -> np.ndarray:
    """Before/after appearance change per frame, in [0, 2] (cosine distance).

    ``desc`` is (n_frames, dim). Edges use the frames available, so the first
    and last ``window_s`` seconds have a shorter context.
    """
    desc = np.asarray(desc, dtype=np.float64)
    n = len(desc)
    if n == 0:
        return np.zeros(0)
    w = max(1, int(round(window_s * fps)))
    csum = np.vstack([np.zeros((1, desc.shape[1])), np.cumsum(desc, axis=0)])
    out = np.zeros(n)
    for t in range(1, n):
        a0, b1 = max(0, t - w), min(n, t + w)
        before = (csum[t] - csum[a0]) / (t - a0)
        after = (csum[b1] - csum[t]) / (b1 - t)
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        out[t] = 1.0 - float(before @ after) / denom if denom > 0 else 0.0
    return out


def extract(video_path: str, frame_indices: Iterable[int],
            resize: Optional[tuple] = None) -> tuple[np.ndarray, np.ndarray]:
    """Descriptors for the given source-frame indices (sequential decode)."""
    want = sorted(set(int(i) for i in frame_indices))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    out_idx, out = [], []
    k, idx = 0, 0
    while k < len(want):
        ok = cap.grab()
        if not ok:
            break
        if idx == want[k]:
            ok, frame = cap.retrieve()
            if ok:
                if resize:
                    frame = cv2.resize(frame, tuple(resize))
                out.append(describe(frame))
                out_idx.append(idx)
            k += 1
        idx += 1
    cap.release()
    return np.array(out_idx), np.array(out)


def _run_frames(results_dir: str):
    with open(os.path.join(results_dir, "features.csv"), encoding="utf-8") as fh:
        return [int(float(r["frame_idx"])) for r in csv.DictReader(fh)]


def _run_resize(results_dir: str):
    """Processing resolution the run used, so offline descriptors match live ones."""
    import json
    p = os.path.join(results_dir, "segmentation_results.json")
    try:
        with open(p, encoding="utf-8") as fh:
            r = (json.load(fh).get("run_config") or {}).get("frame_resize")
        return tuple(int(v) for v in r) if r else None
    except (OSError, ValueError):
        return None


def extract_for_run(video_path: str, results_dir: str) -> str:
    """Compute appearance.npz for a saved run, matching main.py's frames exactly
    (same source-frame indices, same processing resolution)."""
    idx, desc = extract(video_path, _run_frames(results_dir), resize=_run_resize(results_dir))
    out = os.path.join(results_dir, "appearance.npz")
    np.savez_compressed(out, frame_idx=idx, desc=desc)
    return out


def load_for_run(results_dir: str, frame_idx: Iterable[int]) -> Optional[np.ndarray]:
    """Descriptors aligned to ``frame_idx`` (None if not pre-computed)."""
    p = os.path.join(results_dir, "appearance.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    lut = {int(i): k for k, i in enumerate(z["frame_idx"])}
    rows = [lut.get(int(i)) for i in frame_idx]
    if any(r is None for r in rows):
        return None
    return z["desc"][rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", default=".")
    ap.add_argument("--video")
    ap.add_argument("--results")
    ap.add_argument("--only", help="registered clip name to process (default: all)")
    args = ap.parse_args()
    if args.video and args.results:
        print(extract_for_run(args.video, args.results))
        return
    from clip_registry import load_ai_registry, load_registry, resolve_results, resolve_video
    for c in load_registry(args.src) + load_ai_registry(args.src):
        if args.only and c["name"] != args.only:
            continue
        rd = resolve_results(c["results_dir"], args.src)
        if not os.path.exists(os.path.join(rd, "features.csv")):
            print(f"[skip] {c['name']}: no features.csv")
            continue
        try:
            vp = resolve_video(c["video"], args.src)
        except SystemExit:
            print(f"[skip] {c['name']}: video not found ({c['video']})")
            continue
        print(f"[ok]   {c['name']}: {extract_for_run(vp, rd)}")


if __name__ == "__main__":
    main()
