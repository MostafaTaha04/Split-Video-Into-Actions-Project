"""
diagnose_hands.py
=================
Why does MediaPipe find no hands in some clips?

Hand tracking is the pipeline's primary cue, yet the saved features for four of
the five original clips contain zero hand detections — including the two that
produce the headline results. Two candidate causes were identified:

  1. aspect-ratio destruction (portrait 1080x1920 squashed to 960x540),
     since fixed in video_loader.py; and
  2. `min_hand_detection_confidence = 0.7`, well above MediaPipe's 0.5
     default, on footage where hands are frequently partial or cropped.

Fixing (1) alone did not help, so this script isolates the remaining variables
by running the detector directly over a grid of settings and reporting the
detection rate for each. It is a diagnostic, not part of the pipeline.

Usage
-----
    python diagnose_hands.py --video split-video-data/Coolingfaninstallation.mp4
    python diagnose_hands.py --video ... --frames 60 --save-overlay
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = np.linspace(0, max(0, total - 1), n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames, (w, h)


def resize_variants(frame, src_wh):
    """The frame as the pipeline sees it, plus alternatives to test."""
    w, h = src_wh
    out = {"original": frame}

    # what the pipeline used to do: force landscape onto portrait
    out["squashed_960x540"] = cv2.resize(frame, (960, 540))

    # current behaviour: orientation-matched budget
    rw, rh = (540, 960) if h > w else (960, 540)
    out[f"oriented_{rw}x{rh}"] = cv2.resize(frame, (rw, rh))

    # larger, orientation-matched — MediaPipe likes more pixels on the hand
    rw2, rh2 = (720, 1280) if h > w else (1280, 720)
    out[f"oriented_{rw2}x{rh2}"] = cv2.resize(frame, (rw2, rh2))
    return out


def run_grid(frames, src_wh, confidences, save_overlay=False, outdir="."):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        raise SystemExit("mediapipe is required:\n    pip install mediapipe")

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        raise SystemExit(f"{model_path} not found — run the pipeline once to download it")
    with open(model_path, "rb") as fh:
        model_bytes = fh.read()

    variants = list(resize_variants(frames[0], src_wh))
    print(f"{len(frames)} frames  |  source {src_wh[0]}x{src_wh[1]}"
          f" ({'portrait' if src_wh[1] > src_wh[0] else 'landscape'})\n")
    print(f"{'resize':<24}" + "".join(f"conf={c:<8.2f}" for c in confidences))
    print("-" * (24 + 13 * len(confidences)))

    best = (0.0, None, None)
    for vname in variants:
        row = f"{vname:<24}"
        for conf in confidences:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
                running_mode=mp_vision.RunningMode.IMAGE,   # per-frame, no tracking state
                num_hands=2,
                min_hand_detection_confidence=conf,
                min_hand_presence_confidence=conf,
                min_tracking_confidence=conf,
            )
            det = mp_vision.HandLandmarker.create_from_options(opts)
            hits = 0
            for fr in frames:
                img = resize_variants(fr, src_wh)[vname]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if res.hand_landmarks:
                    hits += 1
            det.close()
            rate = 100.0 * hits / len(frames)
            row += f"{rate:>7.1f}%     "
            if rate > best[0]:
                best = (rate, vname, conf)
        print(row)

    print(f"\nBEST: {best[0]:.1f}% detection with resize={best[1]}, confidence={best[2]}")
    if best[0] < 15:
        print("\nUnder ~15% at every setting means the footage itself is the problem —\n"
              "overhead framing with cropped or heavily occluded hands. No threshold\n"
              "change will rescue it; the honest conclusion is that hand tracking does\n"
              "not contribute on this clip, and the report should say so.")
    elif best[1] and "squashed" not in best[1]:
        print("\nA better resize and/or a lower confidence recovers hand tracking.\n"
              "Update config.hand_detection_confidence and re-run the pipeline.")

    if save_overlay and best[1]:
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=best[2],
            min_hand_presence_confidence=best[2], min_tracking_confidence=best[2])
        det = mp_vision.HandLandmarker.create_from_options(opts)
        tiles = []
        for fr in frames[:12]:
            img = resize_variants(fr, src_wh)[best[1]].copy()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            h, w = img.shape[:2]
            for hand in (res.hand_landmarks or []):
                for lm in hand:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            tiles.append(cv2.resize(img, (240, int(240 * h / w))))
        det.close()
        if tiles:
            hh = min(t.shape[0] for t in tiles)
            grid = np.hstack([t[:hh] for t in tiles[:6]])
            path = os.path.join(outdir, "hand_diagnostic.png")
            cv2.imwrite(path, grid)
            print(f"\nwrote {path} — green dots are detected landmarks")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--confidences", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--save-overlay", action="store_true")
    args = ap.parse_args()

    frames, wh = sample_frames(args.video, args.frames)
    if not frames:
        raise SystemExit("no frames read")
    run_grid(frames, wh, args.confidences, args.save_overlay)


if __name__ == "__main__":
    main()
