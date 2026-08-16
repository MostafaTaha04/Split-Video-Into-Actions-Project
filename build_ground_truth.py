"""
build_ground_truth.py
=====================
Turn a list of boundary times into a valid ground-truth file.

Annotation is a human judgement: you read the contact sheets and decide where
one step ends and the next begins. This tool only handles the mechanical part —
converting those times into the contiguous step objects the evaluator expects,
filling in duration and fps from the video itself, and refusing the mistakes
that ``annotate.py validate`` would reject anyway.

It deliberately does NOT propose boundaries. Ground truth produced by the same
system it scores would make the evaluation circular.

Usage
-----
    python build_ground_truth.py --video ytbuildB_01_seg01.mp4 \
        --boundaries 12.3 24.7 38.1 51.4 66.9 80.2 97.5 \
        --annotator "Mostafa Taha" \
        --out ground_truth/ground_truth_ytbuildb1.json

Labels are optional; add them with --labels (one per step, so one more than the
number of boundaries):

    --labels "Unbox parts" "Seat CPU" "Clamp lever" ...

Then check the result:
    python annotate.py validate --ground-truth <out> --video <video>
"""
from __future__ import annotations

import argparse
import json
import os

import cv2

from clip_registry import resolve_video


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--boundaries", nargs="+", type=float, required=True,
                    help="interior step boundaries in seconds, ascending")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one label per step (= len(boundaries) + 1)")
    ap.add_argument("--annotator", required=True, help="who made these judgements")
    ap.add_argument("--method", default="contact sheets at 1.0s, refined to 0.1s",
                    help="how the boundaries were determined")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.video if os.path.exists(args.video) else resolve_video(args.video, ".")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    claimed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Count frames that actually decode rather than trusting the container.
    # Clips produced by split_recording.py are cut with a stream copy, which
    # leaves the header claiming more frames than the file holds — 3407 vs 3300
    # on ytbuildB_03_seg03.mp4, an overstatement of 3.6s. Trusting the header
    # would let a boundary be placed in a region that has no frames, and would
    # record a video_duration the evaluator can never reach.
    real = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        real += 1
    cap.release()

    if real == 0:
        raise SystemExit(f"no decodable frames in {path}")
    if claimed - real > 1:
        print(f"[note] container claims {claimed} frames ({claimed/fps:.2f}s) but only "
              f"{real} decode ({real/fps:.2f}s). Using the decodable length; the last "
              f"{(claimed-real)/fps:.2f}s of this file does not exist.")
    duration = real / fps

    b = sorted(args.boundaries)
    if b != args.boundaries:
        print("[note] boundaries were not ascending; sorted them")
    if len(set(b)) != len(b):
        raise SystemExit("duplicate boundary times")
    if b[0] <= 0 or b[-1] >= duration:
        raise SystemExit(f"boundaries must lie strictly inside 0..{duration:.2f}s")

    # The template-artifact check that annotate.py validate enforces. Catching it
    # here means it is caught before the file is written, not after.
    whole = sum(1 for t in b if abs(t - round(t)) < 1e-6)
    frac = whole / len(b)
    if frac >= 0.6:
        print(f"\n[REFUSED] {whole}/{len(b)} boundaries ({frac:.0%}) fall on a whole second.")
        print("annotate.py validate rejects files at >=60% because that is the signature")
        print("of times typed from memory rather than read off frames. Refine them:")
        print(f"  python annotate.py contact-sheet --video {args.video} "
              f"--around <t> --window 1.0 --step 0.1")
        raise SystemExit(1)
    if frac > 0.3:
        print(f"[warn] {whole}/{len(b)} boundaries ({frac:.0%}) are on whole seconds — "
              "under the 60% limit, but refine what you can.")

    edges = [0.0] + b + [round(duration, 3)]
    n_steps = len(edges) - 1
    labels = args.labels or []
    if labels and len(labels) != n_steps:
        raise SystemExit(f"{len(labels)} labels for {n_steps} steps "
                         f"(need one per step = boundaries + 1)")

    steps = [{"id": i,
              "start": round(edges[i], 3),
              "end": round(edges[i + 1], 3),
              "label": labels[i] if labels else f"Step {i + 1}"}
             for i in range(n_steps)]

    short = [s for s in steps if s["end"] - s["start"] < 1.5]
    if short:
        print(f"[warn] {len(short)} step(s) shorter than 1.5s: "
              f"{[s['id'] for s in short]} — check these are real steps.")

    data = {"video": os.path.basename(path), "annotator": args.annotator,
            "annotation_method": args.method, "video_duration": round(duration, 2),
            "fps": round(fps, 3), "steps": steps}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(f"\nwrote {args.out}: {n_steps} steps, {len(b)} interior boundaries, "
          f"{duration:.2f}s")
    for s in steps:
        print(f"  {s['id']:2d}  {s['start']:7.2f} - {s['end']:7.2f}  "
              f"({s['end']-s['start']:5.2f}s)  {s['label']}")
    print(f"\nnext: python annotate.py validate --ground-truth {args.out} "
          f"--video {os.path.basename(path)}")


if __name__ == "__main__":
    main()
