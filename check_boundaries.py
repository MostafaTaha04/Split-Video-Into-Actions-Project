"""
check_boundaries.py
===================
Generate a close-up contact sheet around every *predicted* boundary of a run, so
the predictions can be eyeballed on clips that have no ground truth.

Why this exists
---------------
Running the pipeline on an unannotated clip produces boundaries but no F1, so
there is no number to check. This does not turn qualitative inspection into a
metric, but it does make inspection fast and specific: instead of watching the
whole video, you look at 15 frames spanning +-0.75 s around each predicted
boundary and ask one question per sheet — did anything actually change here?

Counting the sheets where something did change gives an approximate *precision*
by inspection. It says nothing about recall: boundaries the system missed
entirely leave no sheet behind, because there is no prediction to centre on.
Only annotation measures recall.

Usage
-----
    python check_boundaries.py --run results/batch/ytbuildB_03_seg03
    python check_boundaries.py --run results/demo_standalone --window 1.0 --step 0.1
    python check_boundaries.py --run results/batch/ytbuildA_01_seg01 --limit 6

Sheets are written to <run>/boundary_checks/. Each is named with its timestamp
and the confidence the segmenter assigned, so low-confidence predictions can be
inspected first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a results directory")
    ap.add_argument("--window", type=float, default=0.75,
                    help="+/- seconds around each boundary (default 0.75)")
    ap.add_argument("--step", type=float, default=0.1, help="seconds between frames")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the N least-confident boundaries — the ones most worth a look")
    ap.add_argument("--video", default=None, help="override the video path")
    args = ap.parse_args()

    rj = os.path.join(args.run, "segmentation_results.json")
    if not os.path.exists(rj):
        raise SystemExit(f"no segmentation_results.json in {args.run}")
    with open(rj, encoding="utf-8") as fh:
        data = json.load(fh)

    bounds = data.get("boundaries", [])
    if not bounds:
        raise SystemExit(f"{args.run} contains no boundaries to check")

    # The run records its source under video_info.path, with Windows separators
    # when the run was made on Windows. Normalise so the same results folder can
    # be inspected from either platform.
    video = args.video or (data.get("video_info") or {}).get("path")
    if not video:
        raise SystemExit("could not determine the source video; pass --video")
    video = video.replace("\\", os.sep).replace("/", os.sep)
    if not os.path.exists(video):
        from clip_registry import resolve_video
        video = resolve_video(os.path.basename(video), ".")

    ordered = sorted(bounds, key=lambda b: float(b.get("confidence", 0.0)))
    if args.limit:
        ordered = ordered[:args.limit]
        print(f"{len(bounds)} boundaries; inspecting the {len(ordered)} least confident")
    else:
        ordered = sorted(bounds, key=lambda b: float(b["timestamp"]))
        print(f"{len(ordered)} boundaries to inspect")

    outdir = os.path.join(args.run, "boundary_checks")
    os.makedirs(outdir, exist_ok=True)

    for b in ordered:
        ts = float(b["timestamp"])
        conf = float(b.get("confidence", 0.0))
        tag = os.path.join(outdir, f"t{ts:07.2f}_conf{conf:.2f}")
        r = subprocess.run(
            [sys.executable, "annotate.py", "contact-sheet", "--video", video,
             "--around", f"{ts}", "--window", f"{args.window}", "--step", f"{args.step}",
             "--output", tag],
            capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"  {ts:7.2f}s  conf {conf:.2f}  {status}")

    print(f"\nSheets in {outdir}/")
    print("\nFor each sheet ask: does the action visibly change within these frames?")
    print("The fraction where it does is an approximate precision by inspection.")
    print("It does not measure recall — a missed boundary produces no sheet. Only")
    print("annotating the clip and running evaluate_extended.py measures both.")


if __name__ == "__main__":
    main()
