"""
add_clip.py
===========
Take a new video from raw file to registered evaluation clip in one command.

Adding a clip by hand means running the pipeline, finding the effective fps,
creating an annotation file in the right shape, and editing the registry —
four chances to get something subtly wrong, repeated for every clip. This
script does all of it, validates the result, and refuses to register anything
malformed.

Workflow
--------
    # 1. run the vision pipeline and register the clip (annotation still empty)
    python add_clip.py --video myclip.mp4 --name "SSD install" --footage clean

    # 2. annotate it — contact sheets, then the interactive reviewer
    python annotate.py contact-sheet --video myclip.mp4 --step 1.0
    python annotate.py review --video myclip.mp4 \
        --output ground_truth_ssdinstall.json --annotator "Your Name"

    # 3. check the annotation, then re-score everything with the new clip
    python annotate.py validate --ground-truth ground_truth_ssdinstall.json --video myclip.mp4
    python evaluate_extended.py --src .

By default a new clip is registered as ``heldout``: a clip should not influence
tuning until you have deliberately decided it should. Pass ``--split dev`` to
put it in the development set.

Options
-------
``--skip-pipeline`` register a clip whose results_dir already exists.
``--dry-run``       show what would happen, write nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from clip_registry import add_clip as register
from clip_registry import load_registry, summary


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def effective_fps(results_dir: str) -> float:
    """Effective processing fps of a saved run, measured from its own timestamps.

    Read from the run rather than assumed, because ``--fps`` is a *target* and
    the loader can land slightly off it.

    Uses total span / (n-1) rather than the median frame interval: timestamps
    are rounded when written to CSV, which quantises individual gaps. On the
    project's own clips the median estimator is wrong by 0.06 fps for
    29.97 fps sources (14.925 against the true 14.985) while the span
    estimator is exact for all five.
    """
    import csv

    path = os.path.join(results_dir, "features.csv")
    with open(path, encoding="utf-8") as fh:
        ts = [float(r["timestamp"]) for r in csv.DictReader(fh)]
    if len(ts) < 3:
        raise SystemExit(f"{path} has too few rows to infer fps")
    span = ts[-1] - ts[0]
    if span <= 0:
        raise SystemExit(f"{path}: non-increasing timestamps")
    return round((len(ts) - 1) / span, 3)


def run_pipeline(args, results_dir):
    cmd = [
        sys.executable, "main.py",
        "--video", args.video,
        "--output", results_dir,
        "--fps", str(args.fps),
        "--detector", args.detector,
        "--threshold", str(args.threshold),
        "--min-duration", str(args.min_duration),
        "--resize", args.resize,
        "--grip-window", "7",
        "--object-confidence", "0.10",
    ]
    if args.model:
        cmd += ["--model", args.model, "--open-vocab-imgsz", "960"]
    if args.no_clips:
        cmd.append("--no-clips")

    print("[run] " + " ".join(cmd) + "\n", flush=True)
    if args.dry_run:
        return
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"pipeline failed (exit {res.returncode}); clip not registered")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to the new .mp4")
    ap.add_argument("--name", required=True, help='display name, e.g. "SSD install"')
    ap.add_argument("--footage", choices=["clean", "edited"], required=True,
                    help="'clean' = one continuous shot; 'edited' = cuts/overlays")
    ap.add_argument("--split", choices=["dev", "heldout"], default="heldout",
                    help="default heldout: a new clip should not affect tuning "
                         "until you decide it should")
    ap.add_argument("--results-dir", default=None, help="default: results_<slug>")
    ap.add_argument("--ground-truth", default=None,
                    help="default: ground_truth_<slug>.json")
    ap.add_argument("--src", default=".")

    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--detector", default="open_vocab")
    ap.add_argument("--model", default="yolov8l-worldv2.pt")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--resize", default="960x540")
    ap.add_argument("--no-clips", action="store_true",
                    help="skip exporting per-step clips (faster)")

    ap.add_argument("--skip-pipeline", action="store_true",
                    help="results_dir already exists; just register it")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        from clip_registry import resolve_video
        args.video = resolve_video(os.path.basename(args.video), args.src)

    s = slug(args.name)
    results_dir = args.results_dir or f"results_{s}"
    gt_file = args.ground_truth or f"ground_truth_{s}.json"

    for c in load_registry(args.src):
        if c["name"] == args.name:
            raise SystemExit(f"clip {args.name!r} is already registered")
        if c["results_dir"] == results_dir:
            raise SystemExit(f"results dir {results_dir!r} is already used by "
                             f"{c['name']!r} — pass --results-dir")

    print(f"clip      : {args.name}")
    print(f"video     : {args.video}")
    print(f"results   : {results_dir}")
    print(f"annotation: {gt_file}")
    print(f"footage   : {args.footage}   split: {args.split}\n")

    # 1. vision pipeline -> features.csv
    if args.skip_pipeline:
        if not os.path.exists(os.path.join(results_dir, "features.csv")):
            raise SystemExit(f"--skip-pipeline given but {results_dir}/features.csv "
                             "does not exist")
        print("[run] skipped (--skip-pipeline)\n")
    else:
        run_pipeline(args, results_dir)

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    # 2. effective fps, measured from the run
    fps = effective_fps(results_dir)
    print(f"[fps] effective processing rate: {fps}")

    # 3. annotation stub, if there isn't one yet
    gt_path = os.path.join(args.src, gt_file)
    if os.path.exists(gt_path):
        print(f"[gt]  {gt_file} already exists — leaving it alone")
    else:
        import cv2
        cap = cv2.VideoCapture(args.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / src_fps, 2)
        cap.release()
        with open(gt_path, "w", encoding="utf-8") as fh:
            json.dump({
                "video": os.path.basename(args.video),
                "annotator": "",
                "annotation_method": "",
                "video_duration": duration,
                "fps": round(src_fps, 3),
                "steps": [],
                "_todo": ("Empty on purpose. Annotate with annotate.py review (or "
                          "contact-sheet + hand editing), then run annotate.py "
                          "validate. See ANNOTATION_PROTOCOL.md."),
            }, fh, indent=2)
        print(f"[gt]  wrote empty annotation stub {gt_file} ({duration}s)")

    # 4. register
    register({
        "name": args.name,
        "video": os.path.basename(args.video),
        "results_dir": results_dir,
        "ground_truth": gt_file,
        "fps": fps,
        "footage": args.footage,
        "split": args.split,
    }, args.src)

    print(f"\n[registry] {summary(args.src)}")
    print("\nNext:")
    print(f"  python annotate.py contact-sheet --video {args.video} --step 1.0")
    print(f"  python annotate.py review --video {args.video} "
          f"--output {gt_file} --annotator \"Your Name\"")
    print(f"  python annotate.py validate --ground-truth {gt_file} --video {args.video}")
    print("  python evaluate_extended.py --src .")
    print("\nUntil the annotation has steps, this clip contributes no metrics.")


if __name__ == "__main__":
    main()
