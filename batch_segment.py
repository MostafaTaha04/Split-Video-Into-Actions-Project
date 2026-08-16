"""
batch_segment.py
================
Run the segmentation pipeline over every unregistered video, unattended.

Purpose and limits
------------------
These clips have no ground truth, so nothing here produces an F1 or any other
accuracy number. What it does produce is (a) a choice of demo clip, and (b) the
statistics that *can* be computed without an answer key: segment counts, segment
durations, and whether the run completed at all. That supports a robustness
statement — "processed N unseen clips of varying length and framing without
failure, producing k+-s segments" — and nothing stronger. It is not a substitute
for annotating clips and measuring accuracy.

Usage
-----
    python batch_segment.py                     # every unregistered video
    python batch_segment.py --limit 5           # just the first five
    python batch_segment.py --only ytbuildB     # name filter
    python batch_segment.py --dry-run           # list what would run

Results go to results/batch/<clipname>/. A failure is recorded and the run
continues, so one bad file cannot abort an overnight batch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

def global_config(src="."):
    """Read the reported global configuration from extended_results.json.

    Hardcoding it here caused a 4.6-hour batch to run at threshold 0.50 /
    min-duration 3.0 after the reported configuration had moved to 0.65 / 1.5,
    which made the resulting statistics describe a superseded setting. Reading
    it from the evaluation output means the batch always matches whatever the
    project currently reports.
    """
    path = os.path.join(src, "extended_results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            g = json.load(fh)["global"]
        return str(g["threshold"]), str(g["min_dur"])
    except (OSError, json.JSONDecodeError, KeyError):
        raise SystemExit(
            f"could not read the global configuration from {path}.\n"
            "Run: python evaluate_extended.py --src .   (it writes that file)")


def base_args(src="."):
    thr, md = global_config(src)
    return ["--fps", "10", "--detector", "open_vocab", "--model", "yolov8l-worldv2.pt",
            "--resize", "960x540", "--open-vocab-imgsz", "960", "--threshold", thr,
            "--min-duration", md, "--grip-window", "7", "--object-confidence", "0.10",
            "--no-clips"]


def unregistered(src="."):
    reg = {c["video"] for c in json.load(open(os.path.join(src, "clips.json"),
                                              encoding="utf-8"))["clips"]}
    vids = sorted(glob.glob(os.path.join(src, "split-video-data", "*.mp4")))
    return [v for v in vids if os.path.basename(v) not in reg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default=os.path.join("results", "batch"))
    args = ap.parse_args()

    vids = unregistered(args.src)
    if args.only:
        vids = [v for v in vids if args.only in os.path.basename(v)]
    if args.limit:
        vids = vids[:args.limit]
    if not vids:
        raise SystemExit("no unregistered videos matched")

    thr, md = global_config(args.src)
    print(f"{len(vids)} clip(s) to process -> {args.outdir}/")
    print(f"using the reported global configuration: threshold {thr}, min-duration {md}")
    if args.dry_run:
        for v in vids:
            print("   ", os.path.basename(v))
        return

    os.makedirs(args.outdir, exist_ok=True)
    summary, t_all = [], time.time()

    for i, v in enumerate(vids, 1):
        name = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(args.outdir, name)
        done = os.path.join(out, "segmentation_results.json")
        if os.path.exists(done):
            print(f"[{i}/{len(vids)}] {name}  (already done, skipping)")
        else:
            print(f"[{i}/{len(vids)}] {name} ...", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, "main.py", "--video", v, "--output", out]
                               + base_args(args.src), capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                print(f"    FAILED ({time.time()-t0:.0f}s): {' | '.join(tail)}")
                summary.append({"clip": name, "ok": False,
                                "error": " | ".join(tail)})
                continue
            print(f"    ok ({time.time()-t0:.0f}s)")

        try:
            with open(done, encoding="utf-8") as fh:
                d = json.load(fh)
            segs = d.get("segments", [])
            durs = [s["end_time"] - s["start_time"] for s in segs]
            summary.append({"clip": name, "ok": True, "n_segments": len(segs),
                            "duration_s": round(segs[-1]["end_time"], 2) if segs else 0,
                            "mean_segment_s": round(statistics.mean(durs), 2) if durs else 0,
                            "min_segment_s": round(min(durs), 2) if durs else 0})
        except (OSError, json.JSONDecodeError, KeyError) as e:
            summary.append({"clip": name, "ok": False, "error": f"unreadable result: {e}"})

    ok = [s for s in summary if s["ok"]]
    path = os.path.join(args.outdir, "batch_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"n_clips": len(summary), "n_ok": len(ok),
                   "threshold": global_config(args.src)[0],
                   "min_duration": global_config(args.src)[1],
                   "note": "No ground truth for these clips: these are descriptive "
                           "statistics only, not accuracy measurements.",
                   "clips": summary}, fh, indent=2)

    print(f"\n{'clip':34s} {'segs':>5s} {'dur(s)':>8s} {'mean seg':>9s}")
    print("-" * 60)
    for s in summary:
        if s["ok"]:
            print(f"{s['clip']:34s} {s['n_segments']:5d} {s['duration_s']:8.1f} "
                  f"{s['mean_segment_s']:9.2f}")
        else:
            print(f"{s['clip']:34s}  FAILED")

    if ok:
        n = [s["n_segments"] for s in ok]
        m = [s["mean_segment_s"] for s in ok]
        print("-" * 60)
        print(f"{len(ok)}/{len(summary)} completed in {(time.time()-t_all)/60:.1f} min")
        print(f"segments per clip : {statistics.mean(n):.1f} "
              f"(sd {statistics.pstdev(n):.1f}, range {min(n)}-{max(n)})")
        print(f"mean segment len  : {statistics.mean(m):.2f}s")
        print(f"\nwrote {path}")
        print("\nReminder: no ground truth for these clips, so none of this measures "
              "accuracy.\nIt supports a robustness statement only.")


if __name__ == "__main__":
    main()
