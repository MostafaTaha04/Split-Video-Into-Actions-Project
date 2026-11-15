"""
verify_all.py
=============
End-to-end check of the whole project on every annotated video.

For every clip in BOTH registries (clips.json: the 5 human-annotated clips;
clips_ai_draft.json: the 25 AI-drafted ones) this script

1. runs the full pipeline (main.py) on the video, with its ground truth, at the
   evaluated configuration and the processing rate the saved run used;
2. compares the fresh per-frame features with the saved run's features;
3. scores both runs with each scorer's frozen global configuration
   (extended_results.json) and checks that the F1 is the same.

A clip passes when the fresh run reproduces the saved one: same features, same
F1 for the rule scorer and for the appearance scorer. That shows the code, the
models, the videos and the annotations all still fit together.

Usage
-----
    python verify_all.py --quick          # smoke test, every clip, ~15-20 min:
                                          #   every video opens, every annotation
                                          #   loads, the pipeline finishes
    python verify_all.py                  # full check, every clip (~5-6 h on CPU;
                                          #   resumable: finished clips are skipped)
    python verify_all.py --only "Cooling fan"   # one clip first (~5 min)
    python verify_all.py --summarize      # re-print the table, run nothing

Fresh runs go to results/verify/ (results/verify_quick/ for --quick). The
annotated output video of each run (~50 MB) is deleted after the comparison
unless --keep-video is given. Results: verify_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import evaluate_extended as ee
from clip_registry import (load_ai_registry, load_registry, resolve_gt,
                           resolve_results, resolve_video)

QUICK_FRAMES = 150


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def configs(src):
    res = _load_json(os.path.join(src, "extended_results.json"))
    g = res["global"]
    a = res.get("appearance_scorer", {}).get("global")
    return (g["threshold"], g["min_dur"]), ((a["threshold"], a["min_dur"]) if a else None)


def all_clips(src):
    def group(c):
        if c["split"] == "dev":
            return "dev"
        return "held out" if "annotation_source" not in c else "held/rev"
    human = [dict(c, group=group(c)) for c in load_registry(src)]
    ai = [dict(c, group="AI draft") for c in load_ai_registry(src)]
    return human + ai


def run_clip(src, clip, out, quick, keep_video):
    if os.path.exists(os.path.join(out, "segmentation_results.json")):
        return "skipped (already done)", 0.0
    cmd = [sys.executable, os.path.join(src, "main.py"),
           "--video", resolve_video(clip["video"], src),
           "--fps", str(int(round(float(clip["fps"])))),
           "--output", out, "--no-clips", "--no-report",
           "--ground-truth", resolve_gt(clip["ground_truth"], src)]
    if quick:
        cmd += ["--max-frames", str(QUICK_FRAMES), "--detector", "none"]
    print(f"\n[run] {clip['name']}  ({clip['group']})\n      " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    if not keep_video:
        vid = os.path.join(out, "annotated_output.mp4")
        if os.path.exists(vid):
            os.remove(vid)
    return ("ok" if rc == 0 else f"FAILED (exit {rc})"), dt


def score(src, rd, F, fps, gt, rule_cfg, app_cfg):
    rule = round(ee.f1(ee.segment(F, fps, *rule_cfg), gt, 1.0), 3)
    app = None
    if app_cfg:
        desc = ee.load_appearance(src, rd, F)
        if desc is not None:
            app = round(ee.f1(ee.segment_appearance(F, desc, fps, *app_cfg), gt, 1.0), 3)
    return rule, app


def compare(src, clip, out, rule_cfg, app_cfg):
    """Saved vs fresh: features and F1 for both scorers."""
    row = {"clip": clip["name"], "group": clip["group"]}
    fresh_csv = os.path.join(out, "features.csv")
    if not os.path.exists(fresh_csv):
        row["status"] = "NOT RUN"
        return row
    saved_rd = resolve_results(clip["results_dir"], src)
    fps = float(clip["fps"])
    gt = ee.gt_boundaries(src, clip["ground_truth"])
    Fs = ee.load_features(os.path.join(saved_rd, "features.csv"))
    Ff = ee.load_features(fresh_csv)
    row["n_gt"] = len(gt)
    row["saved_rule"], row["saved_app"] = score(src, saved_rd, Fs, fps, gt, rule_cfg, app_cfg)
    row["fresh_rule"], row["fresh_app"] = score(src, out, Ff, fps, gt, rule_cfg, app_cfg)
    if len(Fs) != len(Ff):
        row["features"] = f"different length ({len(Fs)} saved vs {len(Ff)} fresh)"
        row["max_feature_diff"] = None
    else:
        cols = [k for k in Fs[0] if k in Ff[0]]
        diff = max(float(np.max(np.abs(np.array([r[k] for r in Fs]) - np.array([r[k] for r in Ff]))))
                   for k in cols)
        row["max_feature_diff"] = round(diff, 6)
        row["features"] = "identical" if diff < 1e-6 else f"differ (max {diff:.4g})"
    same = (row["features"] == "identical" and row["saved_rule"] == row["fresh_rule"]
            and row["saved_app"] == row["fresh_app"])
    row["status"] = "MATCH" if same else "DIFFERENT"
    return row


def smoke(src, clip, out):
    """--quick: did the run finish and write its outputs?"""
    row = {"clip": clip["name"], "group": clip["group"]}
    need = ["features.csv", "segmentation_results.json", "evaluation_report.txt", "timeline.png"]
    missing = [f for f in need if not os.path.exists(os.path.join(out, f))]
    if not os.path.exists(os.path.join(out, "features.csv")):
        row["status"] = "NOT RUN / FAILED"
        return row
    row["frames"] = len(ee.load_features(os.path.join(out, "features.csv")))
    row["n_gt"] = len(ee.gt_boundaries(src, clip["ground_truth"]))
    row["status"] = "OK" if not missing else "missing " + ", ".join(missing)
    return row


def print_table(rows, quick):
    print("\n" + "=" * 96)
    if quick:
        print(f"QUICK SMOKE TEST (first {QUICK_FRAMES} processed frames, detector off)")
        print(f"{'clip':19s}{'group':10s}{'frames':>8s}{'GT bnd':>8s}   status")
        for r in rows:
            print(f"{r['clip']:19s}{r['group']:10s}{r.get('frames', ''):>8}{r.get('n_gt', ''):>8}   {r['status']}")
        ok = sum(r["status"] == "OK" for r in rows)
        print("-" * 96)
        print(f"{ok}/{len(rows)} clips OK")
    else:
        def f(v):
            return "  -  " if v is None else f"{v:.3f}"
        print("FULL CHECK: saved run vs fresh run, F1 @1 s at each scorer's frozen global configuration")
        print(f"{'clip':19s}{'group':10s}{'rule saved':>11s}{'fresh':>7s}{'app saved':>11s}{'fresh':>7s}   features     status")
        for r in rows:
            if "saved_rule" not in r:
                print(f"{r['clip']:19s}{r['group']:10s}{'':36s}   {'':12s} {r['status']}")
                continue
            print(f"{r['clip']:19s}{r['group']:10s}{f(r['saved_rule']):>11s}{f(r['fresh_rule']):>7s}"
                  f"{f(r['saved_app']):>11s}{f(r['fresh_app']):>7s}   {r['features']:12s} {r['status']}")
        ok = sum(r["status"] == "MATCH" for r in rows)
        print("-" * 96)
        print(f"{ok}/{len(rows)} clips reproduce their saved run exactly")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description="End-to-end check on every annotated video.")
    ap.add_argument("--src", default=".")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test: first {QUICK_FRAMES} processed frames, detector off")
    ap.add_argument("--only", nargs="+", help="clip names (default: all 30)")
    ap.add_argument("--skip-ai", action="store_true", help="only the 5 hand-annotated clips (skip AI-drafted ones, reviewed or not)")
    ap.add_argument("--summarize", action="store_true", help="print the table, run nothing")
    ap.add_argument("--keep-video", action="store_true", help="keep each run's annotated video")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    src = a.src
    outdir = a.outdir or os.path.join(src, "results", "verify_quick" if a.quick else "verify")
    rule_cfg, app_cfg = configs(src)
    clips = all_clips(src)
    if a.skip_ai:
        clips = [c for c in clips if c["group"] in ("dev", "held out")]   # hand-annotated
    if a.only:
        unknown = set(a.only) - {c["name"] for c in clips}
        if unknown:
            raise SystemExit(f"unknown clip name(s): {sorted(unknown)}")
        clips = [c for c in clips if c["name"] in a.only]

    print(f"{len(clips)} clip(s) | rule {rule_cfg} | appearance {app_cfg} | output {outdir}")
    timing = {}
    if not a.summarize:
        for i, c in enumerate(clips, 1):
            out = os.path.join(outdir, os.path.basename(resolve_results(c["results_dir"], src)))
            status, dt = run_clip(src, c, out, a.quick, a.keep_video)
            timing[c["name"]] = round(dt, 1)
            print(f"[{i}/{len(clips)}] {c['name']}: {status}" + (f" in {dt / 60:.1f} min" if dt else ""),
                  flush=True)

    rows = []
    for c in clips:
        out = os.path.join(outdir, os.path.basename(resolve_results(c["results_dir"], src)))
        rows.append(smoke(src, c, out) if a.quick else compare(src, c, out, rule_cfg, app_cfg))
    print_table(rows, a.quick)
    name = "verify_quick_results.json" if a.quick else "verify_results.json"
    with open(os.path.join(src, name), "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "minutes_per_clip": {k: round(v / 60, 1) for k, v in timing.items()},
                   "rule_config": rule_cfg, "appearance_config": app_cfg}, fh, indent=2)
    print(f"written: {name}")


if __name__ == "__main__":
    main()
