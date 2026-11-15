"""
run_ablations.py
================
Module ablations for every annotated clip, at the evaluated global
configuration -- the way the detector ablation was done.

The earlier flow/scene ablations (results/abl_noflow, abl_noscene) covered
one clip (cooling fan) at an older configuration (threshold 0.50, min-duration
3.0), so they are not comparable with the headline numbers -- and the no-scene
run scored F1 1.000 there, which deserves a proper test. This script re-runs
the pipeline per clip with one module disabled and tabulates F1 against the
saved full-pipeline run of the same clip.

Each run uses main.py's defaults (= the evaluated configuration) plus the
variant's flag, so nothing here can drift from the reported setup.

    python run_ablations.py                       # no-flow + no-scene, all clips
    python run_ablations.py --variants noscene    # one variant
    python run_ablations.py --clips CPU "Cooling fan"
    python run_ablations.py --variants fresh      # full pipeline, reproducibility check
    python run_ablations.py --summarize           # table from finished runs only

Runs are written to results/ablations/<variant>/<results_dir>/ and skipped if
already complete, so the script can be interrupted and resumed. Expect roughly
5-10 minutes per clip per variant on a CPU (YOLO-World dominates).
"""
import argparse
import json
import os
import subprocess
import sys

import evaluate_extended as ee
from clip_registry import dev_clips, hand_heldout_clips, load_registry, resolve_gt, resolve_video

VARIANTS = {
    # "fresh" re-runs the full pipeline with main.py's defaults and no module
    # disabled. Its column should match "full" (the saved runs replayed at the
    # global configuration): that is the end-to-end reproducibility check.
    "fresh": [],
    "noflow": ["--no-flow"],
    "noscene": ["--no-scene"],
    "nodetector": ["--detector", "none"],
}


def run_dir(src, variant, results_dir):
    return os.path.join(src, "results", "ablations", variant, os.path.basename(results_dir))


def run_one(src, clip, variant, extra, report=False):
    out = run_dir(src, variant, clip["results_dir"])
    if os.path.exists(os.path.join(out, "segmentation_results.json")):
        print(f"[done] {variant:10s} {clip['name']}")
        return
    # Process at the rate the saved run used (clips.json "fps"): RAM and Cable
    # were evaluated at 14.985 fps (--fps 15), the others at 10-12.5 fps. Any
    # other rate samples different frames and cannot reproduce the saved run.
    cmd = [sys.executable, os.path.join(src, "main.py"),
           "--video", resolve_video(clip["video"], src),
           "--fps", str(int(round(float(clip["fps"])))),
           "--output", out, "--no-clips",
           "--ground-truth", resolve_gt(clip["ground_truth"], src)] + extra
    if not report:
        cmd.append("--no-report")
    print(f"[run]  {variant:10s} {clip['name']}\n       {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=src)


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _global_config(src):
    try:
        g = _load_json(os.path.join(src, "extended_results.json"))["global"]
        return g["threshold"], g["min_dur"]
    except (OSError, KeyError, ValueError):
        return 0.65, 1.5


def _replay(rd, fps, thr, md):
    """Boundaries of a run re-segmented at (thr, md) from its features.csv.

    Every column is computed this way -- the saved full runs were produced at
    older settings, so their stored boundaries are not comparable, but their
    features are (the threshold only acts after feature extraction)."""
    F = ee.load_features(os.path.join(rd, "features.csv"))
    return ee.segment(F, fps, thr, md)


def summarize(src, variants):
    clips = {**dev_clips(src), **hand_heldout_clips(src)}
    thr, md = _global_config(src)
    rows, table = {}, {}
    for name, (rd, gtf, fps) in clips.items():
        gt = ee.gt_boundaries(src, gtf)
        full = _replay(rd, fps, thr, md)
        row = {"full": round(ee.f1(full, gt, 1.0), 3), "full_n_pred": len(full)}
        for v in variants:
            vd = run_dir(src, v, rd)
            if os.path.exists(os.path.join(vd, "features.csv")):
                b = _replay(vd, fps, thr, md)
                row[v] = round(ee.f1(b, gt, 1.0), 3)
                row[f"{v}_n_pred"] = len(b)
        row["n_gt"] = len(gt)
        rows[name] = row
    cols = ["full"] + list(variants)
    print(f"\nF1 @1.0 s (rule scorer, global configuration threshold {thr}, min-duration {md})")
    print(f"{'clip':22s}" + "".join(f"{c:>12s}" for c in cols))
    for name, row in rows.items():
        print(f"{name:22s}" + "".join(f"{row.get(c, float('nan')):12.3f}" for c in cols))
    for c in cols:
        vals = [r[c] for r in rows.values() if c in r]
        if len(vals) == len(rows):
            table[c] = round(sum(vals) / len(vals), 3)
    print(f"{'mean (all clips)':22s}" + "".join(f"{table.get(c, float('nan')):12.3f}" for c in cols))
    # Appearance scorer: saved descriptors vs a fresh run's descriptors.
    app_rows = {}
    try:
        ag = _load_json(os.path.join(src, "extended_results.json"))["appearance_scorer"]["global"]
        athr, amd = ag["threshold"], ag["min_dur"]
    except (OSError, KeyError, ValueError):
        athr = amd = None
    if athr is not None:
        print(f"\nF1 @1.0 s (appearance scorer, threshold {athr}, min-duration {amd})")
        print(f"{'clip':22s}{'full':>12s}{'fresh':>12s}")
        for name, (rd, gtf, fps) in clips.items():
            gt = ee.gt_boundaries(src, gtf)
            row = {}
            for col, d in (("full", rd), ("fresh", run_dir(src, "fresh", rd))):
                if os.path.exists(os.path.join(d, "appearance.npz")):
                    F = ee.load_features(os.path.join(d, "features.csv"))
                    desc = ee.load_appearance(src, d, F)
                    if desc is not None:
                        row[col] = round(ee.f1(ee.segment_appearance(F, desc, fps, athr, amd), gt, 1.0), 3)
            app_rows[name] = row
            print(f"{name:22s}" + "".join(f"{row.get(c, float('nan')):12.3f}" for c in ("full", "fresh")))

    with open(os.path.join(src, "ablation_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"per_clip": rows, "mean": table, "appearance_per_clip": app_rows,
                   "config": {"threshold": thr, "min_dur": md, "detector": "open_vocab"}},
                  fh, indent=2)
    print("\nwritten: ablation_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    ap.add_argument("--variants", nargs="+", default=["noflow", "noscene"], choices=list(VARIANTS))
    ap.add_argument("--report", action="store_true",
                    help="also write step_report.html for each run (off by default for speed)")
    ap.add_argument("--clips", nargs="+", help="registered clip names (default: all annotated)")
    ap.add_argument("--summarize", action="store_true", help="only tabulate finished runs")
    a = ap.parse_args()
    if not a.summarize:
        annotated = set(dev_clips(a.src)) | set(hand_heldout_clips(a.src))
        for clip in load_registry(a.src):
            if clip["name"] not in annotated or (a.clips and clip["name"] not in a.clips):
                continue
            for v in a.variants:
                run_one(a.src, clip, v, VARIANTS[v], a.report)
    summarize(a.src, a.variants)


if __name__ == "__main__":
    main()
