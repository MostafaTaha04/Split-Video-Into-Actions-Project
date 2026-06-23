"""
evaluate_baselines.py
---------------------
Compare the full segmentation method against naive baselines, using the SAME
metrics as the main evaluator (so the numbers are directly comparable).

Baselines:
  - uniform_oracleK : (G) evenly-spaced boundaries, where G = number of
                      ground-truth boundaries. The strongest "no-model" baseline
                      (it even gets the segment count for free).
  - uniform_modelN  : N evenly-spaced boundaries, where N = number of boundaries
                      the model predicted (fair, count-matched baseline).
  - random_modelN   : N random boundaries (>= min-gap apart), averaged over many
                      trials.

Usage:
  python evaluate_baselines.py --ground-truth ground_truth_cpuplacement.json \
      --results results_cpu_final/segmentation_results.json
  # or give the model boundaries directly:
  python evaluate_baselines.py --ground-truth gt.json --pred "5.2,9.3,14.2"
"""
import argparse
import json
import random

import numpy as np

from evaluator import Evaluator
from utils import MetricsCalculator

TOLERANCES = [0.5, 1.0, 1.5, 2.0, 3.0]


def f1_at(pred, gt, tol):
    return MetricsCalculator.boundary_accuracy(pred, gt, tolerance=tol)["f1_score"]


def uniform_boundaries(duration, n):
    """n interior boundaries splitting [0, duration] into n+1 equal parts."""
    if n <= 0:
        return []
    return [round(duration * (i + 1) / (n + 1), 3) for i in range(n)]


def random_boundaries(duration, n, min_gap, trials=300, seed=0):
    """Average F1 of n random boundaries respecting a minimum gap."""
    rng = random.Random(seed)
    per_tol = {t: [] for t in TOLERANCES}
    if n <= 0:
        return {t: 0.0 for t in TOLERANCES}
    for _ in range(trials):
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 1000:
            attempts += 1
            c = rng.uniform(min_gap, duration - min_gap)
            if all(abs(c - p) >= min_gap for p in pts):
                pts.append(round(c, 3))
        for t in TOLERANCES:
            per_tol[t].append(f1_at(sorted(pts), GT, t))
    return {t: float(np.mean(v)) for t, v in per_tol.items()}


def row(name, pred):
    return name, {t: f1_at(pred, GT, t) for t in TOLERANCES}


def main():
    global GT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", "-g", required=True)
    ap.add_argument("--results", help="Path to a segmentation_results.json")
    ap.add_argument("--pred", help="Comma-separated predicted boundary times (alternative to --results)")
    ap.add_argument("--min-gap", type=float, default=2.5, help="Min gap for random baseline")
    args = ap.parse_args()

    ev = Evaluator(args.ground_truth)
    GT = list(ev.gt_boundaries)
    gt_segments = ev.gt_segments
    duration = max(e for _, e in gt_segments) if gt_segments else 0.0

    if args.results:
        d = json.load(open(args.results, encoding="utf-8"))
        pred = sorted(float(b["timestamp"]) for b in d.get("boundaries", []))
        vi = d.get("video_info", {})
        for k in ("duration_seconds", "duration", "total_duration"):
            if k in vi:
                duration = max(duration, float(vi[k]))
        if d.get("segments"):
            duration = max(duration, max(float(s["end_time"]) for s in d["segments"]))
    elif args.pred:
        pred = sorted(float(x) for x in args.pred.split(",") if x.strip())
    else:
        raise SystemExit("Provide --results or --pred")

    G = len(GT)
    N = len(pred)

    results = [
        row("Full method", pred),
        row(f"Uniform (oracle K={G})", uniform_boundaries(duration, G)),
        row(f"Uniform (model N={N})", uniform_boundaries(duration, N)),
    ]
    rand = random_boundaries(duration, N, args.min_gap)
    results.append((f"Random (N={N}, avg)", rand))

    # ---- print comparison table ----
    print("=" * 72)
    print(f"BASELINE COMPARISON  (F1 by tolerance)   GT boundaries={G}, duration={duration:.1f}s")
    print("=" * 72)
    header = f"{'Method':28s}" + "".join(f"{str(t)+'s':>8s}" for t in TOLERANCES)
    print(header)
    print("-" * 72)
    for name, scores in results:
        line = f"{name:28s}" + "".join(f"{scores[t]:>8.3f}" for t in TOLERANCES)
        print(line)
    print("=" * 72)
    print("Full method should beat the uniform/random baselines, especially at")
    print("the tighter tolerances (0.5-1.5s), to demonstrate it adds real value.")


if __name__ == "__main__":
    main()