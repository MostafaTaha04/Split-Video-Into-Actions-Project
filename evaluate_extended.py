"""
evaluate_extended.py
---------------------
Extended, reproducible evaluation of the segmentation method that runs WITHOUT
re-executing the heavy vision pipeline. Every run already saves a per-frame
``features.csv``; this script replays only the (fast) boundary-detection stage
on those saved features, so it can sweep parameters and run cross-validation in
seconds.

It reproduces, for the four project clips:

  1. A sanity check that re-segmenting the saved features reproduces the saved
     boundaries exactly (guards against drift in the boundary logic).
  2. A single GLOBAL configuration (no per-clip tuning) chosen by grid search.
  3. LEAVE-ONE-CLIP-OUT cross-validation (tune on 3 clips, test on the 4th) ->
     an honest generalisation estimate with no test-set tuning.
  4. SENSITIVITY of F1 to the boundary threshold and minimum-segment duration.
  5. ANNOTATION ROBUSTNESS: F1 under random jitter of the ground-truth
     boundaries (a quantitative stand-in for inter-annotator variation).
  6. A strong CHANGE-POINT baseline (ruptures, if installed) and the
     uniform/random baselines, all under the identical metric.
  7. A FUSION ABLATION: the F1 impact of removing each boundary-score cue.

Figures are written to ``figures/`` and all numbers to ``extended_results.json``.

Usage:
  python evaluate_extended.py --src .            # repo root containing results_*/
  python evaluate_extended.py --src . --no-figures

The boundary-detection logic here mirrors temporal_segmenter.TemporalSegmenter;
the sanity check (step 1) asserts the two agree. The core metric code lives in
utils.MetricsCalculator and is covered by tests/test_metrics.py.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

# Clips: name -> (results_dir, ground_truth_json, effective_fps)
CLIPS = {
    "Cooling fan": ("results_coolingfan_v2run", "ground_truth_coolingfan_v2.json", 10.0),
    "CPU":         ("results_cpu_final",        "ground_truth_cpuplacement.json",  12.5),
    "RAM":         ("results_ram_final",        "ground_truth_raminstallation.json", 14.985),
    "Cable":       ("results_cable_final",      "ground_truth_cableconnection.json", 14.985),
}
CLEAN = ["Cooling fan", "CPU"]
THRESHOLDS = [round(x, 2) for x in np.arange(0.45, 0.81, 0.05)]
MIN_DURS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


# ----------------------------- data loading -----------------------------
def load_features(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "None") else 0.0) for k, v in r.items()})
    return rows


def gt_boundaries(src, gtf):
    with open(os.path.join(src, gtf)) as fh:
        steps = json.load(fh).get("steps", [])
    ends = [float(s["end"]) for s in steps]
    return ends[:-1]  # interior step ends


# ----------------------- boundary-detection (mirror) ---------------------
def _normalize(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, 5), np.percentile(values, 95)
    if hi - lo < 1e-6:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def boundary_score(F, fps, drop=None):
    g = lambda k: np.array([r[k] for r in F], dtype=float)
    comp = {}
    comp["transition"] = g("transition_score")
    act = g("activity_level")
    comp["activity_change"] = 0.80 * _normalize(np.abs(np.diff(act, prepend=act[0])))
    fl = _normalize(g("flow_magnitude"))
    comp["flow_change"] = 0.55 * _normalize(np.abs(np.diff(fl, prepend=fl[0])))
    h = g("hands_present")
    comp["hand_change"] = 0.55 * np.minimum(np.abs(np.diff(h, prepend=h[0])), 1.0)
    it = g("num_interactions")
    comp["interaction_change"] = 0.55 * np.minimum(np.abs(np.diff(it, prepend=it[0])), 1.0)
    tc = g("num_tools")
    comp["tool_count_change"] = 0.45 * np.minimum(np.abs(np.diff(tc, prepend=tc[0])) / 3.0, 1.0)
    use = [v for k, v in comp.items() if k != drop]
    score = np.maximum.reduce(use)
    warm = min(len(score), max(3, int(0.5 * fps)))
    score[:warm] = 0.0
    return np.clip(score, 0.0, 1.0)


def segment(F, fps, threshold=0.70, min_dur=2.0, sigma=2.0, drop=None):
    if len(F) < max(1, int(min_dur * fps)):
        return []
    mf = max(1, int(min_dur * fps))
    s = gaussian_filter1d(boundary_score(F, fps, drop=drop), sigma)
    pk, pr = find_peaks(s, height=threshold, distance=mf, prominence=0.08)
    B = [(int(F[i]["frame_idx"]), float(F[i]["timestamp"]), float(min(hh, 1.0)))
         for i, hh in zip(pk, pr.get("peak_heights", []))]
    if B:
        filt = [B[0]]
        for b in B[1:]:
            if b[0] - filt[-1][0] >= mf:
                filt.append(b)
            elif b[2] > filt[-1][2]:
                filt[-1] = b
        B = filt
    if len(B) > 1:
        mw = max(1, mf // 2)
        mg = [B[0]]
        for b in B[1:]:
            if b[0] - mg[-1][0] < mw:
                if b[2] > mg[-1][2]:
                    mg[-1] = b
            else:
                mg.append(b)
        B = mg
    vs, ve = F[0]["timestamp"], F[-1]["timestamp"]
    return [b[1] for b in B if (b[1] - vs >= min_dur) and (ve - b[1] >= min_dur)]


# ------------------------------- metric ----------------------------------
def f1(pred, gt, tol):
    """F1 with optimal one-to-one matching (Hungarian), matching utils.py."""
    if not pred or not gt:
        return 0.0
    c = np.abs(np.array(pred)[:, None] - np.array(gt)[None, :])
    big = c.max() * (c.size + 1) + 1
    rows, cols = linear_sum_assignment(np.where(c <= tol, c, big))
    tp = sum(1 for a, b in zip(rows, cols) if c[a, b] <= tol)
    p, r = tp / len(pred), tp / len(gt)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ----------------------------- baselines ---------------------------------
def uniform(duration, n):
    return [round(duration * (i + 1) / (n + 1), 3) for i in range(n)] if n > 0 else []


def random_avg(duration, n, gt, tol, min_gap=2.5, trials=300, seed=0):
    if n <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        pts, att = [], 0
        while len(pts) < n and att < 1000:
            att += 1
            c = rng.uniform(min_gap, duration - min_gap)
            if all(abs(c - p) >= min_gap for p in pts):
                pts.append(c)
        out.append(f1(sorted(pts), gt, tol))
    return float(np.mean(out))


def changepoint(F, fps, K, min_dur=2.0):
    try:
        import ruptures as rpt
    except Exception:
        return None
    if K <= 0:
        return []
    sig = np.array([[r["activity_level"], r["flow_magnitude"], r["hands_present"]] for r in F])
    ts = np.array([r["timestamp"] for r in F])
    if len(sig) <= K + 1:
        return []
    try:
        bk = rpt.Dynp(model="l2", min_size=max(2, int(min_dur * fps))).fit(sig).predict(n_bkps=K)
    except Exception:
        return []
    return [float(ts[min(b, len(ts) - 1)]) for b in bk[:-1]]


# ------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".", help="Repo root containing results_*/ and ground_truth_*.json")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    src = args.src

    data = {}
    for name, (rd, gtf, fps) in CLIPS.items():
        F = load_features(os.path.join(src, rd, "features.csv"))
        data[name] = (F, gt_boundaries(src, gtf), fps,
                      max(e for _, _, _ in [(0, 0, 0)] for e in [F[-1]["timestamp"]]))

    res = {}

    # 1) sanity: reproduce saved cooling-fan boundaries at its run config (0.55/2.5)
    F = data["Cooling fan"][0]
    repro = [round(b, 3) for b in segment(F, 10.0, 0.55, 2.5, 2.0)]
    with open(os.path.join(src, "results_coolingfan_v2run", "segmentation_results.json")) as fh:
        saved = [round(float(b["timestamp"]), 3) for b in json.load(fh)["boundaries"]]
    assert repro == saved, f"re-segmentation drift!\n repro={repro}\n saved={saved}"
    print("[sanity] re-segmentation reproduces saved boundaries: OK")

    def cf1(name, thr, md, tol=1.0):
        F, gt, fps, _ = data[name]
        return f1(segment(F, fps, thr, md), gt, tol)

    # 2) global config
    from itertools import product
    best = max(((np.mean([cf1(n, t, m) for n in CLIPS]), t, m) for t, m in product(THRESHOLDS, MIN_DURS)),
               key=lambda x: x[0])
    gmean, gthr, gmd = best
    res["global"] = {"threshold": gthr, "min_dur": gmd, "mean_f1_1s": round(gmean, 3),
                     "per_clip": {n: {"f1_1s": round(cf1(n, gthr, gmd), 3),
                                      "f1_3s": round(cf1(n, gthr, gmd, 3.0), 3)} for n in CLIPS}}

    # 3) leave-one-clip-out
    loo = {}
    for test in CLIPS:
        train = [n for n in CLIPS if n != test]
        _, t, m = max(((np.mean([cf1(n, t, m) for n in train]), t, m)
                       for t, m in product(THRESHOLDS, MIN_DURS)), key=lambda x: x[0])
        loo[test] = {"thr": t, "min_dur": m,
                     "f1_1s": round(cf1(test, t, m), 3), "f1_3s": round(cf1(test, t, m, 3.0), 3)}
    res["loo"] = loo
    res["loo_mean_clean_f1_1s"] = round(np.mean([loo[n]["f1_1s"] for n in CLEAN]), 3)

    # 4) sensitivity (threshold @min=2.0)
    sens = {}
    for t in THRESHOLDS:
        vals = {n: cf1(n, t, 2.0) for n in CLIPS}
        sens[t] = {"clean": round(np.mean([vals[n] for n in CLEAN]), 3),
                   "all": round(np.mean(list(vals.values())), 3)}
    res["sensitivity_threshold"] = sens

    # 5) annotation robustness
    rng = np.random.default_rng(0)
    rob = {}
    for n in CLIPS:
        F, gt, fps, _ = data[n]
        pred = segment(F, fps, gthr, gmd)
        entry = {"base": round(f1(pred, gt, 1.0), 3)}
        for sd in (0.25, 0.5):
            sc = [f1(pred, [g + rng.normal(0, sd) for g in gt], 1.0) for _ in range(300)]
            entry[f"jitter_{sd}"] = [round(float(np.mean(sc)), 3), round(float(np.std(sc)), 3)]
        rob[n] = entry
    res["annotation_robustness"] = rob

    # 6) baselines
    base = {}
    for n in CLIPS:
        F, gt, fps, dur = data[n]
        G = len(gt)
        method = segment(F, fps, gthr, gmd)
        N = len(method)
        cp = changepoint(F, fps, G)
        base[n] = {
            "method": [round(f1(method, gt, 1.0), 3), round(f1(method, gt, 3.0), 3)],
            "uniform_oracleK": [round(f1(uniform(dur, G), gt, 1.0), 3), round(f1(uniform(dur, G), gt, 3.0), 3)],
            "random": [round(random_avg(dur, N, gt, 1.0), 3), round(random_avg(dur, N, gt, 3.0), 3)],
            "changepoint_oracleK": (None if cp is None else
                                    [round(f1(cp, gt, 1.0), 3), round(f1(cp, gt, 3.0), 3)]),
        }
    res["baselines"] = base

    # 7) fusion ablation (clean clips)
    abl = {}
    for drop in ["none", "transition", "activity_change", "flow_change",
                 "hand_change", "interaction_change", "tool_count_change"]:
        vals = []
        for n in CLEAN:
            F, gt, fps, _ = data[n]
            pred = segment(F, fps, gthr, gmd, drop=(None if drop == "none" else drop))
            vals.append(f1(pred, gt, 1.0))
        abl[drop] = round(float(np.mean(vals)), 3)
    res["fusion_ablation_clean_f1_1s"] = abl

    json.dump(res, open(os.path.join(src, "extended_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    if not args.no_figures:
        make_figures(src, res, data)


def make_figures(src, res, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(src, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Sensitivity curve
    ts = sorted(res["sensitivity_threshold"])
    plt.figure(figsize=(6, 4))
    plt.plot(ts, [res["sensitivity_threshold"][t]["clean"] for t in ts], "o-", label="Clean clips")
    plt.plot(ts, [res["sensitivity_threshold"][t]["all"] for t in ts], "s--", label="All clips")
    plt.axvline(res["global"]["threshold"], color="gray", ls=":", label="Chosen threshold")
    plt.xlabel("Boundary threshold"); plt.ylabel("Mean F1 @1.0s")
    plt.title("Sensitivity to boundary threshold (min_dur=2.0s)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sensitivity_threshold.png"), dpi=150); plt.close()

    # Baseline comparison (clean clips, F1@1.0)
    methods = ["method", "changepoint_oracleK", "uniform_oracleK", "random"]
    labels = ["Full method", "Change-point", "Uniform (oracle K)", "Random"]
    x = np.arange(len(CLEAN)); w = 0.2
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        vals = [res["baselines"][c][m][0] if res["baselines"][c][m] else 0 for c in CLEAN]
        plt.bar(x + (i - 1.5) * w, vals, w, label=labels[i])
    plt.xticks(x, CLEAN); plt.ylabel("F1 @1.0s"); plt.ylim(0, 1)
    plt.title("Method vs baselines (clean clips, tight tolerance)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "baseline_comparison.png"), dpi=150); plt.close()
    print(f"[figures] written to {fig_dir}/")


if __name__ == "__main__":
    main()
