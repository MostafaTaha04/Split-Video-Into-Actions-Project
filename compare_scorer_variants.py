"""
compare_scorer_variants.py
==========================
Reproduces the comparison that led to the appearance scorer: every variant
tried against the over-segmentation problem, under the project's standard
protocol (one global configuration chosen on the dev clips, leave-one-clip-out
over the dev clips, held-out clip at the frozen global configuration).

All variants share evaluate_extended's post-processing and metric. Their
grids are listed below. Variants with more free parameters get larger grids,
which flatters them in the global fit -- leave-one-clip-out is the fair column.

    python compare_scorer_variants.py --src .   # -> scorer_variants.json

Needs appearance.npz for every clip (python appearance.py --src .).

Outcome (see README, "Appearance scorer"): appearance novelty alone has the
best leave-one-clip-out mean and held-out score; blends with the rule score,
top-K selection with a step-length prior, appearance-based merging of rule
boundaries and a stricter peak prominence all generalise worse. The held-out
clip was scored for every variant, so it is not untouched for this selection.
"""
import argparse
import json
import os
from itertools import product

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import evaluate_extended as ee
from clip_registry import clean_names, dev_clips, hand_heldout_clips

THR = [round(x, 2) for x in np.arange(0.45, 0.81, 0.05)]
MD = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def _peaks(score, d, thr, md, prom=0.08):
    """ee.peaks_from_score with a configurable prominence."""
    fps, ts, fi = d["fps"], d["ts"], d["fi"]
    if prom == 0.08:
        return ee.peaks_from_score(score, fi, ts, fps, thr, md)
    mf = max(1, int(md * fps))
    s = gaussian_filter1d(np.asarray(score, float), 2.0)
    pk, _ = find_peaks(s, height=thr, distance=mf, prominence=prom)
    return [float(ts[i]) for i in pk if ts[i] - ts[0] >= md and ts[-1] - ts[i] >= md]


def _topk(score, d, L, md):
    """Keep the K strongest peaks, K = duration / typical step length L."""
    s = gaussian_filter1d(np.asarray(score, float), 2.0)
    ts = d["ts"]
    pk, _ = find_peaks(s, distance=max(1, int(md * d["fps"])))
    cand = sorted(((s[i], ts[i]) for i in pk
                   if ts[i] - ts[0] >= md and ts[-1] - ts[i] >= md), reverse=True)
    k = max(0, int(round(d["dur"] / L)) - 1)
    return sorted(float(t) for _, t in cand[:k])


def _merge(d, bounds, tau):
    """Greedily remove the boundary between the two most similar neighbouring
    segments (mean appearance) until all neighbours differ by >= tau."""
    bounds = sorted(bounds)
    while bounds:
        edges = [d["ts"][0]] + bounds + [d["ts"][-1] + 1e-6]
        means = [d["desc"][(d["ts"] >= a) & (d["ts"] < b)].mean(0)
                 for a, b in zip(edges[:-1], edges[1:])]
        dist = [1 - means[i] @ means[i + 1] /
                (np.linalg.norm(means[i]) * np.linalg.norm(means[i + 1]) + 1e-9)
                for i in range(len(bounds))]
        j = int(np.argmin(dist))
        if dist[j] >= tau:
            break
        bounds.pop(j)
    return bounds


def variants():
    rule = lambda d: d["rule"]  # noqa: E731
    app = lambda d: d["app"]    # noqa: E731
    return {
        "rule (current default)": (
            [dict(thr=t, md=m) for t, m in product(THR, MD)],
            lambda d, p: _peaks(rule(d), d, p["thr"], p["md"])),
        "rule + stricter prominence": (
            [dict(thr=t, md=m, prom=q) for t, m, q in product(THR, MD, (0.08, 0.15, 0.25, 0.35))],
            lambda d, p: _peaks(rule(d), d, p["thr"], p["md"], p["prom"])),
        "rule, top-K step-length prior": (
            [dict(L=L, md=m) for L, m in product((4, 5, 6, 7, 8, 10), (1.5, 2.0, 3.0))],
            lambda d, p: _topk(rule(d), d, p["L"], p["md"])),
        "rule + appearance merging": (
            [dict(thr=t, md=m, tau=u) for t, m, u in product(THR, MD, (0.0, 0.005, 0.01, 0.02, 0.04, 0.08))],
            lambda d, p: _merge(d, _peaks(rule(d), d, p["thr"], p["md"]), p["tau"])),
        "blend 0.5*rule + 0.5*appearance": (
            [dict(thr=t, md=m) for t, m in product(THR, MD)],
            lambda d, p: _peaks(0.5 * rule(d) + 0.5 * app(d), d, p["thr"], p["md"])),
        "max(rule, appearance)": (
            [dict(thr=t, md=m) for t, m in product(THR, MD)],
            lambda d, p: _peaks(np.maximum(rule(d), app(d)), d, p["thr"], p["md"])),
        "appearance, top-K step-length prior": (
            [dict(L=L, md=m) for L, m in product((4, 5, 6, 7, 8, 10), (1.5, 2.0, 3.0))],
            lambda d, p: _topk(app(d), d, p["L"], p["md"])),
        "appearance (adopted as --scorer appearance)": (
            [dict(thr=t, md=m) for t, m in product(THR, MD)],
            lambda d, p: _peaks(app(d), d, p["thr"], p["md"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    args = ap.parse_args()
    dev, held = dev_clips(args.src), hand_heldout_clips(args.src)
    clean = clean_names(args.src, "dev")
    data = {}
    for name, (rd, gtf, fps) in {**dev, **held}.items():
        F = ee.load_features(os.path.join(rd, "features.csv")
                             if os.path.isdir(rd) else
                             os.path.join(args.src, "results", rd, "features.csv"))
        desc = ee.load_appearance(args.src, rd, F)
        if desc is None:
            raise SystemExit(f"appearance.npz missing for {name}: run python appearance.py --src .")
        data[name] = dict(gt=ee.gt_boundaries(args.src, gtf), fps=fps, dur=F[-1]["timestamp"],
                          ts=np.array([r["timestamp"] for r in F]),
                          fi=[r["frame_idx"] for r in F], desc=desc,
                          rule=ee.boundary_score(F, fps),
                          app=ee.appearance_boundary_score(desc, fps))

    def f1(n, fn, p, tol=1.0):
        return ee.f1(fn(data[n], p), data[n]["gt"], tol)

    out = {}
    for vname, (grid, fn) in variants().items():
        mean = lambda names, p: np.mean([f1(n, fn, p) for n in names])  # noqa: E731
        g = max(grid, key=lambda p: mean(list(dev), p))
        loo = {}
        for test in dev:
            p = max(grid, key=lambda q: mean([n for n in dev if n != test], q))
            loo[test] = round(f1(test, fn, p), 3)
        # Ties go to the first configuration in grid order and means are
        # rounded with numpy, both exactly as in evaluate_extended.
        out[vname] = {
            "grid_size": len(grid),
            "global_params": g,
            "global_dev_mean_f1_1s": round(mean(list(dev), g), 3),
            "loo_f1_1s": loo,
            "loo_mean_all": float(np.round(np.mean(list(loo.values())), 3)),
            # None when no development clip is a continuous recording
            "loo_mean_clean": (float(np.round(np.mean([loo[n] for n in clean]), 3))
                               if clean else None),
            "heldout_f1_1s": {n: round(f1(n, fn, g), 3) for n in held},
            "n_pred_heldout": {n: len(fn(data[n], g)) for n in held},
        }
        r = out[vname]
        lc = "" if r["loo_mean_clean"] is None else f"LOO(clean) {r['loo_mean_clean']:.3f}  "
        print(f"{vname:45s} LOO(all) {r['loo_mean_all']:.3f}  {lc}"
              f"held-out {list(r['heldout_f1_1s'].values())}  grid {r['grid_size']}")
    with open(os.path.join(args.src, "scorer_variants.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
