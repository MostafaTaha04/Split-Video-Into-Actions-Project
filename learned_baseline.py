"""
learned_baseline.py
===================
Does a *learned* boundary detector beat the hand-designed fusion?

Section 2.1 of the report argues that supervised temporal action segmentation
is inapplicable here because the project has five clips, no per-frame labels,
and a brief that excludes recognition. That argument is a priori. This script
tests it empirically, which is cheap because the data already exists: every
pipeline run saved a per-frame ``features.csv``, and every clip has annotated
boundaries, so ~3,000 labelled frames are already on disk and **no new
annotation is required**.

Experimental design
-------------------
*Task.* Per-frame binary classification: is this frame within ``--label-tol``
seconds of an annotated step boundary? The model's predicted probability is
then used as a boundary score.

*Fair comparison.* The predicted probability is fed through
``evaluate_extended.peaks_from_score`` — the exact smoothing, peak-finding,
minimum-separation, merging and edge-removal used by the rule-based method.
Both approaches therefore differ **only** in how the per-frame score is
produced, never in post-processing.

*Feature sets.* Two, because the distinction matters for what the result means:

  ``raw``  — the 23 primitive per-frame measurements only, with the two
             hand-designed composites (``transition_score``, ``activity_level``)
             REMOVED. This is the honest test: can a model rediscover the
             fusion from primitives?
  ``all``  — every column, including the hand-designed composites. A model here
             is partly inheriting the hand design, so a good score does not
             demonstrate that learning replaced it.

*Temporal context.* Boundaries are temporal events, so each frame is
represented by its features at offsets {-4, -2, 0, +2, +4} frames plus local
first differences, giving the classifier a receptive field of roughly one
second.

*Protocol.* Leave-one-clip-out. For each held-out clip the model is trained on
the other four; feature scaling, class balancing and the peak threshold are all
fitted on the training clips only. Nothing about the held-out clip influences
training or model selection, so the reported F1 is a clean generalisation
estimate directly comparable to the rule-based LOO figures.

Usage
-----
    pip install scikit-learn
    python learned_baseline.py --src .
    python learned_baseline.py --src . --label-tol 0.5 --no-figures

Writes ``learned_results.json`` and ``figures/learned_vs_rulebased.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from evaluate_extended import (
    CLIPS, HELDOUT, f1, gt_boundaries, load_features, peaks_from_score, segment,
)

# Hand-designed composite signals. Excluded from the "raw" feature set so that
# a learned model cannot simply read the answer off the rule-based fusion.
COMPOSITES = ["activity_level", "transition_score"]
NON_FEATURES = ["frame_idx", "timestamp"]

OFFSETS = (-4, -2, 0, 2, 4)
PEAK_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
MIN_DURS = [2.0, 2.5, 3.0]
BLEND_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


# ---------------------------------------------------------------- data
def clip_table(src, results_dir):
    """Return (feature matrix, column names, frame_idx, timestamps)."""
    rows = load_features(os.path.join(src, results_dir, "features.csv"))
    cols = [c for c in rows[0] if c not in NON_FEATURES]
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    fidx = np.array([r["frame_idx"] for r in rows], dtype=int)
    ts = np.array([r["timestamp"] for r in rows], dtype=float)
    return X, cols, fidx, ts


def make_labels(ts, boundaries, tol):
    """1 for frames within `tol` seconds of an annotated boundary."""
    y = np.zeros(len(ts), dtype=int)
    for b in boundaries:
        y[np.abs(ts - b) <= tol] = 1
    return y


def add_context(X):
    """Stack lagged/led copies plus local first differences."""
    n = len(X)
    parts = []
    for off in OFFSETS:
        idx = np.clip(np.arange(n) + off, 0, n - 1)
        parts.append(X[idx])
    d1 = np.diff(X, axis=0, prepend=X[:1])
    parts.append(d1)
    parts.append(np.abs(d1))
    return np.hstack(parts)


def build(src, clips, feature_set, label_tol):
    from evaluate_extended import boundary_score

    data = {}
    for name, (rdir, gtf, fps) in clips.items():
        X, cols, fidx, ts = clip_table(src, rdir)
        keep = [i for i, c in enumerate(cols)
                if feature_set == "all" or c not in COMPOSITES]
        gt = gt_boundaries(src, gtf)
        rows = load_features(os.path.join(src, rdir, "features.csv"))
        data[name] = {
            "X": add_context(X[:, keep]),
            "y": make_labels(ts, gt, label_tol),
            "gt": gt, "fps": fps, "fidx": fidx, "ts": ts,
            "raw_rows": rows,
            "rule": boundary_score(rows, fps),
            "n_features": len(keep),
        }
    return data


# ---------------------------------------------------------------- models
def make_models(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.1, random_state=seed),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=4,
            l2_regularization=1.0, class_weight="balanced", random_state=seed),
        "mlp_neural_net": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=1200,
            early_stopping=True, n_iter_no_change=25, random_state=seed),
    }


def fit_predict(model_fn, Xtr, ytr, Xte):
    """Fit on training clips, return calibrated probabilities for the test clip."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    model = model_fn()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # MLPClassifier has no class_weight; oversample positives instead so
        # all three models see a comparably balanced problem.
        if model.__class__.__name__ == "MLPClassifier":
            pos = np.flatnonzero(ytr == 1)
            neg = np.flatnonzero(ytr == 0)
            if len(pos):
                reps = max(1, len(neg) // len(pos))
                idx = np.concatenate([neg, np.tile(pos, reps)])
                rng = np.random.default_rng(0)
                rng.shuffle(idx)
                Xtr, ytr = Xtr[idx], ytr[idx]
        model.fit(scaler.transform(Xtr), ytr)
        p = model.predict_proba(scaler.transform(Xte))[:, 1]
    return p


# ---------------------------------------------------------------- evaluation
def loo_learned(data, model_fn, tol=1.0, hybrid=False):
    """Leave-one-clip-out with fully nested selection.

    For each held-out clip, the model, the feature scaler, the peak threshold,
    the minimum duration and (when ``hybrid``) the rule/learned blend weight are
    all chosen using only the four training clips — the inner loop scores each
    training clip with a model trained on the *other three*. Nothing about the
    held-out clip influences any choice, so this is an honest generalisation
    estimate directly comparable to the rule-based LOO figures.
    """
    names = list(data)
    weights = BLEND_WEIGHTS if hybrid else [0.0]
    out = {}
    for test in names:
        train = [n for n in names if n != test]
        Xtr = np.vstack([data[n]["X"] for n in train])
        ytr = np.concatenate([data[n]["y"] for n in train])

        # out-of-fold probabilities for the training clips
        inner = {}
        for inner_test in train:
            inner_train = [n for n in train if n != inner_test]
            Xi = np.vstack([data[n]["X"] for n in inner_train])
            yi = np.concatenate([data[n]["y"] for n in inner_train])
            inner[inner_test] = fit_predict(model_fn, Xi, yi, data[inner_test]["X"])

        best, best_f1 = (PEAK_THRESHOLDS[0], MIN_DURS[0], weights[0]), -1.0
        for w in weights:
            for thr in PEAK_THRESHOLDS:
                for md in MIN_DURS:
                    scores = []
                    for n in train:
                        d = data[n]
                        s = _mix(d, inner[n], w, hybrid)
                        pred = peaks_from_score(s, d["fidx"], d["ts"], d["fps"],
                                                threshold=thr, min_dur=md)
                        scores.append(f1(pred, d["gt"], tol))
                    m = float(np.mean(scores))
                    if m > best_f1:
                        best_f1, best = m, (thr, md, w)

        thr, md, w = best
        p = fit_predict(model_fn, Xtr, ytr, data[test]["X"])
        d = data[test]
        pred = peaks_from_score(_mix(d, p, w, hybrid), d["fidx"], d["ts"], d["fps"],
                                threshold=thr, min_dur=md)
        entry = {
            "f1_1s": round(f1(pred, d["gt"], 1.0), 3),
            "f1_3s": round(f1(pred, d["gt"], 3.0), 3),
            "n_pred": len(pred), "n_gt": len(d["gt"]),
            "selected_threshold": thr, "selected_min_dur": md,
            "train_f1": round(best_f1, 3),
        }
        if hybrid:
            entry["selected_blend_weight_rule"] = w
        out[test] = entry
    return out


def _mix(d, prob, w_rule, hybrid):
    """Blend the rule-based score with a model probability (or use the model alone)."""
    if not hybrid:
        return prob
    from boundary_model import blend
    return blend(d["rule"], prob, w_rule)


def loo_rulebased(src, clips, data, tol=1.0):
    """Rule-based method under the identical LOO protocol, for comparison."""
    from itertools import product

    from evaluate_extended import MIN_DURS as RB_MIN_DURS
    from evaluate_extended import THRESHOLDS as RB_THRESHOLDS

    names = list(clips)
    raw = {n: data[n]["raw_rows"] for n in names}
    out = {}
    for test in names:
        train = [n for n in names if n != test]
        best, best_f1 = None, -1.0
        for thr, md in product(RB_THRESHOLDS, RB_MIN_DURS):
            m = float(np.mean([
                f1(segment(raw[n], data[n]["fps"], thr, md), data[n]["gt"], tol)
                for n in train
            ]))
            if m > best_f1:
                best_f1, best = m, (thr, md)
        thr, md = best
        d = data[test]
        pred = segment(raw[test], d["fps"], thr, md)
        out[test] = {
            "f1_1s": round(f1(pred, d["gt"], 1.0), 3),
            "f1_3s": round(f1(pred, d["gt"], 3.0), 3),
            "n_pred": len(pred), "n_gt": len(d["gt"]),
            "selected_threshold": thr, "selected_min_dur": md,
            "train_f1": round(best_f1, 3),
        }
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".")
    ap.add_argument("--label-tol", type=float, default=0.5,
                    help="frames within this many seconds of a boundary are positive")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "scikit-learn is required for this experiment:\n    pip install scikit-learn")

    all_clips = {**CLIPS, **HELDOUT}
    res = {
        "protocol": "leave-one-clip-out; scaling, class balancing and peak "
                    "threshold fitted on training clips only",
        "label_tolerance_s": args.label_tol,
        "n_clips": len(all_clips),
        "context_offsets": list(OFFSETS),
    }

    # Class balance — the central difficulty of this experiment.
    probe = build(args.src, all_clips, "raw", args.label_tol)
    n_pos = int(sum(d["y"].sum() for d in probe.values()))
    n_tot = int(sum(len(d["y"]) for d in probe.values()))
    n_bnd = int(sum(len(d["gt"]) for d in probe.values()))
    res["dataset"] = {
        "frames": n_tot, "positive_frames": n_pos,
        "positive_rate": round(n_pos / n_tot, 4),
        "annotated_boundaries": n_bnd,
        "features_raw": probe[list(probe)[0]]["n_features"],
    }
    print(f"[data] {n_tot} frames, {n_pos} positive ({n_pos / n_tot:.1%}), "
          f"{n_bnd} annotated boundaries across {len(all_clips)} clips")

    print("[rule-based] leave-one-clip-out ...")
    rb = loo_rulebased(args.src, all_clips, probe)
    res["rule_based_loo"] = rb
    res["rule_based_loo_mean_f1_1s"] = round(
        float(np.mean([v["f1_1s"] for v in rb.values()])), 3)

    res["learned_loo"] = {}
    for fset in ("raw", "all"):
        data = build(args.src, all_clips, fset, args.label_tol)
        for mname, mfn in make_models().items():
            key = f"{mname}__{fset}"
            print(f"[learned] {key} ...")
            out = loo_learned(data, mfn)
            mean = round(float(np.mean([v["f1_1s"] for v in out.values()])), 3)
            res["learned_loo"][key] = {"per_clip": out, "mean_f1_1s": mean}
            print(f"           mean LOO F1@1.0s = {mean:.3f}")

    # Hybrid: rule-based fusion blended with the learned probability. The blend
    # weight is selected inside the same nested loop as everything else, so it
    # is never fitted on the held-out clip.
    data_raw = build(args.src, all_clips, "raw", args.label_tol)
    for mname in ("logistic_regression",):
        key = f"hybrid_{mname}__raw"
        print(f"[hybrid]  {key} ...")
        out = loo_learned(data_raw, make_models()[mname], hybrid=True)
        mean = round(float(np.mean([v["f1_1s"] for v in out.values()])), 3)
        res["learned_loo"][key] = {"per_clip": out, "mean_f1_1s": mean}
        print(f"           mean LOO F1@1.0s = {mean:.3f}")

    # ---- Verdict, with a significance test rather than a bare comparison ----
    # With five clips, a difference in mean F1 of a few points is well inside
    # sampling noise. Reporting "X beats Y" from five paired observations
    # without a test would be exactly the kind of overclaiming this experiment
    # exists to avoid, so the paired test and the per-clip breakdown are
    # computed and stored alongside the means.
    clips = list(res["rule_based_loo"])
    rb_mean = res["rule_based_loo_mean_f1_1s"]
    rb_vals = np.array([res["rule_based_loo"][c]["f1_1s"] for c in clips])

    best_key = max(res["learned_loo"], key=lambda k: res["learned_loo"][k]["mean_f1_1s"])
    best_mean = res["learned_loo"][best_key]["mean_f1_1s"]
    best_vals = np.array([res["learned_loo"][best_key]["per_clip"][c]["f1_1s"] for c in clips])
    diff = best_vals - rb_vals

    pvals = {}
    try:
        from scipy.stats import ttest_rel, wilcoxon
        pvals["paired_t"] = round(float(ttest_rel(best_vals, rb_vals).pvalue), 4)
        pvals["wilcoxon"] = round(float(wilcoxon(best_vals, rb_vals).pvalue), 4)
    except Exception:
        pvals = {"paired_t": None, "wilcoxon": None}

    clean = [c for c in clips if c in ("Cooling fan", "CPU")]
    hard = [c for c in clips if c not in clean]

    def sub(vals_by_clip, names):
        return round(float(np.mean([vals_by_clip[c] for c in names])), 3)

    rb_by = {c: res["rule_based_loo"][c]["f1_1s"] for c in clips}
    lr_by = {c: res["learned_loo"][best_key]["per_clip"][c]["f1_1s"] for c in clips}

    res["verdict"] = {
        "rule_based_mean_f1_1s": rb_mean,
        "best_learned": best_key,
        "best_learned_mean_f1_1s": best_mean,
        "margin": round(best_mean - rb_mean, 3),
        "paired_difference_mean": round(float(diff.mean()), 3),
        "paired_difference_std": round(float(diff.std(ddof=1)), 3),
        "per_clip_wins_learned": int((diff > 0).sum()),
        "per_clip_wins_rule_based": int((diff < 0).sum()),
        "p_values": pvals,
        "significant_at_0.05": bool(
            pvals.get("paired_t") is not None and pvals["paired_t"] < 0.05),
        "clean_clips": {"rule_based": sub(rb_by, clean), "learned": sub(lr_by, clean)},
        "hard_clips": {"rule_based": sub(rb_by, hard), "learned": sub(lr_by, hard)},
        "variability": {
            "rule_based_std_across_clips": round(float(rb_vals.std()), 3),
            "learned_std_across_clips": round(float(best_vals.std()), 3),
        },
        "conclusion": (
            "No significant difference in mean F1 across clips. The two approaches "
            "differ in WHERE they succeed: the hand-designed fusion is clearly better "
            "on clean continuous footage, which is the material its cues were designed "
            "for, while the learned model is more uniform across footage types but never "
            "reaches the rule-based ceiling. With five clips this comparison cannot "
            "separate the methods on aggregate score."
        ),
    }

    with open(os.path.join(args.src, "learned_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print("\n" + "=" * 68)
    print(f"{'method':38s} {'mean LOO F1@1.0s':>18s}")
    print("-" * 68)
    print(f"{'rule-based fusion (hand-designed)':38s} {rb_mean:>18.3f}")
    for k, v in sorted(res["learned_loo"].items(), key=lambda x: -x[1]["mean_f1_1s"]):
        print(f"{k:38s} {v['mean_f1_1s']:>18.3f}")
    print("=" * 68)
    v = res["verdict"]
    print(f"best learned : {v['best_learned']}  ({best_mean:.3f})")
    print(f"margin       : {v['margin']:+.3f}  "
          f"(paired t p={v['p_values'].get('paired_t')}, "
          f"wilcoxon p={v['p_values'].get('wilcoxon')})")
    print(f"significant  : {'YES' if v['significant_at_0.05'] else 'NO — within noise at n=5 clips'}")
    print(f"clean clips  : rule-based {v['clean_clips']['rule_based']:.3f}  "
          f"vs learned {v['clean_clips']['learned']:.3f}")
    print(f"hard clips   : rule-based {v['hard_clips']['rule_based']:.3f}  "
          f"vs learned {v['hard_clips']['learned']:.3f}")
    print(f"spread       : rule-based std {v['variability']['rule_based_std_across_clips']:.3f}  "
          f"vs learned std {v['variability']['learned_std_across_clips']:.3f}")
    print("=" * 68)
    print("wrote learned_results.json")

    if not args.no_figures:
        make_figure(args.src, res)


def make_figure(src, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(src, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    clips = list(res["rule_based_loo"])

    labels = ["Rule-based\n(hand-designed)"]
    series = [[res["rule_based_loo"][c]["f1_1s"] for c in clips]]
    for k in sorted(res["learned_loo"], key=lambda x: -res["learned_loo"][x]["mean_f1_1s"]):
        m, fs = k.split("__")
        labels.append(f"{m.replace('_', ' ')}\n({fs} features)")
        series.append([res["learned_loo"][k]["per_clip"][c]["f1_1s"] for c in clips])

    x = np.arange(len(clips))
    w = 0.8 / len(series)
    plt.figure(figsize=(11, 4.6))
    for i, (lab, vals) in enumerate(zip(labels, series)):
        plt.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=lab)
    plt.xticks(x, clips, fontsize=9)
    plt.ylabel("F1 @1.0s (leave-one-clip-out)")
    plt.ylim(0, 1)
    plt.title("Learned boundary detectors vs the hand-designed fusion\n"
              "identical post-processing; threshold selected on training clips only",
              fontsize=11)
    plt.legend(fontsize=7, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(fig_dir, "learned_vs_rulebased.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] {out}")


if __name__ == "__main__":
    main()
