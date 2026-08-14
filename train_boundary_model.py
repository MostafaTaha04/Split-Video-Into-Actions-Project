"""
train_boundary_model.py
=======================
Train the learned boundary scorer used by ``main.py --scorer learned|hybrid``.

Training data comes free: every pipeline run saves a per-frame ``features.csv``
and every clip has annotated step boundaries, so ~3,400 labelled frames already
exist on disk. No new annotation is required.

What it does
------------
1. Loads the per-frame features and boundary annotations for every clip.
2. Labels a frame positive if it lies within ``--label-tol`` seconds of an
   annotated boundary.
3. Selects the rule/learned blend weight by leave-one-clip-out cross-validation,
   so the shipped default is not fitted on any single clip.
4. Retrains on all clips and saves ``boundary_model.joblib`` plus a JSON
   sidecar recording the blend weight and provenance.

Usage
-----
    pip install scikit-learn
    python train_boundary_model.py --src .
    python train_boundary_model.py --src . --model gradient_boosting --label-tol 0.5

Then:
    python main.py --video clip.mp4 --scorer hybrid ...

Honesty note
------------
Because the training clips are also the evaluation clips, the numbers printed
here are NOT a generalisation estimate. The honest, held-out comparison is
produced by ``learned_baseline.py`` under leave-one-clip-out and is what the
report quotes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from boundary_model import (
    DEFAULT_MODEL_PATH, FEATURE_COLUMNS, OFFSETS, add_context, blend, rows_matrix,
)
from evaluate_extended import (
    CLIPS, HELDOUT, boundary_score, f1, gt_boundaries, load_features, peaks_from_score,
)

BLEND_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
MIN_DURS = [2.0, 2.5, 3.0]


def make_model(name, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1,
                                  random_state=seed)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                              max_depth=4, l2_regularization=1.0,
                                              class_weight="balanced", random_state=seed)
    if name == "mlp_neural_net":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=1200,
                             early_stopping=True, n_iter_no_change=25, random_state=seed)
    raise SystemExit(f"unknown model: {name}")


def fit(model_name, Xtr, ytr):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    model = make_model(model_name)
    if model_name == "mlp_neural_net":
        pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
        if len(pos):
            idx = np.concatenate([neg, np.tile(pos, max(1, len(neg) // len(pos)))])
            np.random.default_rng(0).shuffle(idx)
            Xtr, ytr = Xtr[idx], ytr[idx]
    model.fit(scaler.transform(Xtr), ytr)
    return model, scaler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".")
    ap.add_argument("--out", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--model", default="logistic_regression",
                    choices=["logistic_regression", "gradient_boosting", "mlp_neural_net"])
    ap.add_argument("--label-tol", type=float, default=0.5)
    args = ap.parse_args()

    try:
        import joblib  # noqa: F401
    except ImportError:
        raise SystemExit("pip install scikit-learn")
    import joblib

    clips = {**CLIPS, **HELDOUT}
    data = {}
    for name, (rdir, gtf, fps) in clips.items():
        rows = load_features(os.path.join(args.src, rdir, "features.csv"))
        ts = np.array([r["timestamp"] for r in rows], dtype=float)
        gt = gt_boundaries(args.src, gtf)
        y = np.zeros(len(ts), dtype=int)
        for b in gt:
            y[np.abs(ts - b) <= args.label_tol] = 1
        data[name] = {
            "X": add_context(rows_matrix(rows)), "y": y, "gt": gt, "fps": fps,
            "ts": ts, "fidx": np.array([r["frame_idx"] for r in rows], dtype=int),
            "rule": boundary_score(rows, fps),
        }

    n_frames = int(sum(len(d["y"]) for d in data.values()))
    n_pos = int(sum(d["y"].sum() for d in data.values()))
    print(f"[data] {len(clips)} clips, {n_frames} frames, {n_pos} positive "
          f"({n_pos / n_frames:.1%}) at label tolerance {args.label_tol}s")

    # --- choose the blend weight by leave-one-clip-out -----------------------
    names = list(data)
    print(f"[cv]   selecting blend weight for '{args.model}' by leave-one-clip-out ...")
    oof = {}
    for test in names:
        train = [n for n in names if n != test]
        Xtr = np.vstack([data[n]["X"] for n in train])
        ytr = np.concatenate([data[n]["y"] for n in train])
        model, scaler = fit(args.model, Xtr, ytr)
        oof[test] = model.predict_proba(scaler.transform(data[test]["X"]))[:, 1]

    best_w, best_f1, table = 1.0, -1.0, {}
    for w in BLEND_WEIGHTS:
        best_for_w = -1.0
        for thr in THRESHOLDS:
            for md in MIN_DURS:
                vals = []
                for n in names:
                    d = data[n]
                    s = blend(d["rule"], oof[n], w)
                    pred = peaks_from_score(s, d["fidx"], d["ts"], d["fps"],
                                            threshold=thr, min_dur=md)
                    vals.append(f1(pred, d["gt"], 1.0))
                best_for_w = max(best_for_w, float(np.mean(vals)))
        table[w] = round(best_for_w, 3)
        if best_for_w > best_f1:
            best_f1, best_w = best_for_w, w
        print(f"       w_rule={w:.1f}  mean F1@1.0s = {best_for_w:.3f}")

    print(f"[cv]   best blend weight w_rule={best_w:.1f} (mean F1 {best_f1:.3f})")
    print("[note] These figures are OPTIMISTIC and are for weight selection only:")
    print("       model probabilities are out-of-fold, but the peak threshold is")
    print("       chosen across all clips. The honest held-out comparison uses")
    print("       fully nested selection -- run: python learned_baseline.py --src .")

    # --- retrain on everything and save -------------------------------------
    X = np.vstack([data[n]["X"] for n in names])
    y = np.concatenate([data[n]["y"] for n in names])
    model, scaler = fit(args.model, X, y)

    import sklearn

    bundle = {
        "model": model, "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS, "offsets": list(OFFSETS),
        "meta": {
            "model_type": args.model, "n_clips": len(clips), "n_frames": n_frames,
            "n_positive": n_pos, "label_tol": args.label_tol,
            "clips": list(clips),
            # Recorded so a version mismatch produces a precise diagnostic
            # rather than an obscure AttributeError from inside the estimator.
            "sklearn_version": sklearn.__version__,
        },
    }
    out = os.path.join(args.src, args.out)
    joblib.dump(bundle, out)

    sidecar = os.path.splitext(out)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump({
            "model_type": args.model,
            "blend_weight_rule": best_w,
            "loo_blend_sweep_mean_f1_1s": table,
            "label_tolerance_s": args.label_tol,
            "n_clips": len(clips), "n_frames": n_frames, "n_positive": n_pos,
            "note": ("Blend weight selected by leave-one-clip-out. The final model is "
                     "retrained on all clips, so it must not be evaluated on them; "
                     "learned_baseline.py provides the held-out comparison."),
        }, fh, indent=2)

    print(f"[save] {out}")
    print(f"[save] {sidecar}")
    print(f"\nUse it:  python main.py --video clip.mp4 --scorer hybrid ...")


if __name__ == "__main__":
    main()
