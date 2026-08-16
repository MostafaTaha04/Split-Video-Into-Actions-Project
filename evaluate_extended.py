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

# Clips come from clips.json via clip_registry, so adding data needs no code
# change here. CLIPS/CLEAN/HELDOUT keep their original meaning:
#
#   CLIPS   development clips — the global configuration and the
#           leave-one-clip-out study use ONLY these.
#   CLEAN   continuously-recorded clips, the footage the method targets.
#   HELDOUT clips that took no part in tuning at any stage, reported
#           separately. These are the only fully untainted estimates.
from clip_registry import clean_names, dev_clips, heldout_clips  # noqa: E402

CLIPS = dev_clips(".")
CLEAN = clean_names(".", split="dev")
HELDOUT = heldout_clips(".")
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
    from clip_registry import resolve_gt
    with open(resolve_gt(gtf, src)) as fh:
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


def channels(F, legacy=False):
    """Named fusion channels, mirroring TemporalSegmenter._boundary_score_channels.

    ``legacy=True`` additionally returns the four channels that were removed
    after the fusion ablation, so the ablation can still quantify them.
    """
    def g(k):
        return np.array([r[k] for r in F], dtype=float)

    act = g("activity_level")
    comp = {
        "transition": g("transition_score"),
        "activity_change": 0.80 * _normalize(np.abs(np.diff(act, prepend=act[0]))),
    }
    if legacy:
        fl = _normalize(g("flow_magnitude"))
        h = g("hands_present")
        it = g("num_interactions")
        tc = g("num_tools")
        comp["flow_change"] = 0.55 * _normalize(np.abs(np.diff(fl, prepend=fl[0])))
        comp["hand_change"] = 0.55 * np.minimum(np.abs(np.diff(h, prepend=h[0])), 1.0)
        comp["interaction_change"] = 0.55 * np.minimum(np.abs(np.diff(it, prepend=it[0])), 1.0)
        comp["tool_count_change"] = 0.45 * np.minimum(np.abs(np.diff(tc, prepend=tc[0])) / 3.0, 1.0)
    return comp


def boundary_score(F, fps, drop=None, legacy=False):
    comp = channels(F, legacy=legacy)
    use = [v for k, v in comp.items() if k != drop]
    score = np.maximum.reduce(use)
    warm = min(len(score), max(3, int(0.5 * fps)))
    score[:warm] = 0.0
    return np.clip(score, 0.0, 1.0)


def peaks_from_score(score, frame_idx, timestamps, fps, threshold=0.70,
                     min_dur=2.0, sigma=2.0):
    """Turn any per-frame boundary score into boundary timestamps.

    This is the post-processing half of the method — Gaussian smoothing, peak
    finding, minimum-separation filtering, close-peak merging, and edge
    removal — factored out so that alternative scores can be compared against
    the rule-based fusion under *identical* conditions. ``learned_baseline.py``
    feeds a model's predicted probability through this same function, so any
    difference in the resulting F1 is attributable to the score itself and not
    to differences in post-processing.
    """
    score = np.asarray(score, dtype=float)
    if len(score) < max(1, int(min_dur * fps)):
        return []
    mf = max(1, int(min_dur * fps))
    s = gaussian_filter1d(score, sigma)
    pk, pr = find_peaks(s, height=threshold, distance=mf, prominence=0.08)
    B = [(int(frame_idx[i]), float(timestamps[i]), float(min(hh, 1.0)))
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
    vs, ve = timestamps[0], timestamps[-1]
    return [b[1] for b in B if (b[1] - vs >= min_dur) and (ve - b[1] >= min_dur)]


def segment(F, fps, threshold=0.70, min_dur=2.0, sigma=2.0, drop=None, legacy=False):
    return peaks_from_score(
        boundary_score(F, fps, drop=drop, legacy=legacy),
        [r["frame_idx"] for r in F],
        [r["timestamp"] for r in F],
        fps, threshold=threshold, min_dur=min_dur, sigma=sigma,
    )


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
    for name, (rd, gtf, fps) in {**CLIPS, **HELDOUT}.items():
        F = load_features(os.path.join(src, rd, "features.csv"))
        data[name] = (F, gt_boundaries(src, gtf), fps, F[-1]["timestamp"])

    res = {}

    # 1) sanity: reproduce saved cooling-fan boundaries at its run config (0.55/2.5)
    F = data["Cooling fan"][0]
    from clip_registry import resolve_results
    _sanity_dir = resolve_results("results_coolingfan_v2run", src)
    with open(os.path.join(_sanity_dir, "segmentation_results.json")) as fh:
        _saved_run = json.load(fh)
    saved = [round(float(b["timestamp"]), 3) for b in _saved_run["boundaries"]]
    _cfg = _saved_run.get("run_config") or {}
    if _cfg.get("boundary_threshold"):
        # Replay at the configuration the run actually used, read from the run
        # itself. Hardcoding it made this check fire falsely whenever a run was
        # regenerated with different settings.
        repro = [round(b, 3) for b in segment(
            F, _cfg.get("effective_fps", 10.0), _cfg["boundary_threshold"],
            _cfg["min_segment_duration"], _cfg.get("smoothing_sigma", 2.0))]
        assert repro == saved, f"re-segmentation drift!\n repro={repro}\n saved={saved}"
        print("[sanity] re-segmentation reproduces saved boundaries: OK")
    else:
        print("[sanity] skipped — saved run predates run_config "
              "(re-run main.py to record it)")

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

    # 3b) TRUE HELD-OUT clip: scored with the frozen global config. This clip
    # took no part in choosing gthr/gmd, so it is the only fully untainted
    # generalisation number in the project.
    held = {}
    for name in HELDOUT:
        F, gt, fps, dur = data[name]
        pred = segment(F, fps, gthr, gmd)
        held[name] = {
            "f1_1s": round(f1(pred, gt, 1.0), 3),
            "f1_2s": round(f1(pred, gt, 2.0), 3),
            "f1_3s": round(f1(pred, gt, 3.0), 3),
            "n_pred_boundaries": len(pred),
            "n_gt_boundaries": len(gt),
            "duration_s": round(dur, 2),
        }
        # Diagnosis. Two candidate explanations were tested:
        #
        #   (a) "the opening unboxing phase causes it" — REJECTED. Re-scoring
        #       with the first 10 s (or 16.5 s) removed makes F1 *worse*
        #       (0.286 -> 0.182), so the error is not concentrated there.
        #   (b) over-segmentation plus annotation error — SUPPORTED. The
        #       system proposes 9 boundaries against 5 annotated ones, and of
        #       the 3 unmatched references, 2 sit 1.8 s from a prediction,
        #       which is inside this project's own annotation-uncertainty
        #       band. This clip's ground truth is one of the unverified
        #       round-number files (every boundary on a whole or half second)
        #       and fails `annotate.py validate`.
        #
        # Both numbers are recorded so the claim is checkable, not asserted.
        held[name]["over_segmentation_ratio"] = round(len(pred) / max(len(gt), 1), 3)
        for cut in (10.0, 16.5):
            p2 = [t for t in pred if t > cut]
            g2 = [t for t in gt if t > cut]
            held[name][f"f1_1s_after_{cut}s"] = round(f1(p2, g2, 1.0), 3)
        held[name]["nearest_pred_offset_per_gt_s"] = [
            round(min(abs(p - g) for p in pred), 2) for g in gt
        ] if pred else []
        held[name]["ground_truth_validated"] = False
        held[name]["ground_truth_note"] = (
            "unverified template (all boundaries on whole/half seconds); "
            "fails annotate.py validate — re-annotate before quoting this number"
        )
    res["heldout"] = held

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

    # 7a) fusion ablation over the RETAINED channels (clean clips)
    abl = {}
    for drop in ["none", "transition", "activity_change"]:
        vals = []
        for n in CLEAN:
            F, gt, fps, _ = data[n]
            pred = segment(F, fps, gthr, gmd, drop=(None if drop == "none" else drop))
            vals.append(f1(pred, gt, 1.0))
        abl[drop] = round(float(np.mean(vals)), 3)
    res["fusion_ablation_clean_f1_1s"] = abl

    # 7b) justification for REMOVING the four legacy channels: sweep the whole
    # grid on every clip and compare the 2-channel fusion against the old
    # 6-channel one. Reports how often they differ and who wins.
    from itertools import product as _product
    n_diff = n_same = 0
    win_new = win_old = tie = 0
    f_new, f_old = [], []
    for n in CLIPS:
        F, gt, fps, _ = data[n]
        for t, m in _product(THRESHOLDS, MIN_DURS):
            a = [round(b, 4) for b in segment(F, fps, t, m, legacy=True)]
            b_ = [round(b, 4) for b in segment(F, fps, t, m, legacy=False)]
            if a == b_:
                n_same += 1
                continue
            n_diff += 1
            fa, fb = f1(a, gt, 1.0), f1(b_, gt, 1.0)
            f_old.append(fa)
            f_new.append(fb)
            if fb > fa:
                win_new += 1
            elif fa > fb:
                win_old += 1
            else:
                tie += 1
    res["legacy_channel_removal"] = {
        "configs_compared": n_same + n_diff,
        "configs_identical": n_same,
        "configs_differing": n_diff,
        "differing_new_better": win_new,
        "differing_old_better": win_old,
        "differing_tied": tie,
        "mean_f1_over_differing_old": round(float(np.mean(f_old)), 4) if f_old else None,
        "mean_f1_over_differing_new": round(float(np.mean(f_new)), 4) if f_new else None,
    }

    # 8) activity-label quality, collected across every saved run.
    # Reported as a first-class number because it is a NEGATIVE result: the
    # rule-based labeller in activity_recognizer.py scores essentially zero.
    # Surfacing it here means the failure is documented and reproducible rather
    # than buried in a per-run text file. Labelling is out of scope for the
    # brief ("we don't need to detect or recognize these actions"), so this
    # does not affect the segmentation results.
    labels = {}
    import glob as _glob
    for rpt in sorted(_glob.glob(os.path.join(src, "results_*", "evaluation_report.txt"))
                      + _glob.glob(os.path.join(src, "results", "*", "evaluation_report.txt"))):
        run = os.path.basename(os.path.dirname(rpt))
        for line in open(rpt, encoding="utf-8"):
            if line.startswith("Rough activity label score"):
                labels[run] = float(line.split(":")[1])
    if labels:
        res["activity_label_score"] = {
            "per_run": labels,
            "mean": round(float(np.mean(list(labels.values()))), 4),
            "max": round(float(max(labels.values())), 4),
            "note": ("keyword-overlap score of predicted vs annotated step labels. "
                     "Near-zero across every run: the rule-based labeller emits generic "
                     "phrases ('Unspecified hand activity') because zero-shot detection "
                     "cannot identify the components. Documented negative result; see "
                     "report section 7.5. Segmentation metrics are unaffected."),
        }

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

    make_alignment_figure(src, res, data, fig_dir)
    print(f"[figures] written to {fig_dir}/")


def make_alignment_figure(src, res, data, fig_dir):
    """Predicted vs ground-truth boundaries on a shared time axis.

    One row per clip: the smoothed boundary score, the detection threshold,
    ground-truth boundaries (solid) and predicted boundaries (dashed), with
    matched pairs shaded. This makes the central finding of the project — the
    method localises boundaries precisely on clean footage and over-segments
    edited footage — visible at a glance.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    gthr, gmd = res["global"]["threshold"], res["global"]["min_dur"]
    order = (["Cooling fan", "CPU"]                     # clean
             + ["RAM", "Cable"]                          # edited tutorials
             + list(HELDOUT))                            # held out
    kind = {"Cooling fan": "clean", "CPU": "clean",
            "RAM": "edited tutorial", "Cable": "edited tutorial"}
    for n in HELDOUT:
        kind[n] = "held out"

    fig, axes = plt.subplots(len(order), 1, figsize=(11, 2.05 * len(order)), sharex=False)
    if len(order) == 1:
        axes = [axes]

    for ax, name in zip(axes, order):
        F, gt, fps, dur = data[name]
        ts = np.array([r["timestamp"] for r in F])
        score = gaussian_filter1d(boundary_score(F, fps), 2.0)
        pred = segment(F, fps, gthr, gmd)

        ax.plot(ts, score, lw=0.9, color="#3b6ea5", label="Boundary score (smoothed)")
        ax.axhline(gthr, color="gray", ls=":", lw=1, label=f"Threshold {gthr}")

        for i, g in enumerate(gt):
            ax.axvline(g, color="#2ca02c", lw=1.8,
                       label="Ground truth" if i == 0 else None)
        for i, p in enumerate(pred):
            ax.axvline(p, color="#d62728", lw=1.4, ls="--",
                       label="Predicted" if i == 0 else None)

        # Shade matched pairs (within 1.0 s) to show localisation quality.
        used = set()
        for g in gt:
            cands = [(abs(p - g), p) for p in pred if p not in used and abs(p - g) <= 1.0]
            if cands:
                off, p = min(cands)
                used.add(p)
                ax.axvspan(min(g, p), max(g, p), color="#2ca02c", alpha=0.25, lw=0)

        f1v = f1(pred, gt, 1.0)
        ax.set_ylabel("score", fontsize=8)
        ax.set_title(
            f"{name}  ({kind[name]})   F1@1.0s = {f1v:.3f}   "
            f"{len(pred)} predicted vs {len(gt)} annotated boundaries",
            fontsize=9, loc="left")
        ax.set_xlim(0, dur)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("time (s)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(
        "Predicted vs ground-truth step boundaries (single global configuration)\n"
        "green shading = matched pair within 1.0 s",
        fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    out = os.path.join(fig_dir, "boundary_alignment.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
