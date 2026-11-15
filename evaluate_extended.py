"""
evaluate_extended.py
---------------------
Extended, reproducible evaluation of the segmentation method that runs WITHOUT
re-executing the heavy vision pipeline. Every run already saves a per-frame
``features.csv``; this script replays only the (fast) boundary-detection stage
on those saved features, so it can sweep parameters and run cross-validation in
seconds.

It reproduces, for the registered clips (clips.json):

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
  8. The APPEARANCE scorer (appearance.py) under the identical protocol, if
     appearance.npz has been computed for every clip (python appearance.py).
  9. STANDARD METRICS from the literature (segmental F1@10/25/50, Hungarian
     frame accuracy, GEBD Rel.Dis F1) and pooled F1 with a bootstrap CI.

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
#   CLEAN   continuously-recorded development clips, the footage the method
#           targets. EMPTY at present: frame checking showed that the
#           cooling-fan and CPU clips contain jump cuts, so they are 'edited'.
#           Every clean-only number is then None, and analyses that used the
#           clean clips fall back to all development clips.
#   HELDOUT clips that took no part in tuning at any stage, reported
#           separately. These are the only fully untainted estimates.
#   AIDRAFT clips whose annotation was drafted by an AI and is not yet
#           human-reviewed (clips_ai_draft.json). Never used for tuning or
#           in any headline number; scored only at the frozen global
#           configuration and reported in their own 'ai_draft_heldout' section.
#   HELDOUT_HAND the held-out clips annotated by hand. HELDOUT also holds clips
#           whose AI-drafted annotation was reviewed by people and promoted
#           (clips.json "annotation_source": "ai_draft_reviewed"); results on
#           the held-out set are always also split by this provenance.
from clip_registry import (ai_draft_clips, clean_names, dev_clips,  # noqa: E402
                           hand_heldout_clips, heldout_clips)

CLIPS = dev_clips(".")
CLEAN = clean_names(".", split="dev")
HELDOUT = heldout_clips(".")
HELDOUT_HAND = hand_heldout_clips(".")
AIDRAFT = ai_draft_clips(".")


def _mean_or_none(vals):
    """Mean rounded to 3 places, or None when there is nothing to average
    (no development clip is a continuous recording)."""
    vals = list(vals)
    return round(float(np.mean(vals)), 3) if vals else None


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


# ------------------ appearance scorer (mirror) ---------------------------
def _appearance_window():
    from config import SegmenterParams
    return SegmenterParams().appearance_window_seconds


def appearance_boundary_score(desc, fps, window_s=None):
    """Mirror of TemporalSegmenter.appearance_score + warm-up zeroing."""
    from appearance import novelty
    score = _normalize(novelty(desc, fps, window_s or _appearance_window()))
    warm = min(len(score), max(3, int(0.5 * fps)))
    score[:warm] = 0.0
    return np.clip(score, 0.0, 1.0)


def segment_appearance(F, desc, fps, threshold, min_dur, sigma=2.0):
    return peaks_from_score(
        appearance_boundary_score(desc, fps),
        [r["frame_idx"] for r in F], [r["timestamp"] for r in F],
        fps, threshold=threshold, min_dur=min_dur, sigma=sigma,
    )


def load_appearance(src, rd, F):
    """appearance.npz for a run, aligned to its features (None if missing)."""
    from appearance import load_for_run
    from clip_registry import resolve_results
    return load_for_run(resolve_results(rd, src), [r["frame_idx"] for r in F])


def standard_metrics(pred, gt, duration):
    """Literature metrics (class-agnostic) for one clip; see utils.py."""
    from utils import MetricsCalculator as M
    ps = M.boundaries_to_segments(pred, duration)
    gs = M.boundaries_to_segments(gt, duration)
    g = M.gebd_f1(pred, gt, duration)
    return {
        "segF1@10": round(M.segmental_f1(ps, gs, 0.10), 3),
        "segF1@25": round(M.segmental_f1(ps, gs, 0.25), 3),
        "segF1@50": round(M.segmental_f1(ps, gs, 0.50), 3),
        "MoF_hungarian": round(M.frame_accuracy_hungarian(ps, gs), 3),
        "gebd_f1@0.05": round(g["0.05"], 3),
        "gebd_f1_avg": round(g["avg"], 3),
    }


def pooled_f1_with_ci(preds, gts, tol=1.0, reps=2000, seed=0):
    """Micro-averaged boundary F1 over clips, with a 95% bootstrap CI that
    resamples whole clips. With 4-5 clips the interval is wide -- that width is
    the honest statement of how little 5 clips can establish."""
    from utils import MetricsCalculator as M
    names = list(preds)

    def micro(sel):
        tp = sum(len(M.match_boundaries(preds[n], gts[n], tol)) for n in sel)
        npred = sum(len(preds[n]) for n in sel)
        ngt = sum(len(gts[n]) for n in sel)
        p = tp / npred if npred else 0.0
        r = tp / ngt if ngt else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    rng = np.random.default_rng(seed)
    boots = [micro([names[i] for i in rng.integers(0, len(names), len(names))])
             for _ in range(reps)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"f1": round(micro(names), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "n_clips": len(names)}


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
    ap.add_argument("--full-json", action="store_true",
                    help="also print the whole extended_results.json to the console")
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
    res["loo_mean_clean_f1_1s"] = _mean_or_none(loo[n]["f1_1s"] for n in CLEAN)
    # The clean-only mean averages just the continuous dev clips (none at
    # present, so None). The all-dev mean is the leave-one-clip-out estimate.
    res["loo_mean_all_f1_1s"] = round(np.mean([loo[n]["f1_1s"] for n in CLIPS]), 3)

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
        # Diagnosis fields. They were added when this clip scored 0.286 against
        # its original, round-number annotation (5 boundaries, every one on a
        # whole or half second). After re-annotation from frames (10
        # boundaries, validated) the clip scores 0.583; the remaining error is
        # over-segmentation (see over_segmentation_ratio), and removing the
        # opening unboxing phase does not help (f1_1s_after_*). The fields are
        # recomputed every run so the claim stays checkable, not asserted.
        held[name]["over_segmentation_ratio"] = round(len(pred) / max(len(gt), 1), 3)
        for cut in (10.0, 16.5):
            p2 = [t for t in pred if t > cut]
            g2 = [t for t in gt if t > cut]
            held[name][f"f1_1s_after_{cut}s"] = round(f1(p2, g2, 1.0), 3)
        held[name]["nearest_pred_offset_per_gt_s"] = [
            round(min(abs(p - g) for p in pred), 2) for g in gt
        ] if pred else []
        # Report annotation quality by actually measuring it, rather than
        # hardcoding a verdict that goes stale the moment a file is fixed.
        # A boundary landing exactly on a whole second is the signature of a
        # time typed from memory rather than read off frames.
        whole = sum(1 for t in gt if abs(t - round(t)) < 1e-6)
        frac = whole / len(gt) if gt else 0.0
        held[name]["gt_whole_second_fraction"] = round(frac, 3)
        held[name]["ground_truth_validated"] = frac < 0.6
        held[name]["ground_truth_note"] = (
            f"{whole}/{len(gt)} boundaries ({frac:.0%}) fall on a whole second. "
            + ("Above the 60% limit: this looks like an unverified template, and "
               "annotate.py validate rejects it — re-annotate before quoting this "
               "number." if frac >= 0.6 else
               "Below the 60% limit; consistent with times read off frames.")
        )
    res["heldout"] = held

    # 4) sensitivity (threshold @min=2.0)
    sens = {}
    for t in THRESHOLDS:
        vals = {n: cf1(n, t, 2.0) for n in CLIPS}
        sens[t] = {"all": round(np.mean(list(vals.values())), 3)}
        if CLEAN:
            sens[t]["clean"] = _mean_or_none(vals[n] for n in CLEAN)
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

    # 7a) fusion ablation over the RETAINED channels (the continuous dev clips,
    # or every dev clip when there are none)
    abl_clips = list(CLEAN) or list(CLIPS)
    abl = {"clips": abl_clips}
    for drop in ["none", "transition", "activity_change"]:
        vals = []
        for n in abl_clips:
            F, gt, fps, _ = data[n]
            pred = segment(F, fps, gthr, gmd, drop=(None if drop == "none" else drop))
            vals.append(f1(pred, gt, 1.0))
        abl[drop] = round(float(np.mean(vals)), 3)
    res["fusion_ablation_f1_1s"] = abl

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

    # 9) APPEARANCE SCORER (appearance.py): same protocol as the rule scorer --
    # one global (threshold, min_dur) chosen on the dev clips over the SAME
    # grid, leave-one-clip-out, and the held-out clip(s) at the frozen config.
    # The before/after window is fixed (SegmenterParams), not tuned per fold.
    app = {n: load_appearance(src, CLIPS.get(n, HELDOUT.get(n))[0], data[n][0])
           for n in data}
    if all(v is not None for v in app.values()):
        def af1(name, thr, md, tol=1.0):
            F, gt, fps, _ = data[name]
            return f1(segment_appearance(F, app[name], fps, thr, md), gt, tol)

        abest = max(((np.mean([af1(n, t, m) for n in CLIPS]), t, m)
                     for t, m in product(THRESHOLDS, MIN_DURS)), key=lambda x: x[0])
        amean, athr, amd = abest
        aloo = {}
        for test in CLIPS:
            train = [n for n in CLIPS if n != test]
            _, t, m = max(((np.mean([af1(n, t, m) for n in train]), t, m)
                           for t, m in product(THRESHOLDS, MIN_DURS)), key=lambda x: x[0])
            aloo[test] = {"thr": t, "min_dur": m, "f1_1s": round(af1(test, t, m), 3),
                          "f1_3s": round(af1(test, t, m, 3.0), 3)}
        aheld = {}
        for name in HELDOUT:
            F, gt, fps, dur = data[name]
            pred = segment_appearance(F, app[name], fps, athr, amd)
            aheld[name] = {"f1_1s": round(f1(pred, gt, 1.0), 3),
                           "f1_2s": round(f1(pred, gt, 2.0), 3),
                           "f1_3s": round(f1(pred, gt, 3.0), 3),
                           "n_pred_boundaries": len(pred), "n_gt_boundaries": len(gt)}
        res["appearance_scorer"] = {
            "window_s": _appearance_window(),
            "global": {"threshold": athr, "min_dur": amd, "mean_f1_1s": round(amean, 3),
                       "per_clip": {n: {"f1_1s": round(af1(n, athr, amd), 3),
                                        "f1_3s": round(af1(n, athr, amd, 3.0), 3),
                                        "n_pred": len(segment_appearance(
                                            data[n][0], app[n], data[n][2], athr, amd)),
                                        "n_gt": len(data[n][1])}
                                    for n in CLIPS}},
            "loo": aloo,
            "loo_mean_all_f1_1s": round(np.mean([aloo[n]["f1_1s"] for n in CLIPS]), 3),
            "loo_mean_clean_f1_1s": _mean_or_none(aloo[n]["f1_1s"] for n in CLEAN),
            "heldout": aheld,
            "note": ("Selected among several variants (score blends, top-K, merging, "
                     "prominence) by dev leave-one-clip-out; the held-out clip was "
                     "scored for every variant, so it is no longer fully untouched for "
                     "this comparison. See README, 'Appearance scorer'."),
        }

        # 10) standard metrics + pooled F1 with bootstrap CI, both scorers,
        # each at its own frozen global configuration.
        std, std_preds = {}, {}
        for scorer in ("rule", "appearance"):
            per, preds, gts = {}, {}, {}
            for n in list(CLIPS) + list(HELDOUT):
                F, gt, fps, dur = data[n]
                pred = (segment(F, fps, gthr, gmd) if scorer == "rule"
                        else segment_appearance(F, app[n], fps, athr, amd))
                per[n] = {"f1_1s": round(f1(pred, gt, 1.0), 3), **standard_metrics(pred, gt, dur)}
                preds[n], gts[n] = pred, gt
            keys = [k for k in per[next(iter(per))] if k != "f1_1s"] + ["f1_1s"]
            std[scorer] = {
                "per_clip": per,
                "mean_all_clips": {k: round(float(np.mean([per[n][k] for n in per])), 3)
                                   for k in keys},
                "pooled_f1_1s_all_clips": pooled_f1_with_ci(preds, gts, 1.0),
                "pooled_f1_1s_hand_annotated": pooled_f1_with_ci(
                    {n: preds[n] for n in list(CLIPS) + list(HELDOUT_HAND)},
                    {n: gts[n] for n in list(CLIPS) + list(HELDOUT_HAND)}, 1.0),
                "n_clips": len(per),
            }
            std_preds[scorer] = preds
        res["standard_metrics"] = std

        # 10b) EDIT-CUT-ONLY BASELINE. Many clips, including some registered as
        # 'clean', contain editor's cuts, and a cut often lands on a step change.
        # A detector of sudden picture change (edit_cuts.py, no learning, no
        # tuning on annotations) is therefore a baseline every scorer must beat.
        hand = {**CLIPS, **HELDOUT_HAND}
        allc = {**CLIPS, **HELDOUT}
        res["edit_cuts"] = {
            "hand_annotated": edit_cut_analysis(
                src, {n: data[n][1] for n in hand},
                {s: {n: p[n] for n in hand} for s, p in std_preds.items()}, hand),
            "all_clips": edit_cut_analysis(
                src, {n: data[n][1] for n in allc}, std_preds, allc),
        }
    else:
        missing = [n for n, v in app.items() if v is None]
        res["appearance_scorer"] = {
            "skipped": f"appearance.npz missing for {missing}; run: python appearance.py --src ."}

    # 11) AI-DRAFT HELD-OUT CLIPS (clips_ai_draft.json). Scored with each
    # scorer's frozen global configuration, chosen on the dev clips above; the
    # annotations played no part in choosing it. Kept in a separate section
    # because the annotations are AI drafts that no person has reviewed yet.
    ag = res.get("appearance_scorer", {}).get("global")
    app_cfg = (ag["threshold"], ag["min_dur"]) if ag else None
    ai = ai_draft_section(src, gthr, gmd, app_cfg)
    if ai:
        res["ai_draft_heldout"] = ai

    # 12) THE HELD-OUT SET as a whole, when it has more than one clip: both
    # scorers at their frozen configuration, with baselines, the edit-cut
    # analysis and a split by footage and by annotation provenance.
    if len(HELDOUT) > 1:
        res["heldout_set"] = clip_set_section(
            src, HELDOUT, gthr, gmd, app_cfg,
            note=("Every held-out clip, scored at each scorer's frozen global configuration "
                  "chosen on the development clips. 'by_source' separates clips annotated by "
                  "hand from clips whose annotation was drafted with an AI tool and then "
                  "checked by Mostafa Taha and Osama Najjar (clips.json annotation_source)."))

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
    if args.full_json:
        print(json.dumps(res, indent=2))
    print_summary(res)

    if not args.no_figures:
        make_figures(src, res, data)


def _series(name):
    """'YT build C10' -> 'YT build C' (the source build video); 'input22 aug'
    and its siblings -> 'continuous recordings'; any other clip keeps its name."""
    if name.startswith("YT build"):
        return name.rstrip("0123456789")
    if name.startswith("input"):
        return "continuous recordings"
    return name


def _registry_info():
    from clip_registry import load_ai_registry, load_registry
    return {c["name"]: c for c in load_registry(".") + load_ai_registry(".")}


def _footage():
    return {n: c["footage"] for n, c in _registry_info().items()}


def _source(name, info=None):
    c = (info or _registry_info()).get(name, {})
    if c.get("annotation") == "ai_draft":
        return "ai_draft_unreviewed"
    return c.get("annotation_source", "hand")


def ai_draft_section(src, gthr, gmd, app_cfg):
    """Score the not-yet-reviewed AI-annotated clips (clips_ai_draft.json)."""
    if not AIDRAFT:
        return None
    return clip_set_section(
        src, AIDRAFT, gthr, gmd, app_cfg,
        note=("Annotations DRAFTED WITH AN AI TOOL from the video frames, without "
              "consulting the system's predictions, and NOT yet reviewed by a person. "
              "Not part of tuning, leave-one-clip-out or the held-out set. Scored at each "
              "scorer's frozen global configuration."))


def clip_set_section(src, clips, gthr, gmd, app_cfg, note=""):
    """Score a set of clips at the frozen global configuration(s), with
    baselines, a split by footage and provenance, and the edit-cut analysis."""
    scorers = {"rule": (gthr, gmd)}
    if app_cfg:
        scorers["appearance"] = app_cfg
    per, gts, durs = {}, {}, {}
    info = _registry_info()
    footage = {n: c["footage"] for n, c in info.items()}
    source = {n: _source(n, info) for n in clips}
    preds = {s: {} for s in scorers}
    no_desc = []
    for name, (rd, gtf, fps) in clips.items():
        F = load_features(os.path.join(src, rd, "features.csv"))
        gt, dur = gt_boundaries(src, gtf), F[-1]["timestamp"]
        gts[name], durs[name] = gt, dur
        row = {"n_gt": len(gt), "duration_s": round(dur, 2)}
        desc = load_appearance(src, rd, F) if "appearance" in scorers else None
        if "appearance" in scorers and desc is None:
            no_desc.append(name)
        for s, (t, m) in scorers.items():
            if s == "appearance":
                if desc is None:
                    continue
                pred = segment_appearance(F, desc, fps, t, m)
            else:
                pred = segment(F, fps, t, m)
            preds[s][name] = pred
            row[s] = {"f1_1s": round(f1(pred, gt, 1.0), 3),
                      "f1_2s": round(f1(pred, gt, 2.0), 3),
                      "f1_3s": round(f1(pred, gt, 3.0), 3),
                      "n_pred": len(pred), **standard_metrics(pred, gt, dur)}
        row["uniform_oracleK_f1_1s"] = round(f1(uniform(dur, len(gt)), gt, 1.0), 3)
        row["random_f1_1s"] = round(random_avg(dur, len(preds["rule"][name]), gt, 1.0), 3)
        per[name] = row

    out = {
        "note": note,
        "n_clips": len(per),
        "n_clips_by_source": {k: sum(v == k for v in source.values())
                              for k in sorted(set(source.values()))},
        "n_gt_boundaries": sum(len(g) for g in gts.values()),
        "config": {s: {"threshold": t, "min_dur": m} for s, (t, m) in scorers.items()},
        "per_clip": per,
        "summary": {},
    }
    for s in scorers:
        names = list(preds[s])
        if s == "appearance" and no_desc:
            out["summary"][s] = {"skipped": f"appearance.npz missing for {no_desc}; "
                                            "run: python appearance.py --src ."}
            continue
        keys = [k for k in per[names[0]][s] if k != "n_pred"]
        n_pred = sum(per[n][s]["n_pred"] for n in names)
        out["summary"][s] = {
            "mean": {k: round(float(np.mean([per[n][s][k] for n in names])), 3) for k in keys},
            "pooled_f1_1s": pooled_f1_with_ci({n: preds[s][n] for n in names},
                                              {n: gts[n] for n in names}, 1.0),
            "over_segmentation_ratio": round(n_pred / max(out["n_gt_boundaries"], 1), 3),
            "mean_f1_1s_by_source_video": {
                ser: round(float(np.mean([per[n][s]["f1_1s"] for n in names
                                          if _series(n) == ser])), 3)
                for ser in sorted({_series(n) for n in names})},
            # continuous single-shot recordings (the footage the method targets)
            # vs edited tutorials, which are dominated by editor's cuts
            "by_footage": {
                fk: {"n_clips": len(sel),
                     "mean_f1_1s": round(float(np.mean([per[n][s]["f1_1s"] for n in sel])), 3),
                     "pooled_f1_1s": pooled_f1_with_ci({n: preds[s][n] for n in sel},
                                                       {n: gts[n] for n in sel}, 1.0),
                     "over_segmentation_ratio": round(
                         sum(per[n][s]["n_pred"] for n in sel)
                         / max(sum(len(gts[n]) for n in sel), 1), 3)}
                for fk in ("clean", "edited")
                for sel in [[n for n in names if footage.get(n) == fk]] if sel},
            "by_source": {
                sk: {"n_clips": len(sel),
                     "mean_f1_1s": round(float(np.mean([per[n][s]["f1_1s"] for n in sel])), 3),
                     "pooled_f1_1s": pooled_f1_with_ci({n: preds[s][n] for n in sel},
                                                       {n: gts[n] for n in sel}, 1.0)}
                for sk in sorted(set(source.values()))
                for sel in [[n for n in names if source[n] == sk]] if sel},
        }
    out["summary"]["uniform_oracleK_mean_f1_1s"] = round(
        float(np.mean([r["uniform_oracleK_f1_1s"] for r in per.values()])), 3)
    out["summary"]["random_mean_f1_1s"] = round(
        float(np.mean([r["random_f1_1s"] for r in per.values()])), 3)
    out["edit_cuts"] = edit_cut_analysis(src, gts, preds, clips)
    return out


def edit_cut_analysis(src, gts, preds, clips, near=0.5, tol=1.0):
    """How much of the score on edited footage is just finding edit cuts.

    Needs <results_dir>/edit_cuts.json from edit_cuts.py (video frames only).
    A ground-truth boundary counts as 'at a cut' when a cut lies within
    ``near`` seconds of it.
    """
    from edit_cuts import load_cuts
    from utils import MetricsCalculator as M
    cuts = {n: load_cuts(os.path.join(src, clips[n][0])) for n in gts}
    missing = [n for n, c in cuts.items() if c is None]
    if missing:
        return {"skipped": f"edit_cuts.json missing for {len(missing)} clip(s); "
                           "run: python edit_cuts.py --src ."}
    at_cut = {n: [bool(cuts[n]) and min(abs(g - c) for c in cuts[n]) <= near for g in gts[n]]
              for n in gts}
    n_gt = sum(len(v) for v in at_cut.values())
    n_at = sum(sum(v) for v in at_cut.values())
    res = {
        "gt_boundaries_at_a_cut": n_at,
        "gt_boundaries_not_at_a_cut": n_gt - n_at,
        "fraction_at_a_cut": round(n_at / max(n_gt, 1), 3),
        "mean_cuts_per_clip": round(float(np.mean([len(c) for c in cuts.values()])), 1),
        "cut_only_baseline": {
            "mean_f1_1s": round(float(np.mean([f1(cuts[n], gts[n], tol) for n in gts])), 3),
            "pooled_f1_1s": pooled_f1_with_ci(cuts, gts, tol),
            "per_clip_f1_1s": {n: round(f1(cuts[n], gts[n], tol), 3) for n in gts}},
        "recall_1s": {},
        "note": ("Edited footage: an editor's cut often lands exactly on a step change, and "
                 "a detector of sudden picture change gets those boundaries for free. "
                 "recall_1s splits each scorer's recall into boundaries at a cut and "
                 "boundaries not at a cut; the latter is the part that needs the hands."),
    }
    for s, pr in preds.items():
        if set(pr) != set(gts):
            continue
        hit_at = hit_off = 0
        for n in gts:
            found = {g for _, g, _ in M.match_boundaries(pr[n], gts[n], tol)}
            for gi, is_cut in enumerate(at_cut[n]):
                if gi in found:
                    hit_at += is_cut
                    hit_off += not is_cut
        res["recall_1s"][s] = {
            "at_cut": round(hit_at / max(n_at, 1), 3),
            "not_at_cut": round(hit_off / max(n_gt - n_at, 1), 3)}
    return res


def print_summary(res):
    """Compact, paste-able summary of the headline numbers."""
    g, held = res["global"], res["heldout"]
    print("\n================ SUMMARY (F1 @1.0 s) ================")
    print(f"rule scorer        global thr={g['threshold']} min_dur={g['min_dur']}")
    print("  per clip         " + "  ".join(f"{n} {v['f1_1s']:.3f}" for n, v in g["per_clip"].items()))
    print(f"  dev mean {g['mean_f1_1s']:.3f} | LOO all {res['loo_mean_all_f1_1s']:.3f} "
          + _clean_str(res['loo_mean_clean_f1_1s']) + "| held-out " + _held_str(held))
    a = res.get("appearance_scorer", {})
    if "global" in a:
        ag = a["global"]
        print(f"appearance scorer  global thr={ag['threshold']} min_dur={ag['min_dur']}")
        print("  per clip         " + "  ".join(f"{n} {v['f1_1s']:.3f}" for n, v in ag["per_clip"].items()))
        print(f"  dev mean {ag['mean_f1_1s']:.3f} | LOO all {a['loo_mean_all_f1_1s']:.3f} "
              + _clean_str(a['loo_mean_clean_f1_1s']) + "| held-out " + _held_str(a["heldout"]))
    elif a:
        print(f"appearance scorer  {a.get('skipped')}")
    std = res.get("standard_metrics")
    if std:
        print(f"standard metrics, mean over all {std['rule'].get('n_clips', '')} clips (rule -> appearance)")
        for k in ("segF1@10", "segF1@25", "segF1@50", "MoF_hungarian", "gebd_f1@0.05"):
            print(f"  {k:15s} {std['rule']['mean_all_clips'][k]:.3f} -> "
                  f"{std['appearance']['mean_all_clips'][k]:.3f}")
        r, p = std["rule"]["pooled_f1_1s_all_clips"], std["appearance"]["pooled_f1_1s_all_clips"]
        print(f"  pooled F1 [95% CI] rule {r['f1']:.3f} {r['ci95']} -> appearance {p['f1']:.3f} {p['ci95']}")
        if "pooled_f1_1s_hand_annotated" in std["rule"] and std["rule"].get("n_clips", 0) > len(CLIPS) + len(HELDOUT_HAND):
            r, p = std["rule"]["pooled_f1_1s_hand_annotated"], std["appearance"]["pooled_f1_1s_hand_annotated"]
            print(f"  hand-annotated clips only ({r['n_clips']}): rule {r['f1']:.3f} {r['ci95']} "
                  f"-> appearance {p['f1']:.3f} {p['ci95']}")
    for key, label in (("hand_annotated", "hand-annotated clips"), ("all_clips", "all clips")):
        e = res.get("edit_cuts", {}).get(key)
        if not e:
            continue
        if "skipped" in e:
            print(f"edit-cut-only baseline, {label}: {e['skipped']}")
            continue
        c = e["cut_only_baseline"]
        print(f"edit-cut-only baseline, {label} ({c['pooled_f1_1s']['n_clips']}): "
              f"{e['fraction_at_a_cut']:.0%} of boundaries on a cut | mean F1@1s "
              f"{c['mean_f1_1s']:.3f} | pooled {c['pooled_f1_1s']['f1']:.3f} {c['pooled_f1_1s']['ci95']}")
        print("  recall@1s at cut / not at cut: " + "  ".join(
            f"{s} {v['at_cut']:.2f}/{v['not_at_cut']:.2f}" for s, v in e["recall_1s"].items()))
    hs = res.get("heldout_set")
    if hs:
        src_n = ", ".join(f"{v} {_SOURCE_LABEL.get(k, k)}" for k, v in hs["n_clips_by_source"].items())
        print_set(hs, f"--- HELD-OUT SET: {hs['n_clips']} clips ({src_n}), "
                      f"{hs['n_gt_boundaries']} boundaries, frozen config ---")
    ai = res.get("ai_draft_heldout")
    if ai:
        print_set(ai, f"--- AI-DRAFT annotations, NOT human-reviewed: {ai['n_clips']} clips, "
                      f"{['n_gt_boundaries']} boundaries (reported separately) ---")
    print("=====================================================")


_SOURCE_LABEL = {"hand": "hand-annotated", "ai_draft_reviewed": "AI-drafted + human-reviewed",
                 "ai_draft_unreviewed": "AI-drafted, unreviewed"}


def _clean_str(v):
    """'LOO clean' only when there are continuous development clips."""
    return "" if v is None else f"| LOO clean {v:.3f} "


def _held_str(held):
    """Held-out scores for the summary: by name when few, else a count."""
    if len(held) <= 3:
        return ", ".join(f"{n} {v['f1_1s']:.3f}" for n, v in held.items())
    hand = [n for n in held if n in HELDOUT_HAND]
    return (", ".join(f"{n} {held[n]['f1_1s']:.3f}" for n in hand)
            + f" (+{len(held) - len(hand)} reviewed-draft clips: HELD-OUT SET below)")


def print_set(ai, header):
    sm = ai["summary"]
    print(header)
    for s in ("rule", "appearance"):
        if s not in sm:
            continue
        if "skipped" in sm[s]:
            print(f"  {s:10s} {sm[s]['skipped']}")
            continue
        m, pf = sm[s]["mean"], sm[s]["pooled_f1_1s"]
        print(f"  {s:10s} mean F1@1s {m['f1_1s']:.3f} @3s {m['f1_3s']:.3f} | "
              f"pooled {pf['f1']:.3f} {pf['ci95']} | segF1@50 {m['segF1@50']:.3f} | "
              f"pred/gt {sm[s]['over_segmentation_ratio']:.2f}")
        print("             by source video: " + "  ".join(
            f"{k} {v:.3f}" for k, v in sm[s]["mean_f1_1s_by_source_video"].items()))
        print("             by footage: " + "  |  ".join(
            f"{fk} ({v['n_clips']} clips) mean {v['mean_f1_1s']:.3f} pooled "
            f"{v['pooled_f1_1s']['f1']:.3f} {v['pooled_f1_1s']['ci95']} "
            f"pred/gt {v['over_segmentation_ratio']:.2f}"
            for fk, v in sm[s]["by_footage"].items()))
        if len(sm[s].get("by_source", {})) > 1:
            print("             by annotation: " + "  |  ".join(
                f"{_SOURCE_LABEL.get(k, k)} ({v['n_clips']}) mean {v['mean_f1_1s']:.3f} "
                f"pooled {v['pooled_f1_1s']['f1']:.3f}"
                for k, v in sm[s]["by_source"].items()))
    print(f"  baselines  uniform (oracle K) {sm['uniform_oracleK_mean_f1_1s']:.3f} | "
          f"random (rule's count) {sm['random_mean_f1_1s']:.3f}")
    ec = ai.get("edit_cuts", {})
    if "skipped" in ec:
        print(f"  edit cuts  {ec['skipped']}")
    elif ec:
        print(f"  edit cuts  {ec['fraction_at_a_cut']:.0%} of boundaries sit on an edit cut "
              f"({ec['gt_boundaries_at_a_cut']}/{ec['gt_boundaries_at_a_cut'] + ec['gt_boundaries_not_at_a_cut']}); "
              f"cut-only baseline mean F1@1s {ec['cut_only_baseline']['mean_f1_1s']:.3f}")
        print("             recall@1s at cut / not at cut: " + "  ".join(
            f"{s} {v['at_cut']:.2f}/{v['not_at_cut']:.2f}" for s, v in ec["recall_1s"].items()))


def make_figures(src, res, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(src, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Sensitivity curve
    ts = sorted(res["sensitivity_threshold"])
    plt.figure(figsize=(6, 4))
    if all("clean" in res["sensitivity_threshold"][t] for t in ts):
        plt.plot(ts, [res["sensitivity_threshold"][t]["clean"] for t in ts], "o-",
                 label="Continuous dev clips")
    plt.plot(ts, [res["sensitivity_threshold"][t]["all"] for t in ts], "s--",
             label="All development clips")
    plt.axvline(res["global"]["threshold"], color="gray", ls=":", label="Chosen threshold")
    plt.xlabel("Boundary threshold"); plt.ylabel("Mean F1 @1.0s")
    plt.title("Sensitivity to boundary threshold (min_dur=2.0s)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sensitivity_threshold.png"), dpi=150); plt.close()

    # Baseline comparison (development clips, F1@1.0)
    methods = ["method", "changepoint_oracleK", "uniform_oracleK", "random"]
    labels = ["Full method", "Change-point", "Uniform (oracle K)", "Random"]
    names = list(CLIPS)
    x = np.arange(len(names)); w = 0.2
    plt.figure(figsize=(7, 4))
    for i, m in enumerate(methods):
        vals = [res["baselines"][c][m][0] if res["baselines"][c][m] else 0 for c in names]
        plt.bar(x + (i - 1.5) * w, vals, w, label=labels[i])
    plt.xticks(x, names); plt.ylabel("F1 @1.0s"); plt.ylim(0, 1.05)
    plt.title("Rule-based method vs baselines (development clips, 1 s tolerance)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "baseline_comparison.png"), dpi=150); plt.close()

    make_alignment_figure(src, res, data, fig_dir)
    make_alignment_figure(src, res, data, fig_dir, scorer="appearance")
    make_scorer_figure(res, fig_dir)
    make_set_figure(res.get("ai_draft_heldout"), fig_dir, "ai_draft_heldout.png",
                    "clips with AI-DRAFTED annotations (not human-reviewed)")
    make_set_figure(res.get("heldout_set"), fig_dir, "heldout_set.png", "held-out clips")
    print(f"[figures] written to {fig_dir}/")


def make_set_figure(ai, fig_dir, fname, what):
    """Per-clip F1 on a set of clips, both scorers, with the uniform (oracle K)
    and edit-cut-only baselines as reference marks."""
    import matplotlib.pyplot as plt
    if not ai:
        return
    per = ai["per_clip"]
    names = list(per)
    scorers = [s for s in ("rule", "appearance") if s in per[names[0]]]
    x = np.arange(len(names)); w = 0.8 / len(scorers)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    style = {"rule": ("Rule-based fusion", "#8c8c8c"), "appearance": ("Appearance change", "#3b6ea5")}
    for i, s in enumerate(scorers):
        ax.bar(x + (i - (len(scorers) - 1) / 2) * w, [per[n][s]["f1_1s"] for n in names], w,
               label=style[s][0], color=style[s][1])
    ax.scatter(x, [per[n]["uniform_oracleK_f1_1s"] for n in names], marker="_", s=160,
               color="#222222", label="Uniform baseline (oracle K)", zorder=3)
    cut = ai.get("edit_cuts", {}).get("cut_only_baseline", {}).get("per_clip_f1_1s")
    if cut:
        ax.scatter(x, [cut[n] for n in names], marker="D", s=22, color="#d9822b",
                   label="Edit-cut-only baseline", zorder=4)
    ax.set_xticks(x, [n.replace("YT build ", "").replace("input", "in").replace(" aug", "")
                      .replace("Intel CPU install", "Intel*") for n in names], fontsize=8)
    foot = _footage()
    for i in range(1, len(names)):          # separate edited from continuous clips
        if foot.get(names[i]) != foot.get(names[i - 1]):
            ax.axvline(i - 0.5, color="#555555", ls="--", lw=0.8)
    ax.set_xlabel("Clip (A, B, C = edited YouTube build videos; in22-24 = continuous "
                  "recordings" + ("; * = annotated by hand" if any(n in HELDOUT_HAND for n in names)
                                  else "") + ")")
    ax.set_ylabel("Boundary F1 @1.0 s"); ax.set_ylim(0, 1.15)
    sm = ai["summary"]
    parts = [f"{s} {sm[s]['pooled_f1_1s']['f1']:.2f} {sm[s]['pooled_f1_1s']['ci95']}"
             for s in scorers if "pooled_f1_1s" in sm.get(s, {})]
    if cut:
        c = ai["edit_cuts"]["cut_only_baseline"]["pooled_f1_1s"]
        parts.append(f"edit-cut-only {c['f1']:.2f} {c['ci95']}")
    ax.set_title(f"{ai['n_clips']} {what}, frozen global config\n"
                 "pooled F1 [95% CI]: " + "  |  ".join(parts), fontsize=9)
    ax.legend(fontsize=8, loc="upper center", ncol=4, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, fname), dpi=150)
    plt.close(fig)


def make_scorer_figure(res, fig_dir):
    """Rule-based vs appearance scorer, per clip, each at its global config."""
    import matplotlib.pyplot as plt
    std = res.get("standard_metrics")
    if not std:
        return
    names = [n for n in std["rule"]["per_clip"] if n in CLIPS or n in HELDOUT_HAND]
    cut = res.get("edit_cuts", {}).get("hand_annotated", {}).get("cut_only_baseline", {})
    cut_f1 = cut.get("per_clip_f1_1s")
    series = [("rule", "Rule-based fusion", "#8c8c8c"),
              ("appearance", "Appearance change", "#3b6ea5")]
    if cut_f1:
        series.append(("cuts", "Edit cuts only (baseline)", "#d9822b"))
    x = np.arange(len(names)); w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (sc, lab, col) in enumerate(series):
        vals = ([cut_f1[n] for n in names] if sc == "cuts"
                else [std[sc]["per_clip"][n]["f1_1s"] for n in names])
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=7)
    ticks = [n + ("\n(held out)" if n in HELDOUT else "") for n in names]
    ax.set_xticks(x, ticks, fontsize=8)
    ax.set_ylabel("Boundary F1 @1.0 s"); ax.set_ylim(0, 1.25)
    r, a = std["rule"]["pooled_f1_1s_hand_annotated"], std["appearance"]["pooled_f1_1s_hand_annotated"]
    title = ("Hand-annotated clips: rule-based vs appearance scorer (single global config each)\n"
             f"pooled F1: rule {r['f1']:.2f} [{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}]"
             f"  |  appearance {a['f1']:.2f} [{a['ci95'][0]:.2f}-{a['ci95'][1]:.2f}]")
    if cut_f1:
        c = cut["pooled_f1_1s"]
        title += f"  |  edit cuts only {c['f1']:.2f} [{c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}]"
    ax.set_title(title + " (95% CI)", fontsize=8.5)
    ax.legend(fontsize=8, loc="upper center", ncol=len(series), frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scorer_comparison.png"), dpi=150)
    plt.close(fig)


def make_alignment_figure(src, res, data, fig_dir, scorer="rule"):
    """Predicted vs ground-truth boundaries on a shared time axis.

    One row per clip: the smoothed boundary score, the detection threshold,
    ground-truth boundaries (solid) and predicted boundaries (dashed), with
    matched pairs shaded. This makes the central finding of the project — the
    method localises boundaries precisely on the single-scene clips and
    over-segments the edited tutorials — visible at a glance.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    if scorer == "appearance":
        ag = res.get("appearance_scorer", {}).get("global")
        if not ag:
            return
        gthr, gmd = ag["threshold"], ag["min_dur"]
    else:
        gthr, gmd = res["global"]["threshold"], res["global"]["min_dur"]
    order = (["Cooling fan", "CPU"]                     # single scene, jump cuts
             + ["RAM", "Cable"]                          # edited tutorials
             + list(HELDOUT_HAND))                       # held out, by hand
    kind = {"Cooling fan": "single scene, jump cuts", "CPU": "jump cuts, product shots",
            "RAM": "edited tutorial", "Cable": "edited tutorial"}
    for n in HELDOUT_HAND:
        kind[n] = "held out, jump cuts"

    fig, axes = plt.subplots(len(order), 1, figsize=(11, 2.05 * len(order)), sharex=False)
    if len(order) == 1:
        axes = [axes]

    for ax, name in zip(axes, order):
        F, gt, fps, dur = data[name]
        ts = np.array([r["timestamp"] for r in F])
        if scorer == "appearance":
            desc = load_appearance(src, {**CLIPS, **HELDOUT}[name][0], F)
            score = gaussian_filter1d(appearance_boundary_score(desc, fps), 2.0)
            pred = segment_appearance(F, desc, fps, gthr, gmd)
        else:
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
    title = "rule-based fusion" if scorer == "rule" else "appearance-change scorer"
    fig.suptitle(
        f"Predicted vs ground-truth step boundaries -- {title} (single global configuration)\n"
        "green shading = matched pair within 1.0 s",
        fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    out = os.path.join(fig_dir, "boundary_alignment.png" if scorer == "rule"
                       else "boundary_alignment_appearance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
