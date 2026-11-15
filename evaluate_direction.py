"""
evaluate_direction.py
=====================
How well does action_direction.py tell assembly steps from disassembly steps?

Labels come from the ground-truth step names, read by a fixed keyword rule on
the leading verb ("Insert / Place / Screw / Connect ..." = assembly; "Remove /
Unscrew / Take ... out / Lift ... off ..." = disassembly; anything else, e.g.
"Open the box" or "Inspect", is left out). Taking items out of packaging counts
as removing (the motion is the same) and is reported separately.

Three questions, all leave-one-source-out (the model never sees the series a
test clip comes from: YT build A / B / C, the continuous recordings, and each
of the five original clips on its own):

1. Per annotated step (ground-truth boundaries), is the direction right?
2. Per *predicted* segment (appearance scorer, frozen global configuration),
   matched to the annotated step it overlaps most, is it right?
3. Per video, in auto mode, is the verdict ("mostly assembly" / "mostly
   disassembly" / "mixed or unclear") right? And with --task set to the true
   video type, how many steps end up with the right direction?

    python evaluate_direction.py            # evaluate, write direction_results.json
    python evaluate_direction.py --fit      # also refit direction_model.json on all steps

Needs the videos the first time (one cheap 5 fps flow pass per clip, cached as
direction_flow.npz in each run folder); after that the caches are enough.
numpy and OpenCV only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

import action_direction as ad

# ---------------------------------------------------------------- labels
REMOVE_VERBS = {"remove", "unscrew", "detach", "unplug", "unclip", "loosen", "disconnect", "uninstall"}
INSTALL_VERBS = {"insert", "install", "screw", "tighten", "fasten", "attach", "connect", "mount", "seat",
                 "snap", "fit", "refit", "lock", "plug", "join", "route", "secure", "drive", "reinstall"}
PARTICLE_VERBS = {"take", "lift", "slide", "pull", "place", "set", "lay", "put", "lower", "press",
                  "push", "slip"}
OUT_WORDS = {"out", "off", "loose", "away", "free", "apart"}
IN_WORDS = {"in", "into", "onto", "on", "down", "back"}
PACKAGING = re.compile(r"\b(box|bag|bags|paperwork|foam|clamshell|tray|sleeve|insert|pouch|accessory|"
                       r"accessories|contents|manual|bubble|package|packaging|lid|parts case|cloth)\b")
# Clips whose whole video is a disassembly (everything else is an assembly video).
DISASSEMBLY_VIDEOS = {"input22 aug", "input23 aug"}


def _clause_direction(clause: str):
    words = re.findall(r"[a-z']+", clause.lower())
    if not words:
        return None
    verb, rest = words[0], set(words[1:])
    if verb in REMOVE_VERBS:
        return ad.DISASSEMBLY
    if verb in INSTALL_VERBS:
        return ad.ASSEMBLY
    if verb == "place" and "aside" not in rest:
        return ad.ASSEMBLY
    if verb in PARTICLE_VERBS and "aside" not in rest:
        if rest & OUT_WORDS:
            return ad.DISASSEMBLY
        if rest & IN_WORDS:
            return ad.ASSEMBLY
    return None


def direction_from_label(label: str):
    """'assembly', 'disassembly' or None (not a put-in / take-out step)."""
    found = {_clause_direction(c) for c in re.split(r"\band\b|,", label)} - {None}
    return found.pop() if len(found) == 1 else None


def is_packaging(label: str) -> bool:
    return bool(PACKAGING.search(label.lower()))


# ---------------------------------------------------------------- tiny logistic regression
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> dict:
    """L2-regularised logistic regression by Newton's method on standardised
    features (intercept not penalised). Same objective as scikit-learn's
    default LogisticRegression(C=1); no dependency needed."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    R = np.eye(Z.shape[1]) * l2
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return {"features": list(ad.FEATURES), "mean": mu.round(6).tolist(), "std": sd.round(6).tolist(),
            "coef": w[1:].round(6).tolist(), "intercept": round(float(w[0]), 6)}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties averaged)."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):          # average tied ranks
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(y, p):
    """AUC, accuracy at 0.5, and coverage / accuracy of the confident band."""
    y, p = np.asarray(y, bool), np.asarray(p, float)
    conf = (p >= ad.HIGH) | (p <= ad.LOW)
    return {
        "n": int(len(y)), "n_disassembly": int(y.sum()),
        "auc": round(auc(y, p), 3),
        "accuracy_all": round(float(((p >= 0.5) == y).mean()), 3) if len(y) else None,
        "confident_coverage": round(float(conf.mean()), 3) if len(y) else None,
        "confident_accuracy": (round(float(((p[conf] >= 0.5) == y[conf]).mean()), 3)
                               if conf.any() else None),
    }


# ---------------------------------------------------------------- data
def _series(name: str) -> str:
    if name.startswith("YT build"):
        return name.rstrip("0123456789")
    if name.startswith("input"):
        return "continuous recordings"
    return name


def load_clips(src: str):
    from clip_registry import load_registry, resolve_gt, resolve_results
    out = []
    for c in load_registry(src):
        video = os.path.join(src, "split-video-data", c["video"])
        run = resolve_results(c["results_dir"], src)
        if not (os.path.exists(video) or os.path.exists(os.path.join(run, "direction_flow.npz"))):
            print(f"  skip {c['name']}: neither the video nor direction_flow.npz found")
            continue
        with open(resolve_gt(c["ground_truth"], src), encoding="utf-8") as fh:
            steps = json.load(fh)["steps"]
        out.append({"clip": c, "video": video, "run": run,
                    "steps": steps, "series": _series(c["name"])})
    return out


def predicted_segments(src: str, item: dict):
    """Segments from the appearance scorer at its frozen global configuration."""
    import evaluate_extended as ee
    F = ee.load_features(os.path.join(item["run"], "features.csv"))
    desc = ee.load_appearance(src, item["clip"]["results_dir"], F)
    if desc is None:
        return None
    fps = float(item["clip"].get("fps", 10.0))
    bounds = ee.segment_appearance(F, desc, fps, 0.75, 1.5)
    t0, t1 = F[0]["timestamp"], F[-1]["timestamp"]
    edges = [t0] + sorted(bounds) + [t1]
    return list(zip(edges[:-1], edges[1:]))


def best_step(a: float, b: float, steps):
    best, ov = None, 0.0
    for s in steps:
        o = min(b, s["end"]) - max(a, s["start"])
        if o > ov:
            best, ov = s, o
    return best, ov / max(b - a, 1e-6)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=".")
    ap.add_argument("--fit", action="store_true", help="refit direction_model.json on all steps")
    ap.add_argument("--out", default="direction_results.json")
    args = ap.parse_args()

    items = load_clips(args.src)
    rows = []                      # (item index, step, features)
    for k, it in enumerate(items):
        it["flow"] = ad.flow_for_run(it["video"], it["run"])
        for s in it["steps"]:
            d = direction_from_label(s["label"])
            if d is None:
                continue
            x = ad.window_features(it["flow"], float(s["start"]), float(s["end"]))
            if x is not None:
                rows.append((k, s, d, x))
        print(f"  {it['clip']['name']}: flow ready")

    X = np.array([r[3] for r in rows])
    y = np.array([r[2] == ad.DISASSEMBLY for r in rows], dtype=float)
    series = np.array([items[r[0]]["series"] for r in rows])
    pack = np.array([is_packaging(r[1]["label"]) for r in rows])

    # 1) per annotated step, leave-one-source-out
    models, p = {}, np.zeros(len(y))
    for g in sorted(set(series)):
        tr = series != g
        models[g] = fit_logistic(X[tr], y[tr])
        p[~tr] = [ad.p_remove(x, models[g]) for x in X[~tr]]
    single = {f: round(auc(y.astype(bool), X[:, i]), 3) for i, f in enumerate(ad.FEATURES)}

    res = {
        "protocol": ("Labels from ground-truth step names (keyword rule on the leading verb); "
                     "leave-one-source-out: the model for a clip is fitted without its series. "
                     f"Confident band: p_remove <= {ad.LOW} or >= {ad.HIGH}."),
        "label_counts": dict(Counter(r[2] for r in rows)),
        "single_cue_auc": single,
        "annotated_steps": {"all": scores(y, p), "hardware_only": scores(y[~pack], p[~pack]),
                            "packaging_only": scores(y[pack], p[pack])},
    }
    by_series = {g: scores(y[series == g], p[series == g]) for g in sorted(set(series))}
    res["annotated_steps"]["by_series"] = by_series

    # 2) predicted segments + 3) video verdicts and --task
    seg_y, seg_p, verdicts, task_rows = [], [], {}, []
    for it in items:
        segs = predicted_segments(args.src, it)
        if segs is None:
            continue
        m = models[it["series"]]
        dec = ad.analyse(it["flow"], segs, "auto", m)
        true_type = ad.DISASSEMBLY if it["clip"]["name"] in DISASSEMBLY_VIDEOS else ad.ASSEMBLY
        dec_task = ad.analyse(it["flow"], segs, true_type, m)
        summ = ad.summarise([d["direction"] for d in dec], "auto")
        verdicts[it["clip"]["name"]] = {"video_type": true_type, "verdict": summ["verdict"],
                                        "n_assembly": summ["n_assembly"],
                                        "n_disassembly": summ["n_disassembly"],
                                        "n_unclear": summ["n_unclear"]}
        for (a, b), d, dt in zip(segs, dec, dec_task):
            s, frac = best_step(a, b, it["steps"])
            truth = direction_from_label(s["label"]) if (s and frac >= 0.5) else None
            if truth is None or d["direction_p_remove"] is None:
                continue
            seg_y.append(truth == ad.DISASSEMBLY)
            seg_p.append(d["direction_p_remove"])
            p_rm = d["direction_p_remove"]
            override = (ad.DISASSEMBLY if p_rm >= 0.8 else ad.ASSEMBLY if p_rm <= 0.2 else true_type)
            task_rows.append((truth, dt["direction"], true_type, override))
    res["predicted_segments"] = scores(np.array(seg_y), np.array(seg_p))
    res["predicted_segments"]["note"] = ("appearance scorer at 0.75 / 1.5; each segment matched to "
                                         "the annotated step covering at least half of it")
    t = np.array(task_rows, dtype=object)
    if len(t):
        res["with_task_set_to_true_video_type"] = {
            "n": int(len(t)),
            "accuracy_task": round(float((t[:, 0] == t[:, 1]).mean()), 3),
            "accuracy_task_plus_strong_evidence": round(float((t[:, 0] == t[:, 3]).mean()), 3),
            "note": ("Every step named for the video type. Assembly videos also contain "
                     "take-out steps (side panels, packaging), so this is the share of matched "
                     "segments whose direction equals the video type. "
                     "'task_plus_strong_evidence' is the rejected variant in which evidence "
                     "beyond 0.2 / 0.8 overrides --task."),
        }
    res["video_verdicts_auto"] = verdicts
    right = sum((v["video_type"] == ad.ASSEMBLY and v["verdict"] == "mostly assembly")
                or (v["video_type"] == ad.DISASSEMBLY and v["verdict"] == "mostly disassembly")
                for v in verdicts.values())
    wrong = sum((v["video_type"] == ad.ASSEMBLY and v["verdict"] == "mostly disassembly")
                or (v["video_type"] == ad.DISASSEMBLY and v["verdict"] == "mostly assembly")
                for v in verdicts.values())
    res["video_verdict_counts"] = {"right": right, "wrong": wrong,
                                   "mixed_or_unclear": len(verdicts) - right - wrong}

    if args.fit:
        model = fit_logistic(X, y)
        model["fitted_on"] = (f"{len(y)} annotated put-in / take-out steps from "
                              f"{len(items)} clips (evaluate_direction.py --fit)")
        with open(os.path.join(args.src, ad.MODEL_FILE), "w", encoding="utf-8") as fh:
            json.dump(model, fh, indent=2)
        res["fitted_model"] = model
        print("DEFAULT_MODEL =", json.dumps({k: model[k] for k in
                                             ("features", "mean", "std", "coef", "intercept")}))

    with open(os.path.join(args.src, args.out), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    a = res["annotated_steps"]
    print("\nASSEMBLY vs DISASSEMBLY (leave-one-source-out)")
    print(f"  labelled steps: {res['label_counts']}")
    print(f"  single-cue AUC: {single}")
    for k in ("all", "hardware_only", "packaging_only"):
        s = a[k]
        print(f"  annotated steps, {k:15s} n={s['n']:3d}  AUC {s['auc']:.3f}  "
              f"acc {s['accuracy_all']:.3f}  confident: {s['confident_coverage']:.0%} of steps, "
              f"acc {s['confident_accuracy']}")
    s = res["predicted_segments"]
    print(f"  predicted segments          n={s['n']:3d}  AUC {s['auc']:.3f}  acc {s['accuracy_all']:.3f}  "
          f"confident: {s['confident_coverage']:.0%}, acc {s['confident_accuracy']}")
    if "with_task_set_to_true_video_type" in res:
        w = res["with_task_set_to_true_video_type"]
        print(f"  with --task = video type: acc {w['accuracy_task']:.3f} "
              f"(with strong-evidence override: {w['accuracy_task_plus_strong_evidence']:.3f})")
    print(f"  video verdicts (auto): {res['video_verdict_counts']}")
    for n, v in verdicts.items():
        print(f"     {n:18s} {v['video_type']:12s} -> {v['verdict']} "
              f"({v['n_assembly']}A/{v['n_disassembly']}D/{v['n_unclear']}?)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
