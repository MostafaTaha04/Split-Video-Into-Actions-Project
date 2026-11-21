"""
evaluate.py
===========
How close the model's activities and cuts are to the annotation files.

All scores are computed on the half-second samples (features.py) of a video the
model was NOT trained on:

* frame accuracy      share of samples with the right label (BACKGROUND included)
* per-activity F1     for each label, how well its samples are found (precision
                      = of the samples the model called X, how many really are X;
                      recall = of the real X samples, how many it found; F1 = both)
* segment F1@10/25/50 the standard action-segmentation score: a predicted
                      activity segment counts as correct if it has the right label
                      and overlaps a real segment by at least 10 / 25 / 50 %
                      (intersection over union); BACKGROUND is not counted
* edit score          how similar the ORDER of activities is (100 = same list of
                      activities in the same order, regardless of exact times)
* cut F1 (+/- 2 s)    whether the model cuts where the annotation changes
                      activity: a real cut is found if the model cuts within 2 s
                      (also given for +/- 5 s, since the annotations themselves
                      are accurate to about 2 s)
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from labels import BACKGROUND, CLASSES


def _runs(labels: Sequence[str]):
    out = []
    for i, lab in enumerate(labels):
        if out and out[-1][0] == lab:
            out[-1][2] = i + 1
        else:
            out.append([lab, i, i + 1])
    return out


def _segments(labels: Sequence[str]):
    return [(lab, a, b) for lab, a, b in _runs(labels) if lab != BACKGROUND]


def segment_f1(pred: Sequence[str], gt: Sequence[str], overlap: float) -> float:
    """F1@overlap as in MS-TCN (Farha & Gall, 2019), BACKGROUND excluded."""
    P, G = _segments(pred), _segments(gt)
    if not P and not G:
        return 1.0
    used = [False] * len(G)
    tp = 0
    for lab, a, b in P:
        best, best_j = 0.0, -1
        for j, (gl, ga, gb) in enumerate(G):
            if gl != lab:
                continue
            inter = max(0, min(b, gb) - max(a, ga))
            union = max(b, gb) - min(a, ga)
            iou = inter / union if union else 0.0
            if iou > best:
                best, best_j = iou, j
        if best >= overlap and not used[best_j]:
            tp += 1
            used[best_j] = True
    fp, fn = len(P) - tp, len(G) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def edit_score(pred: Sequence[str], gt: Sequence[str]) -> float:
    """100 * (1 - normalised Levenshtein distance) between the two lists of activities."""
    p = [s[0] for s in _segments(pred)]
    g = [s[0] for s in _segments(gt)]
    m, n = len(p), len(g)
    if max(m, n) == 0:
        return 100.0
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (p[i - 1] != g[j - 1]))
    return float(100 * (1 - d[m, n] / max(m, n)))


def cut_f1(pred: Sequence[str], gt: Sequence[str], fps: float, tolerance_s: float = 2.0) -> Dict[str, float]:
    """Cuts = the times where the label changes. A real cut is found if a predicted cut is within tolerance."""
    def cuts(lab):
        return [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]
    pc, gc = cuts(pred), cuts(gt)
    tol = tolerance_s * fps
    used = set()
    tp = 0
    for g in gc:
        cand = [(abs(p - g), k) for k, p in enumerate(pc) if k not in used and abs(p - g) <= tol]
        if cand:
            used.add(min(cand)[1])
            tp += 1
    prec = tp / len(pc) if pc else (1.0 if not gc else 0.0)
    rec = tp / len(gc) if gc else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "predicted": len(pc), "real": len(gc)}


def per_class(pred: Sequence[str], gt: Sequence[str]) -> Dict[str, dict]:
    pred, gt = np.asarray(pred, dtype=object), np.asarray(gt, dtype=object)
    out = {}
    for c in CLASSES:
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "real_s": (tp + fn), "predicted_s": (tp + fp)}   # in samples; divided by fps by the caller
    return out


def score(pred: Sequence[str], gt: Sequence[str], fps: float) -> dict:
    pred, gt = list(pred), list(gt)
    acc = float(np.mean([p == g for p, g in zip(pred, gt)]))
    pc = per_class(pred, gt)
    for v in pc.values():
        v["real_s"] /= fps
        v["predicted_s"] /= fps
    real = [c for c, v in pc.items() if v["real_s"] > 0 and c != BACKGROUND]
    return {
        "frame_accuracy": 100 * acc,
        "macro_f1": 100 * float(np.mean([pc[c]["f1"] for c in real])) if real else 0.0,
        "f1@10": 100 * segment_f1(pred, gt, 0.10),
        "f1@25": 100 * segment_f1(pred, gt, 0.25),
        "f1@50": 100 * segment_f1(pred, gt, 0.50),
        "edit": edit_score(pred, gt),
        "cuts": cut_f1(pred, gt, fps, 2.0),
        "cuts_5s": cut_f1(pred, gt, fps, 5.0),
        "per_class": pc,
    }


def confusion(pairs: List[tuple]) -> Dict[str, Dict[str, int]]:
    """{real label: {predicted label: samples}} over all (pred, gt) lists given."""
    out: Dict[str, Dict[str, int]] = {}
    for pred, gt in pairs:
        for p, g in zip(pred, gt):
            out.setdefault(g, {}).setdefault(p, 0)
            out[g][p] += 1
    return out
