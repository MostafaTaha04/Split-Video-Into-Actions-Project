"""Probe: does fusing the rule-based score with a learned score beat either alone?
Strict LOO: model, scaler, blend weight and peak threshold all fitted on train clips."""
import numpy as np
from itertools import product
from evaluate_extended import (CLIPS, HELDOUT, f1, gt_boundaries, load_features,
                               peaks_from_score, boundary_score)
import learned_baseline as lb

SRC = "."
clips = {**CLIPS, **HELDOUT}
data = lb.build(SRC, clips, "raw", 0.5)
for n, (rd, gtf, fps) in clips.items():
    rows = load_features(f"{SRC}/{rd}/features.csv")
    data[n]["rule"] = boundary_score(rows, fps)

names = list(clips)
THR = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
MD = [2.0, 2.5, 3.0]
WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # w=1 pure rule, w=0 pure learned

model_fn = lb.make_models()["logistic_regression"]

def score_of(d, p, w):
    # normalise learned prob to [0,1] range comparable with rule score
    return w * d["rule"] + (1 - w) * p

results = {}
for test in names:
    train = [n for n in names if n != test]
    # inner LOO to get out-of-fold probs for the training clips
    inner = {}
    for it in train:
        itr = [n for n in train if n != it]
        Xi = np.vstack([data[n]["X"] for n in itr]); yi = np.concatenate([data[n]["y"] for n in itr])
        inner[it] = lb.fit_predict(model_fn, Xi, yi, data[it]["X"])
    best, bf1 = None, -1
    for w, thr, md in product(WEIGHTS, THR, MD):
        s = np.mean([f1(peaks_from_score(score_of(data[n], inner[n], w), data[n]["fidx"],
                                         data[n]["ts"], data[n]["fps"], threshold=thr, min_dur=md),
                        data[n]["gt"], 1.0) for n in train])
        if s > bf1: bf1, best = s, (w, thr, md)
    w, thr, md = best
    Xtr = np.vstack([data[n]["X"] for n in train]); ytr = np.concatenate([data[n]["y"] for n in train])
    p = lb.fit_predict(model_fn, Xtr, ytr, data[test]["X"])
    d = data[test]
    pred = peaks_from_score(score_of(d, p, w), d["fidx"], d["ts"], d["fps"], threshold=thr, min_dur=md)
    results[test] = (round(f1(pred, d["gt"], 1.0), 3), w, thr, md)

print("%-20s %8s %6s %6s %5s" % ("clip", "hybrid", "w_rule", "thr", "md"))
for n, (v, w, thr, md) in results.items():
    print("%-20s %8.3f %6.2f %6.2f %5.1f" % (n, v, w, thr, md))
print("%-20s %8.3f" % ("MEAN", np.mean([v for v, *_ in results.values()])))
