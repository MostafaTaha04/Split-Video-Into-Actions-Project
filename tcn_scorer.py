"""
tcn_scorer.py
=============
A small temporal convolutional network over the per-frame feature sequence —
the architecture family that supervised temporal action segmentation actually
uses (MS-TCN, reference [7]), scaled to this project's data.

Why this and not a bigger model
-------------------------------
Every learned scorer so far sees a *fixed* hand-picked window: features at
offsets {-4, -2, 0, +2, +4}. A TCN instead learns which temporal context
matters, through stacked dilated convolutions, and produces a smooth per-frame
probability directly — replacing the Gaussian filter that is currently applied
by hand.

Expectations, stated up front
-----------------------------
A window/tolerance sweep (``learned_baseline.py --sweep``) previously suggested
that *widening* the fixed context hurts. That measurement predates the
frame-loader aspect-ratio fix, when every hand-derived feature was zero on
portrait clips, so it should be re-run before being relied on. If it still
holds, a TCN may inherit the problem, since it has a wide receptive field by
construction — though unlike fixed offsets it *learns* which taps to use and can
in principle ignore unhelpful context.

At five clips this will almost certainly overfit. The reason to build it now is
that it is the right architecture once the dataset reaches a few dozen clips,
and it costs nothing to have waiting.

Design
------
* Input  : (batch, 23 features, time) — the same raw per-frame features the
           other learned scorers use, with the hand-designed composites removed.
* Body   : ``--layers`` residual blocks of dilated 1-D convolution, dilation
           doubling each layer (1, 2, 4, 8, ...), so the receptive field grows
           exponentially while the parameter count stays small.
* Head   : 1x1 convolution to a single logit per frame -> sigmoid.
* Loss   : BCE with positive-class weighting, since ~13% of frames are positive.
* Output : P(boundary) per frame, fed through ``peaks_from_score`` exactly like
           every other scorer, so the comparison stays like-for-like.

A whole clip is one training example (fully convolutional, variable length), so
"batch size" is a number of clips.

Usage
-----
    pip install torch
    python tcn_scorer.py --src .                 # leave-one-clip-out
    python tcn_scorer.py --src . --smoke         # quick path check

Writes ``tcn_results.json``. Runs on CPU in a few minutes at five clips; use a
GPU once the dataset grows.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluate_extended import CLIPS, HELDOUT, f1, gt_boundaries, load_features, peaks_from_score

PEAK_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
MIN_DURS = [2.0, 2.5, 3.0]


def build_sequences(src, clips, label_tol):
    """One (features, labels) sequence per clip, z-scored per feature."""
    from boundary_model import FEATURE_COLUMNS

    data = {}
    for name, (rdir, gtf, fps) in clips.items():
        rows = load_features(os.path.join(src, rdir, "features.csv"))
        X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=np.float32)
        ts = np.array([r["timestamp"] for r in rows], dtype=float)
        gt = gt_boundaries(src, gtf)
        y = np.zeros(len(ts), dtype=np.float32)
        for b in gt:
            y[np.abs(ts - b) <= label_tol] = 1.0
        data[name] = {
            "X": X, "y": y, "gt": gt, "ts": ts, "fps": fps,
            "fidx": np.array([r["frame_idx"] for r in rows], dtype=int),
        }
    return data


def make_net(n_features, channels, layers, dropout):
    import torch.nn as nn

    class Block(nn.Module):
        """Residual dilated conv block. Padding keeps the length unchanged so
        the output is one probability per input frame."""

        def __init__(self, ch, dilation):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation,
                                  dilation=dilation)
            self.norm = nn.BatchNorm1d(ch)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.act(self.norm(self.conv(x))))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_features, channels, kernel_size=1)
            self.blocks = nn.Sequential(*[Block(channels, 2 ** i) for i in range(layers)])
            self.out = nn.Conv1d(channels, 1, kernel_size=1)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x))).squeeze(1)   # (B, T) logits

    return TCN()


def train_tcn(train_data, args, n_features):
    """Fit on a list of clip dicts; returns the trained network + normaliser."""
    import torch

    Xs = [d["X"] for d in train_data]
    mu = np.concatenate(Xs, 0).mean(0, keepdims=True)
    sd = np.concatenate(Xs, 0).std(0, keepdims=True) + 1e-6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_net(n_features, args.channels, args.layers, args.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    pos = sum(float(d["y"].sum()) for d in train_data)
    neg = sum(float(len(d["y"]) - d["y"].sum()) for d in train_data)
    pw = torch.tensor([neg / max(pos, 1.0)], device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    tensors = [(torch.tensor(((d["X"] - mu) / sd).T[None], device=dev),
                torch.tensor(d["y"][None], device=dev)) for d in train_data]

    net.train()
    for ep in range(args.epochs):
        tot = 0.0
        for xb, yb in tensors:
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss)
        if args.verbose and (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"      epoch {ep+1}/{args.epochs}  loss {tot/len(tensors):.4f}", flush=True)
    return net, mu, sd


def predict(net, mu, sd, clip):
    import torch

    dev = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        x = torch.tensor(((clip["X"] - mu) / sd).T[None], device=dev)
        return torch.sigmoid(net(x))[0].cpu().numpy().astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".")
    ap.add_argument("--out", default="tcn_results.json")
    ap.add_argument("--label-tol", type=float, default=0.5)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4, help="dilations 1,2,4,8...")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 5 epochs")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 5
        print("[smoke] 1 fold, 5 epochs\n")

    try:
        import torch
    except ImportError:
        raise SystemExit("This scorer needs PyTorch:\n    pip install torch")

    from boundary_model import FEATURE_COLUMNS

    all_clips = {**CLIPS, **HELDOUT}
    data = build_sequences(args.src, all_clips, args.label_tol)
    n_feat = len(FEATURE_COLUMNS)

    rf = 1 + 2 * sum(2 ** i for i in range(args.layers))    # kernel 3, dilations
    print(f"[env] torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    print(f"[net] {args.layers} blocks x {args.channels} ch  ->  receptive field "
          f"~{rf} frames (~{rf/10:.1f}s at 10 fps)")
    n_par = sum(p.numel() for p in make_net(n_feat, args.channels, args.layers,
                                            args.dropout).parameters())
    print(f"[net] {n_par:,} parameters\n")

    names = list(data)
    if args.smoke:
        names = names[:1]

    per_clip = {}
    for test in names:
        train = [data[n] for n in data if n != test]
        print(f"=== fold: hold out {test} ===", flush=True)
        net, mu, sd = train_tcn(train, args, n_feat)

        # threshold chosen on training clips only
        best, best_f1 = (0.5, 2.0), -1.0
        tr_scores = {n: predict(net, mu, sd, data[n]) for n in data if n != test}
        for thr in PEAK_THRESHOLDS:
            for md in MIN_DURS:
                vals = []
                for n, s in tr_scores.items():
                    d = data[n]
                    pred = peaks_from_score(s, d["fidx"], d["ts"], d["fps"],
                                            threshold=thr, min_dur=md)
                    vals.append(f1(pred, d["gt"], 1.0))
                m = float(np.mean(vals))
                if m > best_f1:
                    best_f1, best = m, (thr, md)
        thr, md = best

        d = data[test]
        s = predict(net, mu, sd, d)
        pred = peaks_from_score(s, d["fidx"], d["ts"], d["fps"], threshold=thr, min_dur=md)
        per_clip[test] = {
            "f1_1s": round(f1(pred, d["gt"], 1.0), 3),
            "f1_3s": round(f1(pred, d["gt"], 3.0), 3),
            "n_pred": len(pred), "n_gt": len(d["gt"]),
            "selected_threshold": thr, "selected_min_dur": md,
            "train_f1": round(best_f1, 3),
            "score_std": round(float(s.std()), 4),
            "score_spread": round(float(s.max() - s.min()), 4),
        }
        print(f"  -> {test}: F1@1s={per_clip[test]['f1_1s']:.3f}  "
              f"({len(pred)} predicted vs {len(d['gt'])} annotated)\n", flush=True)

    res = {
        "model": "TCN (dilated 1-D conv over per-frame features)",
        "channels": args.channels, "layers": args.layers,
        "receptive_field_frames": rf, "parameters": n_par,
        "epochs": args.epochs, "lr": args.lr, "label_tol": args.label_tol,
        "protocol": "leave-one-clip-out; peak threshold chosen on training clips only",
        "per_clip": per_clip,
        "mean_f1_1s": round(float(np.mean([v["f1_1s"] for v in per_clip.values()])), 3),
        "smoke": bool(args.smoke),
    }
    with open(os.path.join(args.src, args.out), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    print("=" * 60)
    print(f"TCN mean LOO F1@1.0s = {res['mean_f1_1s']:.3f}")
    print("compare against learned_results.json / videomae_results.json "
          "(regenerate those before comparing — figures change with the features)")
    print("=" * 60)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
