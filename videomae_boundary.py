"""
videomae_boundary.py
====================
End-to-end deep baseline: fine-tune VideoMAE [10] directly on the step-boundary
detection task and evaluate it with the project's own segmentation metrics.

Why this experiment exists
--------------------------
Section 6.7 compares learned scorers against the hand-designed fusion, but those
models are shallow classifiers over *hand-crafted* per-frame features. They
therefore inherit the feature engineering they were meant to test. This script
removes that confound: a video transformer consumes raw pixels and is trained on
the boundary task itself, so the comparison is between a hand-designed pipeline
and a modern end-to-end model on equal footing.

Task formulation
----------------
A sliding window of ``--window`` seconds is labelled POSITIVE when its centre
lies within ``--label-tol`` seconds of an annotated step boundary. VideoMAE is
fine-tuned as a binary classifier over these windows; at inference its
P(boundary) is interpolated onto the same per-frame timestamp grid used by every
other scorer and pushed through ``evaluate_extended.peaks_from_score``. Post
-processing, metrics and the leave-one-clip-out split are therefore *identical*
to the rule-based, logistic-regression and hybrid scorers, so any difference is
attributable to the score itself.

Honest expectations
-------------------
The training set is roughly 90 positive windows. VideoMAE-base has ~86 M
parameters. Heavy overfitting is expected and a result below the hand-designed
fusion would not be surprising — that outcome is itself the finding, and is
reported rather than tuned away.

Protocol caveat
---------------
Full nested selection (an inner leave-one-out to choose the peak threshold)
would require 4x more fine-tuning runs, which is not affordable on a Colab GPU.
The threshold is therefore chosen on the four training clips using that fold's
own model. Those predictions are in-sample, so the threshold choice is mildly
optimistic — but the held-out clip is never involved in any decision, so the
reported per-clip F1 remains an honest generalisation estimate. This is stated
in the results JSON under "protocol".

Usage
-----
    # smoke test first — one fold, a handful of steps, verifies the whole path
    python videomae_boundary.py --smoke

    # full leave-one-clip-out run
    python videomae_boundary.py --epochs 3 --out videomae_results.json

Requires a CUDA GPU. See notebooks/colab_videomae.ipynb.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from evaluate_extended import CLIPS, HELDOUT, f1, gt_boundaries, load_features, peaks_from_score

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16
PEAK_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
MIN_DURS = [2.0, 2.5, 3.0]


# --------------------------------------------------------------- video access
class VideoFrames:
    """Random access to video frames, using decord if present, else OpenCV."""

    def __init__(self, path):
        self.path = path
        self._decord = None
        try:
            from decord import VideoReader, cpu
            self._decord = VideoReader(path, ctx=cpu(0))
            self.fps = float(self._decord.get_avg_fps())
            self.n = len(self._decord)
        except Exception:
            import cv2
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise SystemExit(f"cannot open video: {path}")
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def clip(self, t_start, t_end):
        """NUM_FRAMES RGB frames sampled uniformly across [t_start, t_end]."""
        i0 = max(0, int(round(t_start * self.fps)))
        i1 = min(self.n - 1, int(round(t_end * self.fps)))
        if i1 <= i0:
            i1 = min(self.n - 1, i0 + 1)
        idx = np.linspace(i0, i1, NUM_FRAMES).astype(int)

        if self._decord is not None:
            return list(self._decord.get_batch(idx).asnumpy())

        import cv2
        out = []
        for i in idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = self._cap.read()
            if ok:
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        while len(out) < NUM_FRAMES and out:
            out.append(out[-1])
        return out


# --------------------------------------------------------------- window build
def build_windows(duration, window, stride):
    """(start, end, centre) for every sliding window covering the clip."""
    out, t = [], 0.0
    while t + window <= duration + 1e-9:
        out.append((t, t + window, t + window / 2.0))
        t += stride
    return out


def label_windows(centres, boundaries, tol):
    y = np.zeros(len(centres), dtype=np.int64)
    for b in boundaries:
        y[np.abs(np.asarray(centres) - b) <= tol] = 1
    return y


# Per-process cache of open video readers, keyed by path. DataLoader workers are
# forked, and a decord/OpenCV handle created in the parent does NOT survive the
# fork — reusing one deadlocks the worker. Each process therefore opens its own
# reader on first use, which is why the dataset stores paths rather than
# VideoFrames objects.
_READERS: "dict[str, VideoFrames]" = {}


def reader_for(path: str) -> "VideoFrames":
    if path not in _READERS:
        _READERS[path] = VideoFrames(path)
    return _READERS[path]


class WindowDataset:
    """Torch Dataset over sliding windows. Defined lazily so the module imports
    without torch installed (the rest of the repo must not depend on it)."""

    def __new__(cls, *a, **kw):
        import torch
        from torch.utils.data import Dataset

        class _DS(Dataset):
            def __init__(self, clips, processor):
                # clips: list of (video_path, windows, labels) — paths, not
                # open readers, so the dataset is safe to fork.
                self.items = []
                self.processor = processor
                for path, wins, ys in clips:
                    for (s, e, _), y in zip(wins, ys):
                        self.items.append((path, s, e, int(y)))

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                path, s, e, y = self.items[i]
                frames = reader_for(path).clip(s, e)
                px = self.processor(frames, return_tensors="pt")["pixel_values"][0]
                return {"pixel_values": px, "labels": torch.tensor(y, dtype=torch.long)}

        return _DS(*a, **kw)


# --------------------------------------------------------------- train / infer
def make_model(pos_weight=None):
    from transformers import VideoMAEForVideoClassification

    # Pass only the label maps, not num_labels: transformers infers the head
    # size from them, and supplying both makes newer versions warn about a
    # mismatch against the checkpoint's 400 Kinetics labels.
    return VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        label2id={"no_boundary": 0, "boundary": 1},
        id2label={0: "no_boundary", 1: "boundary"},
        ignore_mismatched_sizes=True,   # replaces the 400-class Kinetics head
    )


def train_fold(train_clips, processor, args):
    """Fine-tune on the training clips; returns the model."""
    import torch
    from torch.utils.data import DataLoader

    ds = WindowDataset(train_clips, processor)
    n_pos = sum(1 for it in ds.items if it[3] == 1)
    n_neg = len(ds) - n_pos
    print(f"    train windows: {len(ds)}  positive: {n_pos} ({n_pos / max(len(ds),1):.1%})")

    model = make_model().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    # Class imbalance is severe (~10% positive); weight the loss rather than
    # oversample, so every window is still seen exactly once per epoch.
    w = torch.tensor([1.0, max(1.0, n_neg / max(n_pos, 1))], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    model.train()
    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        for batch in dl:
            px = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(pixel_values=px).logits
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % args.log_every == 0:
                print(f"      ep{ep+1} step{step} loss {loss.item():.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                print("      (max-steps reached)")
                return model
        print(f"    epoch {ep+1}/{args.epochs} done in {time.time()-t0:.0f}s", flush=True)
    return model


def predict_clip(model, processor, path, wins, batch_size, max_windows=0):
    """P(boundary) at each window centre.

    ``max_windows`` subsamples the windows (smoke mode only) so the pipeline can
    be verified in a couple of minutes; the returned centres are subsampled to
    match, and interpolation fills the gaps.
    """
    import torch

    use = list(wins)
    if max_windows and len(use) > max_windows:
        step = int(np.ceil(len(use) / max_windows))
        use = use[::step]

    vf = reader_for(path)
    model.eval()
    device = next(model.parameters()).device
    probs = []
    with torch.no_grad():
        for i in range(0, len(use), batch_size):
            chunk = use[i:i + batch_size]
            px = torch.stack([
                processor(vf.clip(s, e), return_tensors="pt")["pixel_values"][0]
                for s, e, _ in chunk
            ]).to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(pixel_values=px).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probs, dtype=float), [c for _, _, c in use]


def to_frame_grid(centres, probs, timestamps):
    """Interpolate window-centre probabilities onto the per-frame timeline.

    Puts the deep model's score on exactly the grid used by features.csv, so
    peaks_from_score behaves identically for every scorer.
    """
    if len(centres) == 0:
        return np.zeros(len(timestamps))
    return np.interp(timestamps, np.asarray(centres, dtype=float), probs)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=".")
    ap.add_argument("--out", default="videomae_results.json")
    ap.add_argument("--window", type=float, default=2.0, help="window length (s)")
    ap.add_argument("--stride", type=float, default=0.25, help="window stride (s)")
    ap.add_argument("--label-tol", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--infer-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. 0 decodes in-process, which is "
                         "safest; video readers are not fork-safe, so >0 relies "
                         "on each worker opening its own (see reader_for).")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--max-infer-windows", type=int, default=0,
                    help="subsample windows at inference (0 = all). Smoke mode "
                         "sets this so a check takes ~2 min, not ~6.")
    ap.add_argument("--folds", type=int, default=0, help="0 = all clips")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 6 steps — verifies the whole path in ~2 minutes")
    args = ap.parse_args()

    if args.smoke:
        args.folds, args.max_steps, args.epochs = 1, 6, 1
        args.max_infer_windows = 24
        print("[smoke] 1 fold, 6 training steps — checking the pipeline end to end\n")

    try:
        import torch
        from transformers import VideoMAEImageProcessor
    except ImportError as exc:
        raise SystemExit(
            "This experiment needs torch + transformers:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "    pip install transformers decord av\n"
            f"({exc})"
        )

    print(f"[env] torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("[env] WARNING: no GPU — this will take many hours. Use Colab.")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

    # ---- load every clip: video, boundaries, per-frame timeline -------------
    all_clips = {**CLIPS, **HELDOUT}
    data = {}
    for name, (rdir, gtf, fps) in all_clips.items():
        rows = load_features(os.path.join(args.src, rdir, "features.csv"))
        ts = np.array([r["timestamp"] for r in rows], dtype=float)
        fidx = np.array([r["frame_idx"] for r in rows], dtype=int)
        with open(os.path.join(args.src, gtf), encoding="utf-8") as fh:
            video_name = json.load(fh).get("video")
        vpath = os.path.join(args.src, video_name)
        if not os.path.exists(vpath):
            raise SystemExit(f"missing video: {vpath}\n"
                             "Mount your Drive folder or copy the .mp4 files next to the code.")
        # Probe once in the parent to fail fast on a bad path, then discard;
        # each process opens its own reader lazily (see reader_for).
        VideoFrames(vpath)
        duration = float(ts[-1])
        wins = build_windows(duration, args.window, args.stride)
        gt = gt_boundaries(args.src, gtf)
        y = label_windows([c for _, _, c in wins], gt, args.label_tol)
        data[name] = dict(path=vpath, wins=wins, y=y, gt=gt, ts=ts, fidx=fidx,
                          fps=fps, duration=duration)
        print(f"[data] {name:20s} {duration:5.1f}s  {len(wins):4d} windows  "
              f"{int(y.sum()):3d} positive  {len(gt)} boundaries")

    names = list(data)
    if args.folds:
        names = names[:args.folds]

    results, per_clip = {}, {}
    for test in names:
        print(f"\n=== fold: hold out {test} ===", flush=True)
        train = [n for n in data if n != test]
        model = train_fold([(data[n]["path"], data[n]["wins"], data[n]["y"]) for n in train],
                           processor, args)

        # threshold selection on TRAINING clips only
        train_scores = {}
        for n in train:
            d = data[n]
            p, centres = predict_clip(model, processor, d["path"], d["wins"],
                                      args.infer_batch_size, args.max_infer_windows)
            train_scores[n] = to_frame_grid(centres, p, d["ts"])
        best, best_f1 = (0.5, 2.0), -1.0
        for thr in PEAK_THRESHOLDS:
            for md in MIN_DURS:
                vals = []
                for n in train:
                    d = data[n]
                    pred = peaks_from_score(train_scores[n], d["fidx"], d["ts"], d["fps"],
                                            threshold=thr, min_dur=md)
                    vals.append(f1(pred, d["gt"], 1.0))
                m = float(np.mean(vals))
                if m > best_f1:
                    best_f1, best = m, (thr, md)
        thr, md = best

        d = data[test]
        p, centres = predict_clip(model, processor, d["path"], d["wins"],
                                  args.infer_batch_size, args.max_infer_windows)
        score = to_frame_grid(centres, p, d["ts"])
        pred = peaks_from_score(score, d["fidx"], d["ts"], d["fps"], threshold=thr, min_dur=md)
        entry = {
            "f1_1s": round(f1(pred, d["gt"], 1.0), 3),
            "f1_2s": round(f1(pred, d["gt"], 2.0), 3),
            "f1_3s": round(f1(pred, d["gt"], 3.0), 3),
            "n_pred": len(pred), "n_gt": len(d["gt"]),
            "selected_threshold": thr, "selected_min_dur": md,
            "train_f1": round(best_f1, 3),
            "max_prob": round(float(score.max()), 3),
            "mean_prob": round(float(score.mean()), 3),
        }
        per_clip[test] = entry
        print(f"  -> {test}: F1@1s={entry['f1_1s']:.3f}  F1@3s={entry['f1_3s']:.3f}  "
              f"({entry['n_pred']} predicted vs {entry['n_gt']} annotated)", flush=True)

        del model
        torch.cuda.empty_cache()

    results = {
        "model": MODEL_NAME,
        "task": "binary boundary detection on sliding windows",
        "window_s": args.window, "stride_s": args.stride, "label_tol_s": args.label_tol,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "protocol": ("leave-one-clip-out; peak threshold chosen on training clips "
                     "(model predictions on training clips are in-sample, so threshold "
                     "selection is mildly optimistic; the held-out clip is untouched)"),
        "per_clip": per_clip,
        "mean_f1_1s": round(float(np.mean([v["f1_1s"] for v in per_clip.values()])), 3),
        "smoke": bool(args.smoke),
    }
    with open(os.path.join(args.src, args.out), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"VideoMAE mean LOO F1@1.0s = {results['mean_f1_1s']:.3f}")
    print("compare with (from learned_results.json):")
    print("  hybrid 0.476 | logistic regression 0.435 | rule-based 0.405")
    print("=" * 60)
    print(f"wrote {args.out}")
    if args.smoke:
        print("\nSmoke run only — numbers are meaningless. Now run the full job:")
        print("  python videomae_boundary.py --epochs 3")


if __name__ == "__main__":
    main()
