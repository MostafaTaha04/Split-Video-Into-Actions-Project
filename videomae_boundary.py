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
DECODER = "auto"        # set once from --decoder in main()
DECODE_HEIGHT = 256     # see VideoFrames: decode small, not at source resolution


class VideoFrames:
    """Random access to video frames.

    Prefers decord (fast seeking) but falls back to OpenCV, both when the file
    cannot be opened AND — importantly — when a *read* fails partway through.
    decord's multithreaded FFmpeg path raises intermittent
    "Error sending packet" (EAGAIN) failures under heavy random seeking,
    especially when the file is on a high-latency filesystem such as a mounted
    Google Drive. It is therefore opened single-threaded, and any decode error
    permanently demotes this reader to OpenCV rather than aborting the run.
    """

    def __init__(self, path, decoder=None):
        self.path = path
        self._decord = None
        self._cap = None
        mode = decoder or DECODER

        # Source resolution, needed to pick a decode size that keeps the aspect
        # ratio. Read with OpenCV because it is cheap and always available.
        import cv2
        probe = cv2.VideoCapture(path)
        if not probe.isOpened():
            raise SystemExit(f"cannot open video: {path}")
        sw = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        sh = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        probe.release()

        # Decode at ~256px on the SHORT side rather than source resolution.
        # VideoMAE consumes 224x224, so full-resolution decoding wastes roughly
        # 20x the memory per frame: a 16-frame 1080p window is ~100 MB before
        # the processor shrinks it. Across thousands of windows that exhausts
        # Colab's RAM and kills the session.
        #
        # Scaling by the short side (not by height) matters because some of the
        # project's clips are portrait 1080x1920 phone footage. Scaling those by
        # height would give 144x256 — narrower than the 224 the model crops to,
        # so the processor would upscale and lose detail. Short-side scaling
        # keeps >=256px on both axes for any orientation.
        scale = min(1.0, DECODE_HEIGHT / max(1, min(sw, sh)))
        self.dec_w = max(2, int(round(sw * scale / 2)) * 2)
        self.dec_h = max(2, int(round(sh * scale / 2)) * 2)

        if mode in ("auto", "decord"):
            try:
                from decord import VideoReader, cpu
                # num_threads=1 avoids decord's flaky threaded decoder.
                self._decord = VideoReader(path, ctx=cpu(0), num_threads=1,
                                           width=self.dec_w, height=self.dec_h)
                self.fps = float(self._decord.get_avg_fps())
                self.n = len(self._decord)
                return
            except Exception as exc:
                if mode == "decord":
                    raise SystemExit(f"decord could not open {path}: {exc}")
                self._decord = None

        self._open_cv2()

    def _open_cv2(self):
        import cv2
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise SystemExit(f"cannot open video: {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        """Release decoder resources (called between folds to bound memory)."""
        self._decord = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def clip(self, t_start, t_end):
        """NUM_FRAMES RGB frames sampled uniformly across [t_start, t_end]."""
        i0 = max(0, int(round(t_start * self.fps)))
        i1 = min(self.n - 1, int(round(t_end * self.fps)))
        if i1 <= i0:
            i1 = min(self.n - 1, i0 + 1)
        idx = np.linspace(i0, i1, NUM_FRAMES).astype(int)

        if self._decord is not None:
            try:
                return list(self._decord.get_batch(idx).asnumpy())
            except Exception as exc:
                print(f"      [decode] decord failed on {os.path.basename(self.path)} "
                      f"({type(exc).__name__}); switching this file to OpenCV", flush=True)
                self._decord = None
                self._open_cv2()

        return self._clip_cv2(idx)

    def _clip_cv2_seq(self, idx):
        """Decode the given (sorted) indices sequentially.

        Reads forward through the file and keeps the frames it is asked for,
        instead of seeking per frame — seeking dominates otherwise.
        """
        import cv2
        want = list(idx)
        out, k, pos = [], 0, 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while k < len(want):
            ok, fr = self._cap.read()
            if not ok:
                break
            while k < len(want) and want[k] == pos:
                f = fr
                if f.shape[0] != self.dec_h or f.shape[1] != self.dec_w:
                    f = cv2.resize(f, (self.dec_w, self.dec_h),
                                   interpolation=cv2.INTER_AREA)
                out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                k += 1
            pos += 1
        if not out:
            raise SystemExit(f'could not read any frame from {self.path}')
        while len(out) < len(want):
            out.append(out[-1])
        return out

    def _clip_cv2(self, idx):
        import cv2
        out = []
        for i in idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = self._cap.read()
            if ok:
                # Downscale immediately, for the same memory reason as decord.
                if fr.shape[0] != self.dec_h or fr.shape[1] != self.dec_w:
                    fr = cv2.resize(fr, (self.dec_w, self.dec_h),
                                    interpolation=cv2.INTER_AREA)
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if not out:
            raise SystemExit(f"could not read any frame from {self.path}")
        while len(out) < NUM_FRAMES:
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


class ClipCache:
    """Decodes a clip ONCE at the rate the model needs, then serves windows as
    array slices.

    Decoding per window is the dominant cost otherwise, and almost all of it is
    wasted: with a 2 s window and 0.25 s stride, consecutive windows overlap by
    87%, so the same frames are re-decoded eight times over — each with an
    expensive random seek into H.264. Measured on a Colab T4 that was ~5.6 s per
    training step, projecting to roughly ten hours for the full run.

    Decoding sequentially at exactly ``NUM_FRAMES / window`` fps means any
    window is ``NUM_FRAMES`` consecutive cached frames, so extraction is a
    memory slice. A 60 s clip at 8 fps and 256 px short side is ~170 MB as
    uint8, which is affordable for five clips and eliminates all repeat work.
    """

    def __init__(self, path: str, window_s: float, decoder=None):
        self.path = path
        self.rate = NUM_FRAMES / float(window_s)      # fps we must sample at
        vf = VideoFrames(path, decoder=decoder)
        duration = vf.n / vf.fps if vf.fps else 0.0
        n_out = max(NUM_FRAMES, int(np.floor(duration * self.rate)) + 1)

        # Sorted, sequential indices — decoders handle these far better than
        # scattered random access.
        idx = np.clip(np.round(np.arange(n_out) / self.rate * vf.fps).astype(int),
                      0, max(0, vf.n - 1))
        self.frames = self._decode_all(vf, idx)
        self.n = len(self.frames)
        vf.close()

    @staticmethod
    def _decode_all(vf, idx):
        if vf._decord is not None:
            try:
                out = []
                for i in range(0, len(idx), 256):      # chunk to bound peak memory
                    out.append(vf._decord.get_batch(idx[i:i + 256]).asnumpy())
                return np.concatenate(out, axis=0)
            except Exception as exc:
                print(f"      [decode] decord failed on {os.path.basename(vf.path)} "
                      f"({type(exc).__name__}); using OpenCV", flush=True)
                vf._decord = None
                vf._open_cv2()
        return np.stack(vf._clip_cv2_seq(idx))

    def window(self, t_start: float):
        """NUM_FRAMES frames starting at t_start, as a list of HxWx3 arrays."""
        i0 = int(round(t_start * self.rate))
        i0 = max(0, min(i0, self.n - NUM_FRAMES))
        return list(self.frames[i0:i0 + NUM_FRAMES])

    def nbytes(self):
        return self.frames.nbytes


# Per-process cache, keyed by path. DataLoader workers are forked and a decoder
# handle from the parent does not survive the fork, so each process builds its
# own cache on first use — which is why the dataset stores paths, not objects.
_READERS: "dict[str, ClipCache]" = {}
_WINDOW_S = 2.0        # set from --window in main()


def reader_for(path: str) -> "ClipCache":
    if path not in _READERS:
        c = ClipCache(path, _WINDOW_S)
        print(f"      [cache] {os.path.basename(path)}: {c.n} frames, "
              f"{c.nbytes()/1e6:.0f} MB", flush=True)
        _READERS[path] = c
    return _READERS[path]


def close_readers():
    """Drop cached clips between folds so decoded frames do not accumulate."""
    _READERS.clear()


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
                frames = reader_for(path).window(s)
                px = self.processor(frames, return_tensors="pt")["pixel_values"][0]
                return {"pixel_values": px, "labels": torch.tensor(y, dtype=torch.long)}

        return _DS(*a, **kw)


# --------------------------------------------------------------- train / infer
def make_model(freeze_backbone=True):
    """VideoMAE with a fresh 2-class head.

    ``freeze_backbone`` trains only the classifier, using the pre-trained
    encoder as a fixed feature extractor. This is the standard choice at this
    data scale and it matters here: fine-tuning all ~86 M parameters on ~110
    positive windows collapsed the model onto the class prior — it emitted a
    near-constant 0.12-0.26 on every clip (spread as low as 0.003) and produced
    zero boundaries even in-sample. Freezing reduces the trainable parameters
    to ~1.5 K, which is a defensible ratio against the available labels.

    Pass ``--full-finetune`` to reproduce the collapsed variant for comparison;
    both outcomes are worth reporting.
    """
    from transformers import VideoMAEForVideoClassification

    # Pass only the label maps, not num_labels: transformers infers the head
    # size from them, and supplying both makes newer versions warn about a
    # mismatch against the checkpoint's 400 Kinetics labels.
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        label2id={"no_boundary": 0, "boundary": 1},
        id2label={0: "no_boundary", 1: "boundary"},
        ignore_mismatched_sizes=True,   # replaces the 400-class Kinetics head
    )
    if freeze_backbone:
        for p in model.videomae.parameters():
            p.requires_grad = False
    return model


def train_fold(train_clips, processor, args):
    """Fine-tune on the training clips; returns the model."""
    import torch
    from torch.utils.data import DataLoader

    ds = WindowDataset(train_clips, processor)
    n_pos = sum(1 for it in ds.items if it[3] == 1)
    n_neg = len(ds) - n_pos
    print(f"    train windows: {len(ds)}  positive: {n_pos} ({n_pos / max(len(ds),1):.1%})")

    model = make_model(freeze_backbone=not args.full_finetune).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    # Class imbalance is severe (~10% positive); weight the loss rather than
    # oversample, so every window is still seen exactly once per epoch.
    w = torch.tensor([1.0, max(1.0, n_neg / max(n_pos, 1))], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'    trainable params: {n_train:,} of {n_total:,} '
          f'({100*n_train/n_total:.3f}%)', flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
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
                processor(vf.window(s), return_tensors="pt")["pixel_values"][0]
                for s, e, _ in chunk
            ]).to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(pixel_values=px).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probs, dtype=float), [c for _, _, c in use]


def normalize_score(x):
    """5th/95th-percentile scaling to [0, 1] — the same transform the
    hand-designed scorer applies in TemporalSegmenter._normalize.

    Without this the comparison is unfair: the rule-based channels are
    normalised before peak detection, so their scores always span the
    threshold grid, while a raw classifier probability need not. VideoMAE's
    output sits at 0.12-0.26 with a spread as small as 0.003, which is below
    the grid floor and far below the required peak prominence — it could not
    produce a boundary regardless of what it had learned.

    Returns zeros when the input is genuinely flat, which is itself the
    diagnostic: a model that outputs a constant has learned nothing to detect.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_frame_grid(centres, probs, timestamps, normalize=True):
    """Interpolate window-centre probabilities onto the per-frame timeline.

    Puts the deep model's score on exactly the grid used by features.csv, so
    peaks_from_score behaves identically for every scorer.
    """
    if len(centres) == 0:
        return np.zeros(len(timestamps))
    grid = np.interp(timestamps, np.asarray(centres, dtype=float), probs)
    return normalize_score(grid) if normalize else grid


def save_results(args, per_clip, complete):
    """Write the results JSON. Called after every fold so a disconnect is survivable."""
    results = {
        "model": MODEL_NAME,
        "task": "binary boundary detection on sliding windows",
        "window_s": args.window, "stride_s": args.stride, "label_tol_s": args.label_tol,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "freeze_backbone": not args.full_finetune,
        "score_normalized": not args.raw_score,
        "protocol": ("leave-one-clip-out; peak threshold chosen on training clips "
                     "(model predictions on training clips are in-sample, so threshold "
                     "selection is mildly optimistic; the held-out clip is untouched)"),
        "per_clip": per_clip,
        "mean_f1_1s": (round(float(np.mean([v["f1_1s"] for v in per_clip.values()])), 3)
                       if per_clip else None),
        "folds_completed": len(per_clip),
        "complete": bool(complete),
        "smoke": bool(args.smoke),
    }
    with open(os.path.join(args.src, args.out), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


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
    ap.add_argument("--full-finetune", action="store_true",
                    help="Update the whole backbone instead of just the head. "
                         "Collapsed to the class prior on this dataset; kept so the "
                         "comparison can be reproduced.")
    ap.add_argument("--raw-score", action="store_true",
                    help="Skip percentile normalisation of the model score "
                         "(diagnostic only; makes the comparison unfair).")
    ap.add_argument("--decoder", choices=["auto", "decord", "opencv"], default="auto",
                    help="Frame decoder. 'auto' tries decord then falls back to OpenCV "
                         "per file on any decode error; 'opencv' forces the slower but "
                         "more forgiving path.")
    ap.add_argument("--max-infer-windows", type=int, default=0,
                    help="subsample windows at inference (0 = all). Smoke mode "
                         "sets this so a check takes ~2 min, not ~6.")
    ap.add_argument("--folds", type=int, default=0, help="0 = all clips")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 6 steps — verifies the whole path in ~2 minutes")
    args = ap.parse_args()

    global DECODER, _WINDOW_S
    DECODER = args.decoder
    _WINDOW_S = args.window

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
            train_scores[n] = to_frame_grid(centres, p, d["ts"],
                                            normalize=not args.raw_score)
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
        score = to_frame_grid(centres, p, d["ts"], normalize=not args.raw_score)
        pred = peaks_from_score(score, d["fidx"], d["ts"], d["fps"], threshold=thr, min_dur=md)
        entry = {
            "f1_1s": round(f1(pred, d["gt"], 1.0), 3),
            "f1_2s": round(f1(pred, d["gt"], 2.0), 3),
            "f1_3s": round(f1(pred, d["gt"], 3.0), 3),
            "n_pred": len(pred), "n_gt": len(d["gt"]),
            "selected_threshold": thr, "selected_min_dur": md,
            "train_f1": round(best_f1, 3),
            "score_max": round(float(score.max()), 3),
            "score_mean": round(float(score.mean()), 3),
            "raw_prob_max": round(float(p.max()), 4),
            "raw_prob_min": round(float(p.min()), 4),
            "raw_prob_std": round(float(p.std()), 4),
            "raw_prob_spread": round(float(p.max() - p.min()), 4),
        }
        per_clip[test] = entry
        print(f"  -> {test}: F1@1s={entry['f1_1s']:.3f}  F1@3s={entry['f1_3s']:.3f}  "
              f"({entry['n_pred']} predicted vs {entry['n_gt']} annotated)", flush=True)

        # Checkpoint after every fold. A full run is ~1 hour and free Colab can
        # disconnect at any point; without this, a drop at fold 4 of 5 would
        # discard every completed fold.
        save_results(args, per_clip, complete=False)
        print(f"     (partial results saved — {len(per_clip)} fold(s) done)", flush=True)

        del model
        torch.cuda.empty_cache()
        close_readers()
        import gc
        gc.collect()
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / 1e9
            print(f"     (RAM in use after fold: {rss:.1f} GB)", flush=True)
        except Exception:
            pass

    results = save_results(args, per_clip, complete=True)

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
