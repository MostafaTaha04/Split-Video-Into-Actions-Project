"""
model.py
========
The activity model: from a video's features (features.py) to its activities
and cuts.

    1. Context. Each half-second sample is described by the image features of
       that moment, averaged over the 4 s before it and the 4 s after it, and
       over the 30 s around it, plus how much the picture moves and how far
       into the video it is. One frame alone rarely shows what a hand is doing;
       a few seconds around it do.
    2. Assembly or disassembly. A first classifier decides whether the PC is
       being put together or taken apart, from the first fifth of the video: an
       assembly starts with an empty case and the parts on the table, a
       disassembly with a closed, full PC. (Over the whole video both pass
       through both states, so the start is what tells them apart.)
    3. Activity. A second classifier gives, for every sample, a probability for
       each of the task's activities and BACKGROUND. (Learning each pair without
       its direction - SCREW and UNSCREW as one "screwing" - and taking the
       direction from step 2 was tried too: it scored lower, so it is off;
       Settings.direction_free turns it on.)
    4. Smoothing and cutting. The labels are chosen for the whole video at once
       (Viterbi decoding): changing activity costs a penalty, and changes that
       are usual in the training videos (UNSCREW -> EXTRACT, say) cost less than
       unusual ones, so a label has to be clearly better for a while before the
       video is cut. Segments shorter than min_segment_s are merged into a
       neighbour. The video is cut wherever the label changes; BACKGROUND parts
       are the gaps between activities.

The classifiers are logistic regressions (scikit-learn) on the image features
reduced to their 128 main directions (PCA). Training on the 8 videos takes
seconds; the model is saved with pickle. The values in Settings were chosen by
testing (run_model.py test) and comparing the scores.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from labels import (ACTION_TYPE, BACKGROUND, TASKS, normalise, task_action_types,
                    with_direction)

ACTION_TYPES = [BACKGROUND] + sorted({v for v in ACTION_TYPE.values() if v != BACKGROUND})


@dataclass
class Settings:
    pca_dims: int = 128           # image features are reduced to this many numbers
    near_s: float = 4.0           # "just before" / "just after" windows
    wide_s: float = 30.0          # the wider window around each moment
    use_position: bool = True     # how far into the video (0 = start, 1 = end)
    use_motion: bool = True
    direction_free: bool = False  # True: learn SCREW and UNSCREW together, direction from the task (tested: worse)
    C: float = 0.1                # logistic regression strength (smaller = simpler model)
    balanced: bool = True         # give rare activities as much weight as common ones
    task_start_share: float = 0.2 # the part of the video, from its start, used to tell assembly from disassembly
    switch_penalty: float = 6.0   # cost of changing label in the smoothing
    order_weight: float = 4.0     # how much the usual order of activities (learned from training) counts
    min_segment_s: float = 2.0    # shorter segments are merged into a neighbour


# --------------------------------------------------------------------------
# labels of one video at its sample times
# --------------------------------------------------------------------------
def labels_at(times: np.ndarray, segments: Sequence[dict]) -> np.ndarray:
    """The annotated label at every sample time (BACKGROUND outside every segment)."""
    out = np.array([BACKGROUND] * len(times), dtype=object)
    for s in segments:
        lab = normalise(s["label"])
        out[(times >= s["start"]) & (times < s["end"])] = lab
    return out


def task_of(annotation: dict, name: str = "") -> str:
    t = str(annotation.get("task", "")).lower()
    if t in TASKS:
        return t
    return "disassembly" if "disassembly" in name else "assembly"


# --------------------------------------------------------------------------
# context features
# --------------------------------------------------------------------------
def _window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """For each row i, the mean of rows i+lo .. i+hi (clipped to the video)."""
    n = len(x)
    c = np.vstack([np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    i = np.arange(n)
    a = np.clip(i + lo, 0, n - 1)
    b = np.clip(i + hi, 0, n - 1)
    b = np.maximum(a, b)
    return ((c[b + 1] - c[a]) / (b - a + 1)[:, None]).astype(np.float32)


def context_features(z: np.ndarray, motion: np.ndarray, fps: float, s: Settings) -> np.ndarray:
    near = max(1, int(round(s.near_s * fps)))
    wide = max(1, int(round(s.wide_s * fps / 2)))
    parts = [z, _window_mean(z, -near, -1), _window_mean(z, 1, near), _window_mean(z, -wide, wide)]
    if s.use_motion:
        m = np.log1p(motion * 20.0)
        parts += [m, _window_mean(m, -near, near)]
    if s.use_position:
        n = len(z)
        pos = (np.arange(n) / max(1, n - 1))[:, None]
        parts += [pos, pos ** 2]
    return np.hstack(parts).astype(np.float32)


# --------------------------------------------------------------------------
# smoothing
# --------------------------------------------------------------------------
def viterbi(logp: np.ndarray, switch_penalty: float, order: Optional[np.ndarray] = None) -> np.ndarray:
    """Best label per sample: sum of log-probabilities minus switch_penalty per change.
    order[i, j] (optional) is added when label i is followed by label j (log of how usual that is)."""
    n, k = logp.shape
    trans = -switch_penalty * (1 - np.eye(k))
    if order is not None:
        trans = trans + order * (1 - np.eye(k))
    score = logp[0].copy()
    back = np.zeros((n, k), dtype=np.int32)
    for t in range(1, n):
        cand = score[:, None] + trans            # from i (rows) to j (columns)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(k)] + logp[t]
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(labels: Sequence[str]) -> List[list]:
    """[[label, first index, last index], ...] for each run of equal labels."""
    out: List[list] = []
    for i, lab in enumerate(labels):
        if out and out[-1][0] == lab:
            out[-1][2] = i
        else:
            out.append([lab, i, i])
    return out


def merge_short(labels: List[str], logp: np.ndarray, classes: List[str], min_len: int) -> List[str]:
    """Give runs shorter than min_len samples to the neighbour the model finds more likely there."""
    labels = list(labels)
    idx = {c: i for i, c in enumerate(classes)}
    for _ in range(50):
        rs = runs(labels)
        short = [r for r in rs if r[2] - r[1] + 1 < min_len]
        if not short or len(rs) == 1:
            break
        r = min(short, key=lambda r: r[2] - r[1])
        j = rs.index(r)
        cands = [rs[k][0] for k in (j - 1, j + 1) if 0 <= k < len(rs)]
        span = logp[r[1]:r[2] + 1]
        best = max(cands, key=lambda c: span[:, idx[c]].sum())
        for i in range(r[1], r[2] + 1):
            labels[i] = best
    return labels


def to_segments(labels: Sequence[str], times: np.ndarray, fps: float, duration: float) -> List[dict]:
    """Activity segments (BACKGROUND runs are left out: they are the gaps)."""
    half = 0.5 / fps
    out = []
    for lab, a, b in runs(labels):
        if lab == BACKGROUND:
            continue
        out.append({"start": round(max(0.0, float(times[a]) - half), 2),
                    "end": round(min(duration, float(times[b]) + half), 2), "label": lab})
    return out


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
@dataclass
class ActivityModel:
    settings: Settings = field(default_factory=Settings)
    scaler_mean: Optional[np.ndarray] = None
    pca: object = None
    ctx_mean: Optional[np.ndarray] = None
    ctx_std: Optional[np.ndarray] = None
    task_clf: object = None
    act_clf: object = None
    classes: List[str] = field(default_factory=list)
    trained_on: List[str] = field(default_factory=list)
    next_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ---- features ----
    def _reduce(self, appearance: np.ndarray) -> np.ndarray:
        return self.pca.transform(appearance.astype(np.float32) - self.scaler_mean)

    def _context(self, feats: dict) -> np.ndarray:
        z = self._reduce(feats["appearance"])
        x = context_features(z, feats["motion"], float(feats["fps"]), self.settings)
        return (x - self.ctx_mean) / self.ctx_std

    def _target(self, label: str) -> str:
        return ACTION_TYPE[label] if self.settings.direction_free else label

    # ---- training ----
    def fit(self, videos: List[dict]) -> "ActivityModel":
        """videos: [{"name", "features", "segments", "task"}, ...]"""
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        s = self.settings
        A = np.vstack([v["features"]["appearance"].astype(np.float32) for v in videos])
        self.scaler_mean = A.mean(axis=0)
        self.pca = PCA(n_components=s.pca_dims, random_state=0).fit(A - self.scaler_mean)
        self.ctx_mean, self.ctx_std = 0.0, 1.0
        X = [self._context(v["features"]) for v in videos]
        allx = np.vstack(X)
        self.ctx_mean, self.ctx_std = allx.mean(axis=0), allx.std(axis=0) + 1e-6
        X = [(x - self.ctx_mean) / self.ctx_std for x in X]
        Y = [np.array([self._target(l) for l in labels_at(v["features"]["times"], v["segments"])])
             for v in videos]
        T = [np.array([v["task"]] * len(x)) for v, x in zip(videos, X)]
        Xa, Ya, Ta = np.vstack(X), np.concatenate(Y), np.concatenate(T)
        cw = "balanced" if s.balanced else None
        Xs = np.vstack([x[:self._start_len(len(x))] for x in X])
        Ts = np.concatenate([t[:self._start_len(len(t))] for t in T])
        self.task_clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(Xs, Ts)
        self.act_clf = LogisticRegression(C=s.C, max_iter=3000, class_weight=cw).fit(Xa, Ya)
        self.classes = list(self.act_clf.classes_)
        self.trained_on = [v["name"] for v in videos]
        # which activity follows which, in the training videos
        self.next_counts = {}
        for y in Y:
            seq = [r[0] for r in runs(list(y))]
            for a, b in zip(seq, seq[1:]):
                self.next_counts.setdefault(a, {}).setdefault(b, 0.0)
                self.next_counts[a][b] += 1.0
        return self

    def _order(self, classes: List[str]) -> Optional[np.ndarray]:
        w = self.settings.order_weight
        if not w:
            return None
        k = len(classes)
        m = np.zeros((k, k))
        for i, a in enumerate(classes):
            row = self.next_counts.get(a, {})
            tot = sum(row.get(b, 0.0) for b in classes if b != a) + 0.5 * (k - 1)
            for j, b in enumerate(classes):
                if i != j:
                    m[i, j] = w * np.log((row.get(b, 0.0) + 0.5) / tot)
        return m

    # ---- use ----
    def _start_len(self, n: int) -> int:
        return min(n, max(20, int(round(n * self.settings.task_start_share))))

    def predict_task(self, feats: dict, x: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Probability of assembly / disassembly, from the start of the video."""
        x = self._context(feats) if x is None else x
        x = x[:self._start_len(len(x))]
        lp = np.log(np.clip(self.task_clf.predict_proba(x), 1e-6, 1.0)).mean(axis=0)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.task_clf.classes_, p)}

    def predict(self, feats: dict, duration: Optional[float] = None, task: Optional[str] = None) -> dict:
        """Labels per sample, segments and the task of one video."""
        s = self.settings
        times = feats["times"]
        fps = float(feats["fps"])
        duration = float(duration if duration is not None else times[-1] + 0.5 / fps)
        x = self._context(feats)
        task_probs = self.predict_task(feats, x)
        task = task or max(task_probs, key=task_probs.get)
        prob = self.act_clf.predict_proba(x)
        # only the actions that can happen in this task
        if s.direction_free:
            allowed = set(task_action_types(task))
        else:
            from labels import TASK_ACTIVITIES
            allowed = set(TASK_ACTIVITIES[task]) | {BACKGROUND}
        keep = [i for i, c in enumerate(self.classes) if c in allowed]
        classes = [self.classes[i] for i in keep]
        p = prob[:, keep]
        p = p / p.sum(axis=1, keepdims=True)
        logp = np.log(np.clip(p, 1e-6, 1.0))
        path = viterbi(logp, s.switch_penalty, self._order(classes))
        lab = [classes[i] for i in path]
        lab = merge_short(lab, logp, classes, max(1, int(round(s.min_segment_s * fps))))
        if s.direction_free:
            lab = [with_direction(l, task) for l in lab]
        return {"task": task, "task_probs": task_probs, "times": times, "labels": lab,
                "segments": to_segments(lab, times, fps, duration)}

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "ActivityModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
