"""
boundary_model.py
=================
The learned half of the segmentation method.

The pipeline can produce its per-frame boundary score in three ways, selected
with ``main.py --scorer``:

  ``rule``    the hand-designed fusion in ``temporal_segmenter.py`` (default).
  ``learned`` a trained classifier's probability that a frame lies on a step
              boundary.
  ``hybrid``  a convex blend of the two: ``w * rule + (1 - w) * learned``.

Why a learned scorer exists
---------------------------
The hand-designed fusion and the learned scorer make different mistakes, so
blending them beats either alone. Under leave-one-clip-out the hybrid currently
wins on every clip in the evaluation set.

Numbers are deliberately NOT quoted here. They have already changed twice: once
when four redundant fusion channels were removed, and again when an aspect-ratio
bug in the frame loader was fixed (portrait clips were squashed to landscape,
which destroyed every hand-derived feature). Hardcoding results in a docstring
guarantees they drift. For current figures see ``learned_results.json`` or run:

    python learned_baseline.py --src .

One caveat survives any re-measurement: with five clips no difference between
scorers reaches conventional significance, so claims should be about consistency
(how often a scorer wins) rather than the size of the gap.

Feature construction is shared with ``learned_baseline.py`` so that a model
trained by ``train_boundary_model.py`` sees exactly the representation it was
trained on at inference time.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np

# Frame-level columns the model consumes, in a FIXED order. Changing this list
# invalidates any saved model, which is why the order is persisted alongside
# the weights and checked on load.
FEATURE_COLUMNS: List[str] = [
    "hand_velocity_left", "hand_velocity_right",
    "hand_acceleration_left", "hand_acceleration_right",
    "hands_present", "grip_state_left", "grip_state_right",
    "hand_distance", "trajectory_curvature",
    "num_tools", "tool_changed", "tool_stability",
    "num_interactions", "contact_point_shift", "contact_point_variance",
    "interaction_density", "interaction_rhythm",
    "flow_magnitude", "flow_uniformity", "flow_discontinuity",
    "direction_change", "scene_change_score", "visual_stability",
]

# Temporal offsets (in processed frames) stacked to give the classifier a
# receptive field of roughly one second at 10 fps.
OFFSETS = (-4, -2, 0, 2, 4)

DEFAULT_MODEL_PATH = "boundary_model.joblib"


def frame_matrix(features: Sequence) -> np.ndarray:
    """Extract the raw per-frame feature matrix from FrameFeatures objects."""
    return np.array(
        [[float(getattr(f, c)) for c in FEATURE_COLUMNS] for f in features],
        dtype=float,
    )


def rows_matrix(rows: Sequence[dict]) -> np.ndarray:
    """Same, but from features.csv dict rows (used by the offline tooling)."""
    return np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows], dtype=float)


def add_context(X: np.ndarray) -> np.ndarray:
    """Stack lagged/led copies plus local first differences.

    Must stay identical to ``learned_baseline.add_context`` — a mismatch here
    silently feeds the model a different representation than it was trained on,
    which degrades accuracy without raising an error.
    """
    n = len(X)
    parts = []
    for off in OFFSETS:
        idx = np.clip(np.arange(n) + off, 0, n - 1)
        parts.append(X[idx])
    d1 = np.diff(X, axis=0, prepend=X[:1])
    parts.append(d1)
    parts.append(np.abs(d1))
    return np.hstack(parts)


class BoundaryScorer:
    """Loads a trained model and turns per-frame features into a 0..1 score."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "The learned scorer needs scikit-learn/joblib:\n"
                "    pip install scikit-learn\n"
                f"(original error: {exc})"
            )

        if not os.path.exists(model_path):
            raise SystemExit(
                f"No trained boundary model at '{model_path}'.\n"
                "Train one first:\n"
                "    python train_boundary_model.py --src .\n"
                "or run with the default hand-designed scorer: --scorer rule"
            )

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.columns = bundle["feature_columns"]
        self.offsets = tuple(bundle["offsets"])
        self.meta = bundle.get("meta", {})

        if self.columns != FEATURE_COLUMNS or self.offsets != OFFSETS:
            raise SystemExit(
                f"Saved model '{model_path}' was trained on a different feature "
                "layout than this code produces. Retrain it:\n"
                "    python train_boundary_model.py --src ."
            )

        self._check_usable(model_path)

    def _check_usable(self, model_path: str):
        """Verify the estimator actually runs under the installed scikit-learn.

        A pickled estimator is tied to the version that created it. Loading a
        newer pickle under an older scikit-learn succeeds, then fails deep
        inside predict_proba with an obscure AttributeError (for example
        'LogisticRegression' object has no attribute 'multi_class'). Probing
        once here converts that into an actionable message at construction
        time. This is also why boundary_model.joblib is not committed to the
        repository — it is regenerated locally in seconds.
        """
        n_features = len(self.columns) * (len(self.offsets) + 2)
        probe = np.zeros((1, n_features), dtype=float)
        try:
            self.model.predict_proba(self.scaler.transform(probe))
        except Exception as exc:
            import sklearn

            trained_with = self.meta.get(
                "sklearn_version", getattr(self.model, "__sklearn_version__", "unknown"))
            raise SystemExit(
                f"The saved model '{model_path}' cannot run under the installed "
                f"scikit-learn.\n"
                f"  installed    : {sklearn.__version__}\n"
                f"  model built with: {trained_with}\n"
                f"  error        : {type(exc).__name__}: {exc}\n\n"
                "A pickled estimator is not portable across scikit-learn versions.\n"
                "Retrain it locally (takes seconds):\n"
                "    python train_boundary_model.py --src .\n"
                "or run with the hand-designed scorer: --scorer rule"
            ) from exc

    def score(self, features: Sequence) -> np.ndarray:
        """Per-frame probability that the frame lies on a step boundary."""
        X = add_context(frame_matrix(features))
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def describe(self) -> str:
        m = self.meta
        return (f"{m.get('model_type', 'model')} trained on {m.get('n_clips', '?')} clips, "
                f"{m.get('n_frames', '?')} frames "
                f"({m.get('n_positive', '?')} positive @ tol {m.get('label_tol', '?')}s)")


def blend(rule_score: np.ndarray, learned_score: np.ndarray, w_rule: float) -> np.ndarray:
    """Convex blend of the two scores, clipped to the unit interval."""
    return np.clip(w_rule * np.asarray(rule_score) + (1.0 - w_rule) * np.asarray(learned_score),
                   0.0, 1.0)


def load_default_blend_weight(model_path: str = DEFAULT_MODEL_PATH) -> Optional[float]:
    """Blend weight chosen during training (cross-validated), if recorded."""
    sidecar = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as fh:
            return json.load(fh).get("blend_weight_rule")
    return None
