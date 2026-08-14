"""Tests for the learned boundary scorer and its integration with the segmenter.

These skip cleanly if scikit-learn is unavailable or the model has not been
trained, so CI stays green on a minimal environment.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "boundary_model.joblib")

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

from boundary_model import FEATURE_COLUMNS, OFFSETS, add_context, blend  # noqa: E402


class TestFeatureConstruction(unittest.TestCase):
    def test_context_shape(self):
        X = np.random.rand(50, len(FEATURE_COLUMNS))
        out = add_context(X)
        # len(OFFSETS) lagged copies + first difference + |first difference|
        self.assertEqual(out.shape, (50, len(FEATURE_COLUMNS) * (len(OFFSETS) + 2)))

    def test_context_preserves_length(self):
        for n in (1, 2, 5, 37, 200):
            X = np.random.rand(n, len(FEATURE_COLUMNS))
            self.assertEqual(len(add_context(X)), n)

    def test_context_matches_learned_baseline(self):
        """The training and inference paths must build identical features."""
        import learned_baseline as lb
        X = np.random.RandomState(0).rand(40, len(FEATURE_COLUMNS))
        np.testing.assert_allclose(add_context(X), lb.add_context(X))

    def test_offsets_are_centred(self):
        self.assertIn(0, OFFSETS)


class TestBlend(unittest.TestCase):
    def test_endpoints(self):
        a = np.array([0.1, 0.9, 0.5])
        b = np.array([0.9, 0.1, 0.5])
        np.testing.assert_allclose(blend(a, b, 1.0), a)
        np.testing.assert_allclose(blend(a, b, 0.0), b)

    def test_midpoint(self):
        a, b = np.array([0.2]), np.array([0.8])
        np.testing.assert_allclose(blend(a, b, 0.5), np.array([0.5]))

    def test_output_is_bounded(self):
        a = np.array([5.0, -3.0, 0.4])
        b = np.array([-2.0, 9.0, 0.6])
        out = blend(a, b, 0.5)
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))


def model_is_usable():
    """True only if the saved model loads AND runs under the installed sklearn.

    A model pickled by a different scikit-learn version loads but fails on
    first use, so existence of the file is not sufficient.
    """
    if not (HAVE_SKLEARN and os.path.exists(MODEL)):
        return False
    try:
        from boundary_model import BoundaryScorer
        BoundaryScorer(MODEL)
        return True
    except BaseException:
        # SystemExit included: BoundaryScorer raises it for version mismatch.
        return False


USABLE = model_is_usable()


@unittest.skipUnless(
    USABLE,
    "needs scikit-learn and a boundary_model.joblib trained with the installed "
    "scikit-learn version (run: python train_boundary_model.py --src .)")
class TestScorerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import csv

        from feature_extractor import FrameFeatures

        path = os.path.join(REPO, "results_coolingfan_v2run", "features.csv")
        feats = []
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                f = FrameFeatures(frame_idx=int(float(row["frame_idx"])),
                                  timestamp=float(row["timestamp"]))
                for k, v in row.items():
                    if k in ("frame_idx", "timestamp") or not hasattr(f, k):
                        continue
                    cur = getattr(f, k)
                    try:
                        if isinstance(cur, bool):
                            setattr(f, k, float(v) > 0.5)
                        elif isinstance(cur, (int, float)):
                            setattr(f, k, type(cur)(float(v)))
                    except (TypeError, ValueError):
                        pass
                feats.append(f)
        cls.features = feats

    def _scorer(self):
        from boundary_model import BoundaryScorer
        return BoundaryScorer(MODEL)

    def test_score_shape_and_range(self):
        p = self._scorer().score(self.features)
        self.assertEqual(len(p), len(self.features))
        self.assertTrue(np.all(p >= 0.0) and np.all(p <= 1.0))

    def test_learned_mode_changes_boundaries(self):
        from temporal_segmenter import TemporalSegmenter

        rule = TemporalSegmenter(boundary_threshold=0.55, min_segment_duration=2.5,
                                 fps=10.0)
        _, b_rule = rule.segment(self.features)

        learned = TemporalSegmenter(boundary_threshold=0.55, min_segment_duration=2.5,
                                    fps=10.0)
        learned.set_learned_scorer(self._scorer(), mode="learned")
        _, b_learned = learned.segment(self.features)

        self.assertNotEqual([b.timestamp for b in b_rule],
                            [b.timestamp for b in b_learned])

    def test_hybrid_endpoints_match_pure_modes(self):
        """w_rule=1 must reproduce the rule-based result exactly."""
        from temporal_segmenter import TemporalSegmenter

        rule = TemporalSegmenter(boundary_threshold=0.55, min_segment_duration=2.5,
                                 fps=10.0)
        _, b_rule = rule.segment(self.features)

        hyb = TemporalSegmenter(boundary_threshold=0.55, min_segment_duration=2.5,
                                fps=10.0)
        hyb.set_learned_scorer(self._scorer(), mode="hybrid", w_rule=1.0)
        _, b_hyb = hyb.segment(self.features)

        self.assertEqual([round(b.timestamp, 6) for b in b_rule],
                         [round(b.timestamp, 6) for b in b_hyb])

    def test_segments_still_tile_under_learned_scorer(self):
        from temporal_segmenter import TemporalSegmenter

        seg = TemporalSegmenter(boundary_threshold=0.55, min_segment_duration=2.5,
                                fps=10.0)
        seg.set_learned_scorer(self._scorer(), mode="hybrid", w_rule=0.7)
        segments, boundaries = seg.segment(self.features)
        self.assertEqual(len(segments), len(boundaries) + 1)
        for a, b in zip(segments, segments[1:]):
            self.assertAlmostEqual(a.end_time, b.start_time, places=6)

    def test_model_metadata_present(self):
        s = self._scorer()
        self.assertIn("model_type", s.meta)
        self.assertGreater(s.meta.get("n_frames", 0), 0)
        self.assertIsInstance(s.describe(), str)


if __name__ == "__main__":
    unittest.main()
