"""
Integration tests for the offline re-segmentation used by evaluate_extended.py.

These guard against drift in the boundary-detection logic: re-running the
segmenter on a saved features.csv must reproduce the saved boundaries, and the
helper functions (F1, uniform baseline, ground-truth loading) must behave.

No video, model, mediapipe, or ultralytics needed — only the saved features.
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluate_extended as ee  # noqa: E402

CF_DIR = os.path.join(ROOT, "results_coolingfan_v2run")
HAVE_CF = os.path.exists(os.path.join(CF_DIR, "features.csv"))


@unittest.skipUnless(HAVE_CF, "cooling-fan results not present")
class TestReSegmentation(unittest.TestCase):
    def test_reproduces_saved_boundaries(self):
        F = ee.load_features(os.path.join(CF_DIR, "features.csv"))
        pred = [round(b, 3) for b in ee.segment(F, 10.0, threshold=0.55, min_dur=2.5, sigma=2.0)]
        with open(os.path.join(CF_DIR, "segmentation_results.json")) as fh:
            saved = [round(float(b["timestamp"]), 3) for b in json.load(fh)["boundaries"]]
        self.assertEqual(pred, saved)

    def test_segment_returns_sorted_unique(self):
        F = ee.load_features(os.path.join(CF_DIR, "features.csv"))
        pred = ee.segment(F, 10.0, threshold=0.70, min_dur=2.0)
        self.assertEqual(pred, sorted(pred))
        self.assertEqual(len(pred), len(set(pred)))


class TestHelpers(unittest.TestCase):
    def test_uniform_spacing(self):
        self.assertEqual(ee.uniform(12.0, 3), [3.0, 6.0, 9.0])
        self.assertEqual(ee.uniform(10.0, 0), [])

    def test_f1_perfect(self):
        self.assertEqual(ee.f1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.5), 1.0)

    def test_f1_partial(self):
        # 2 preds, 1 within tolerance -> P=0.5, R=1.0 -> F1=2/3
        self.assertAlmostEqual(ee.f1([1.0, 9.0], [1.0], 0.5), 2 / 3, places=3)

    def test_f1_empty(self):
        self.assertEqual(ee.f1([], [1.0], 0.5), 0.0)

    def test_gt_boundaries_interior_ends(self):
        gtf = os.path.join(ROOT, "ground_truth_coolingfan_v2.json")
        if os.path.exists(gtf):
            gt = ee.gt_boundaries(ROOT, "ground_truth_coolingfan_v2.json")
            # 8 steps -> 7 interior boundaries
            self.assertEqual(len(gt), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
