"""
Unit tests for the deterministic evaluation pieces.

Run from the project root with either:
    python -m pytest tests/
    python -m unittest discover -s tests

These tests cover the parts the reported numbers depend on: boundary
precision/recall/F1, segment IoU, the uniform baseline, and the ground-truth
loader (both JSON formats). No video, model, or GPU is required.
"""
import json
import os
import sys
import tempfile
import unittest

# Make the project root importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import MetricsCalculator
from evaluate_baselines import uniform_boundaries
from evaluator import Evaluator


class TestBoundaryAccuracy(unittest.TestCase):
    def test_perfect_match(self):
        m = MetricsCalculator.boundary_accuracy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], tolerance=0.5)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1_score"], 1.0)
        self.assertEqual(m["true_positives"], 3)
        self.assertEqual(m["false_positives"], 0)
        self.assertEqual(m["false_negatives"], 0)

    def test_within_tolerance(self):
        # 0.4 s off, tolerance 0.5 -> still a match.
        m = MetricsCalculator.boundary_accuracy([1.4], [1.0], tolerance=0.5)
        self.assertEqual(m["true_positives"], 1)

    def test_outside_tolerance(self):
        # 0.6 s off, tolerance 0.5 -> no match.
        m = MetricsCalculator.boundary_accuracy([1.6], [1.0], tolerance=0.5)
        self.assertEqual(m["true_positives"], 0)
        self.assertEqual(m["false_positives"], 1)
        self.assertEqual(m["false_negatives"], 1)

    def test_partial(self):
        # 2 preds, 1 correct -> P=0.5, R=1.0
        m = MetricsCalculator.boundary_accuracy([1.0, 9.0], [1.0], tolerance=0.5)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1_score"], 2 * 0.5 * 1.0 / 1.5, places=3)

    def test_no_double_count(self):
        # Two predictions near the same single GT must only score one TP.
        m = MetricsCalculator.boundary_accuracy([1.0, 1.1], [1.0], tolerance=0.5)
        self.assertEqual(m["true_positives"], 1)
        self.assertEqual(m["false_positives"], 1)

    def test_empty_predictions(self):
        m = MetricsCalculator.boundary_accuracy([], [1.0, 2.0], tolerance=0.5)
        self.assertEqual(m["precision"], 0)
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["false_negatives"], 2)

    def test_no_ground_truth(self):
        m = MetricsCalculator.boundary_accuracy([1.0], [], tolerance=0.5)
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["false_positives"], 1)


class TestSegmentIoU(unittest.TestCase):
    def test_identical(self):
        iou = MetricsCalculator.segment_iou([(0.0, 10.0)], [(0.0, 10.0)])
        self.assertAlmostEqual(iou, 1.0, places=6)

    def test_disjoint(self):
        iou = MetricsCalculator.segment_iou([(0.0, 5.0)], [(10.0, 15.0)])
        self.assertEqual(iou, 0.0)

    def test_half_overlap(self):
        # pred [0,10], gt [5,15]: inter=5, union=15 -> 1/3
        iou = MetricsCalculator.segment_iou([(0.0, 10.0)], [(5.0, 15.0)])
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=6)

    def test_empty(self):
        self.assertEqual(MetricsCalculator.segment_iou([], [(0.0, 1.0)]), 0.0)


class TestUniformBaseline(unittest.TestCase):
    def test_count_and_spacing(self):
        # 3 interior boundaries splitting [0, 12] into 4 equal parts.
        b = uniform_boundaries(12.0, 3)
        self.assertEqual(len(b), 3)
        self.assertEqual(b, [3.0, 6.0, 9.0])

    def test_zero(self):
        self.assertEqual(uniform_boundaries(10.0, 0), [])


class TestGroundTruthLoader(unittest.TestCase):
    def _write(self, data):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_steps_format(self):
        path = self._write({
            "steps": [
                {"start": 0.0, "end": 4.0, "label": "a"},
                {"start": 4.0, "end": 9.0, "label": "b"},
                {"start": 9.0, "end": 12.0, "label": "c"},
            ]
        })
        ev = Evaluator(path)
        self.assertEqual(ev.gt_segments, [(0.0, 4.0), (4.0, 9.0), (9.0, 12.0)])
        # Boundaries are the interior step-end times (all ends except the last).
        self.assertEqual(ev.gt_boundaries, [4.0, 9.0])
        self.assertEqual(ev.gt_labels, ["a", "b", "c"])

    def test_segments_boundaries_format(self):
        path = self._write({
            "segments": [
                {"start_time": 0.0, "end_time": 4.0, "activity": "a"},
                {"start_time": 4.0, "end_time": 9.0, "activity": "b"},
            ],
            "boundaries": [{"timestamp": 4.0}],
        })
        ev = Evaluator(path)
        self.assertEqual(ev.gt_segments, [(0.0, 4.0), (4.0, 9.0)])
        self.assertEqual(ev.gt_boundaries, [4.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
