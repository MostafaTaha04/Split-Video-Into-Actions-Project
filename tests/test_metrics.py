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


class TestStandardMetrics(unittest.TestCase):
    """Segmental F1@k, Hungarian frame accuracy and GEBD F1 (utils.py)."""

    def test_boundaries_to_segments(self):
        segs = MetricsCalculator.boundaries_to_segments([3.0, 1.0, 12.0], 10.0)
        # 12.0 lies outside the video and is dropped; output is sorted
        self.assertEqual(segs, [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0)])

    def test_segmental_f1_perfect(self):
        gt = [(0.0, 5.0), (5.0, 10.0)]
        for k in (0.10, 0.25, 0.50):
            self.assertEqual(MetricsCalculator.segmental_f1(gt, gt, k), 1.0)

    def test_segmental_f1_over_segmentation(self):
        # GT: two 5 s steps. Prediction splits each in half (4 pieces).
        gt = [(0.0, 5.0), (5.0, 10.0)]
        pred = [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]
        # Each piece has IoU 0.5 with its step; only one piece per step can
        # claim it -> TP=2, FP=2, FN=0 -> P=0.5, R=1 -> F1=2/3 at IoU<=0.5.
        self.assertAlmostEqual(MetricsCalculator.segmental_f1(pred, gt, 0.50), 2 / 3, places=6)
        # At a stricter overlap nothing qualifies.
        self.assertEqual(MetricsCalculator.segmental_f1(pred, gt, 0.75), 0.0)

    def test_segmental_f1_empty(self):
        self.assertEqual(MetricsCalculator.segmental_f1([], [(0.0, 1.0)], 0.1), 0.0)

    def test_frame_accuracy_perfect_and_split(self):
        gt = [(0.0, 5.0), (5.0, 10.0)]
        self.assertAlmostEqual(MetricsCalculator.frame_accuracy_hungarian(gt, gt), 1.0)
        # One predicted segment covering everything can be matched to only
        # one step -> half the time is correct.
        self.assertAlmostEqual(
            MetricsCalculator.frame_accuracy_hungarian([(0.0, 10.0)], gt), 0.5)
        # Splitting each step in half: each step keeps one half -> 0.5.
        pred = [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]
        self.assertAlmostEqual(MetricsCalculator.frame_accuracy_hungarian(pred, gt), 0.5)

    def test_gebd_f1_relative_tolerance(self):
        # 100 s video: Rel.Dis 0.05 -> 5 s tolerance. A 4 s error counts;
        # a 6 s error only counts from Rel.Dis 0.10 upwards.
        out = MetricsCalculator.gebd_f1([14.0, 56.0], [10.0, 50.0], 100.0)
        self.assertAlmostEqual(out["0.05"], 0.5)
        self.assertAlmostEqual(out["0.10"], 1.0)
        self.assertEqual(len(out), 11)  # 10 thresholds + "avg"
        self.assertAlmostEqual(out["avg"], (0.5 + 9 * 1.0) / 10)


class TestImportSegments(unittest.TestCase):
    """import_segments.py: external (start, end, label) tables -> ground truth."""

    def test_time_formats(self):
        from import_segments import parse_time
        self.assertEqual(parse_time("12.5"), 12.5)
        self.assertEqual(parse_time("01:02.5"), 62.5)
        self.assertEqual(parse_time("1:00:01"), 3601.0)
        self.assertIsNone(parse_time("start"))

    def test_table_with_header_and_gaps(self):
        from import_segments import read_table, to_steps
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("start,end,keystep\n2.0,10.0,open case\n10.2,20.0,fix hard disk\n25.0,30.0,close case\n")
            path = fh.name
        try:
            segs = read_table(path)
        finally:
            os.unlink(path)
        self.assertEqual(segs[0], (2.0, 10.0, "open case"))
        steps = to_steps(segs, 32.0, min_gap=0.5)
        labels = [s["label"] for s in steps]
        # leading and inner gaps >= 0.5 s become background; 0.2 s gap is closed
        self.assertEqual(labels, ["background", "open case", "fix hard disk",
                                  "background", "close case", "background"])
        # contiguous tiling of the whole video
        self.assertEqual(steps[0]["start"], 0.0)
        self.assertEqual(steps[-1]["end"], 32.0)
        for a, b in zip(steps, steps[1:]):
            self.assertEqual(a["end"], b["start"])
