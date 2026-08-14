"""Unit tests for TemporalSegmenter — fusion, peak detection, and the
segment-construction edge cases that produced bugs during development.

These tests import only numpy/scipy (temporal_segmenter has no heavy vision
dependencies), so they run in CI without mediapipe or ultralytics.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SegmenterParams  # noqa: E402
from feature_extractor import FrameFeatures  # noqa: E402
from temporal_segmenter import TemporalSegmenter  # noqa: E402


def make_features(n, fps=10.0, transition=None, activity=None, **kw):
    """Build a synthetic FrameFeatures sequence."""
    feats = []
    for i in range(n):
        f = FrameFeatures(frame_idx=i, timestamp=i / fps)
        f.transition_score = 0.0 if transition is None else float(transition[i])
        f.activity_level = 0.0 if activity is None else float(activity[i])
        for k, v in kw.items():
            setattr(f, k, v[i] if isinstance(v, (list, np.ndarray)) else v)
        feats.append(f)
    return feats


class TestFusionChannels(unittest.TestCase):
    def test_only_two_channels_remain(self):
        """The fusion was reduced to the two load-bearing channels."""
        seg = TemporalSegmenter(fps=10.0)
        feats = make_features(50, transition=np.zeros(50), activity=np.zeros(50))
        ch = seg._boundary_score_channels(feats)
        self.assertEqual(set(ch), {"transition", "activity_change"})

    def test_channels_have_frame_length(self):
        seg = TemporalSegmenter(fps=10.0)
        feats = make_features(37, transition=np.random.rand(37), activity=np.random.rand(37))
        for name, arr in seg._boundary_score_channels(feats).items():
            self.assertEqual(len(arr), 37, f"channel {name} wrong length")

    def test_fusion_is_max_of_channels(self):
        seg = TemporalSegmenter(fps=10.0)
        t = np.random.rand(60)
        a = np.random.rand(60)
        feats = make_features(60, transition=t, activity=a)
        ch = seg._boundary_score_channels(feats)
        expected = np.maximum(ch["transition"], ch["activity_change"])
        got = seg._build_boundary_score(feats)
        warm = max(3, int(0.5 * 10.0))
        np.testing.assert_allclose(got[warm:], np.clip(expected, 0, 1)[warm:], atol=1e-9)

    def test_score_is_bounded(self):
        seg = TemporalSegmenter(fps=10.0)
        feats = make_features(80, transition=np.full(80, 5.0), activity=np.random.rand(80) * 9)
        score = seg._build_boundary_score(feats)
        self.assertTrue(np.all(score >= 0.0) and np.all(score <= 1.0))

    def test_warmup_is_zeroed(self):
        """Tracker/flow warm-up must never produce a boundary."""
        seg = TemporalSegmenter(fps=20.0)
        feats = make_features(100, transition=np.ones(100), activity=np.ones(100))
        score = seg._build_boundary_score(feats)
        warm = max(3, int(0.5 * 20.0))
        self.assertTrue(np.all(score[:warm] == 0.0))
        self.assertGreater(score[warm + 5], 0.0)

    def test_activity_channel_weight_is_configurable(self):
        # A single isolated spike would be flattened by the 5th/95th-percentile
        # normalisation, so use a signal with genuine spread.
        a = np.linspace(0.0, 1.0, 60) + np.random.RandomState(0).rand(60) * 0.3
        feats = make_features(60, transition=np.zeros(60), activity=a)
        hi = TemporalSegmenter(fps=10.0, params=SegmenterParams(
            channel_activity_change_weight=0.80))._build_boundary_score(feats)
        lo = TemporalSegmenter(fps=10.0, params=SegmenterParams(
            channel_activity_change_weight=0.20))._build_boundary_score(feats)
        self.assertGreater(hi.max(), lo.max())


class TestNormalize(unittest.TestCase):
    def test_constant_signal_gives_zeros(self):
        out = TemporalSegmenter._normalize(np.full(50, 7.0))
        np.testing.assert_allclose(out, np.zeros(50))

    def test_empty(self):
        self.assertEqual(TemporalSegmenter._normalize(np.array([])).size, 0)

    def test_range_is_clipped_to_unit(self):
        out = TemporalSegmenter._normalize(np.linspace(-10, 10, 200))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)


class TestSegmentation(unittest.TestCase):
    def _spiky(self, n=200, peaks=(50, 100, 150), fps=10.0):
        t = np.zeros(n)
        for p in peaks:
            t[p] = 1.0
        return make_features(n, fps=fps, transition=t, activity=np.zeros(n))

    def test_detects_planted_boundaries(self):
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=1.0, fps=10.0)
        segments, boundaries = seg.segment(self._spiky())
        self.assertEqual(len(boundaries), 3)
        for b, expected in zip(boundaries, (5.0, 10.0, 15.0)):
            self.assertAlmostEqual(b.timestamp, expected, delta=0.3)

    def test_segments_partition_the_timeline(self):
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=1.0, fps=10.0)
        segments, boundaries = seg.segment(self._spiky())
        self.assertEqual(len(segments), len(boundaries) + 1)
        for a, b in zip(segments, segments[1:]):
            self.assertAlmostEqual(a.end_time, b.start_time, places=6)
        for s in segments:
            self.assertGreater(s.end_time, s.start_time)

    def test_higher_threshold_gives_no_more_boundaries(self):
        feats = self._spiky()
        counts = [
            len(TemporalSegmenter(boundary_threshold=thr, min_segment_duration=1.0,
                                  fps=10.0).segment(feats)[1])
            for thr in (0.05, 0.2, 0.5, 0.95)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_min_duration_enforced(self):
        """No two boundaries closer than min_segment_duration."""
        n = 300
        t = np.zeros(n)
        t[::7] = 1.0  # a spike every 0.7 s at 10 fps
        feats = make_features(n, fps=10.0, transition=t, activity=np.zeros(n))
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=3.0, fps=10.0)
        _, boundaries = seg.segment(feats)
        times = [b.timestamp for b in boundaries]
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(b - a, 3.0 - 1e-6)

    def test_edge_boundaries_removed(self):
        """Boundaries within min_duration of either end are dropped."""
        n = 200
        t = np.zeros(n)
        t[5] = 1.0          # 0.5 s — too close to the start
        t[100] = 1.0        # 10.0 s — valid
        t[195] = 1.0        # 19.5 s — too close to the end
        feats = make_features(n, fps=10.0, transition=t, activity=np.zeros(n))
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=2.0, fps=10.0)
        _, boundaries = seg.segment(feats)
        self.assertEqual(len(boundaries), 1)
        self.assertAlmostEqual(boundaries[0].timestamp, 10.0, delta=0.3)

    def test_flat_signal_gives_single_segment(self):
        feats = make_features(200, transition=np.zeros(200), activity=np.zeros(200))
        seg = TemporalSegmenter(boundary_threshold=0.3, min_segment_duration=1.5, fps=10.0)
        segments, boundaries = seg.segment(feats)
        self.assertEqual(boundaries, [])
        self.assertEqual(len(segments), 1)

    def test_empty_input(self):
        segments, boundaries = TemporalSegmenter(fps=10.0).segment([])
        self.assertEqual(segments, [])
        self.assertEqual(boundaries, [])

    def test_input_shorter_than_min_segment(self):
        """A clip shorter than one minimum segment yields exactly one segment."""
        feats = make_features(5, fps=10.0, transition=np.ones(5), activity=np.ones(5))
        seg = TemporalSegmenter(min_segment_duration=3.0, fps=10.0)
        segments, boundaries = seg.segment(feats)
        self.assertEqual(len(segments), 1)
        self.assertEqual(boundaries, [])

    def test_boundaries_are_sorted_and_unique(self):
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=1.0, fps=10.0)
        _, boundaries = seg.segment(self._spiky(peaks=(30, 60, 90, 120, 160)))
        times = [b.timestamp for b in boundaries]
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(times), len(set(times)))

    def test_every_boundary_has_a_reason(self):
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=1.0, fps=10.0)
        _, boundaries = seg.segment(self._spiky())
        for b in boundaries:
            self.assertTrue(b.reason, "boundary has empty reason string")
            self.assertIsInstance(b.signal_strengths, dict)
            self.assertGreater(len(b.signal_strengths), 0)


class TestSegmentConfidence(unittest.TestCase):
    def test_confidence_in_unit_range(self):
        seg = TemporalSegmenter(boundary_threshold=0.10, min_segment_duration=1.0, fps=10.0)
        n = 200
        t = np.zeros(n)
        t[[50, 100, 150]] = 1.0
        feats = make_features(n, fps=10.0, transition=t, activity=np.random.rand(n))
        segments, _ = seg.segment(feats)
        for s in segments:
            self.assertGreaterEqual(s.confidence, 0.0)
            self.assertLessEqual(s.confidence, 1.0)

    def test_steady_segment_scores_above_erratic(self):
        seg = TemporalSegmenter(fps=10.0)
        steady = make_features(60, activity=np.full(60, 0.5), transition=np.zeros(60))
        erratic = make_features(60, activity=np.random.RandomState(0).rand(60),
                                transition=np.zeros(60))
        self.assertGreater(seg._compute_segment_confidence(steady),
                           seg._compute_segment_confidence(erratic))


class TestDominantActivity(unittest.TestCase):
    def test_idle_when_no_hands_and_no_flow(self):
        seg = TemporalSegmenter(fps=10.0)
        feats = make_features(40, transition=np.zeros(40), activity=np.zeros(40),
                              hands_present=0, flow_magnitude=0.0)
        self.assertEqual(seg._determine_dominant_activity(feats), "idle_no_hands")

    def test_active_assembly_when_interacting_and_moving(self):
        seg = TemporalSegmenter(fps=10.0)
        feats = make_features(40, transition=np.zeros(40), activity=np.full(40, 0.6),
                              hands_present=2, flow_magnitude=6.0,
                              num_interactions=2, hand_velocity_right=30.0)
        self.assertEqual(seg._determine_dominant_activity(feats), "active_assembly")

    def test_thresholds_come_from_params(self):
        """Raising the idle flow threshold reclassifies a borderline segment."""
        feats = make_features(40, transition=np.zeros(40), activity=np.zeros(40),
                              hands_present=0, flow_magnitude=2.0)
        default = TemporalSegmenter(fps=10.0)
        raised = TemporalSegmenter(fps=10.0, params=SegmenterParams(dominant_idle_flow=5.0))
        self.assertNotEqual(default._determine_dominant_activity(feats),
                            raised._determine_dominant_activity(feats))


if __name__ == "__main__":
    unittest.main()
