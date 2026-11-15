"""
Tests for the appearance scorer (appearance.py) and for the guarantees that
keep it honest:

* the live path (TemporalSegmenter.set_appearance) and the offline replay used
  for every reported number (evaluate_extended.segment_appearance) must agree
  exactly, the same drift guard that exists for the rule-based scorer;
* main.py's command-line defaults must equal the evaluated global
  configuration recorded in extended_results.json, so `python main.py --video
  x.mp4` always runs the system that was actually evaluated.

Needs only numpy, scipy and opencv.
"""
import json
import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import appearance as ap  # noqa: E402
import evaluate_extended as ee  # noqa: E402
from feature_extractor import FrameFeatures  # noqa: E402
from temporal_segmenter import TemporalSegmenter  # noqa: E402


def synthetic_video(states=(0, 1, 2, 1), seconds_each=6.0, fps=10.0, seed=0):
    """Descriptors for a video that holds a few distinct 'scene states'."""
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(max(states) + 1, 64))
    desc, feats = [], []
    n_each = int(seconds_each * fps)
    for k, st in enumerate(states):
        for i in range(n_each):
            d = protos[st] + 0.15 * rng.normal(size=64)
            desc.append(d / np.linalg.norm(d))
    for i in range(len(desc)):
        feats.append(FrameFeatures(frame_idx=i, timestamp=i / fps))
    return np.array(desc), feats, fps, n_each


class TestDescriptor(unittest.TestCase):
    def test_unit_norm_and_shape_stable(self):
        rng = np.random.default_rng(0)
        land = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
        port = rng.integers(0, 255, (960, 540, 3), dtype=np.uint8)
        a, b = ap.describe(land), ap.describe(port)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=4)
        self.assertEqual(a.shape, b.shape)  # orientation-independent length

    def test_identical_frames_identical_descriptors(self):
        f = np.full((100, 200, 3), 120, np.uint8)
        np.testing.assert_allclose(ap.describe(f), ap.describe(f.copy()))


class TestNovelty(unittest.TestCase):
    def test_peaks_at_state_changes(self):
        desc, _, fps, n = synthetic_video()
        nov = ap.novelty(desc, fps, window_s=2.0)
        # the strongest frames should sit at the three state changes
        top = sorted(np.argsort(nov)[-3:])
        for t, change in zip(top, (n, 2 * n, 3 * n)):
            self.assertLessEqual(abs(int(t) - change), 1)

    def test_constant_video_has_no_novelty(self):
        desc = np.tile(np.eye(8)[0], (50, 1))
        self.assertLess(float(ap.novelty(desc, 10.0).max()), 1e-9)


class TestLiveMatchesReplay(unittest.TestCase):
    def test_segmenter_equals_offline_mirror(self):
        desc, feats, fps, n = synthetic_video()
        seg = TemporalSegmenter(boundary_threshold=0.75, min_segment_duration=1.5, fps=fps)
        seg.set_appearance(desc)
        _, bounds = seg.segment(feats)
        live = [round(b.timestamp, 3) for b in bounds]
        F = [{"frame_idx": f.frame_idx, "timestamp": f.timestamp} for f in feats]
        replay = [round(t, 3) for t in ee.segment_appearance(F, desc, fps, 0.75, 1.5)]
        self.assertEqual(live, replay)
        # and it finds the three state changes
        self.assertEqual(len(live), 3)
        for t, change in zip(live, (n / fps, 2 * n / fps, 3 * n / fps)):
            self.assertLessEqual(abs(t - change), 0.3)

    def test_wrong_descriptor_count_is_an_error(self):
        desc, feats, fps, _ = synthetic_video()
        seg = TemporalSegmenter(fps=fps)
        seg.set_appearance(desc[:-5])
        with self.assertRaises(ValueError):
            seg.segment(feats)


class TestDefaultsMatchEvaluation(unittest.TestCase):
    """CLI defaults must be the configuration the reported numbers came from."""

    def setUp(self):
        p = os.path.join(ROOT, "extended_results.json")
        if not os.path.exists(p):
            self.skipTest("extended_results.json not present")
        with open(p, encoding="utf-8") as fh:
            self.res = json.load(fh)
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
            self.main_src = fh.read()

    def _defaults(self):
        m = re.search(r'scorer_defaults\s*=\s*(\{.*?\})', self.main_src, re.S)
        return eval(m.group(1), {})  # literal dict of tuples

    def test_rule_defaults(self):
        g = self.res["global"]
        self.assertEqual(self._defaults()["rule"], (g["threshold"], g["min_dur"]))

    def test_appearance_defaults(self):
        a = self.res.get("appearance_scorer", {}).get("global")
        if not a:
            self.skipTest("appearance scorer not evaluated (run appearance.py)")
        self.assertEqual(self._defaults()["appearance"], (a["threshold"], a["min_dur"]))


if __name__ == "__main__":
    unittest.main()
