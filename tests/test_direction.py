"""
Tests for the assembly / disassembly pass (action_direction.py), the
direction-aware activity names (activity_recognizer.reword) and the label
rule used to evaluate it (evaluate_direction.direction_from_label).
"""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import action_direction as ad  # noqa: E402
from activity_recognizer import ActivityRecognizer, reword  # noqa: E402
from evaluate_direction import auc, direction_from_label, fit_logistic, is_packaging  # noqa: E402


def radial_flow(n, outward=True, speed=2.0, aspect=16 / 9):
    """n flow frames of pure radial motion about the frame centre."""
    yy, xx = np.mgrid[0:ad.GRID_H, 0:ad.GRID_W]
    dx = ((xx + 0.5) / ad.GRID_W - 0.5) * aspect
    dy = (yy + 0.5) / ad.GRID_H - 0.5
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    sign = 1.0 if outward else -1.0
    g = np.stack([sign * speed * dx / norm, sign * speed * dy / norm], axis=-1)
    return {"t": np.arange(n) / ad.FLOW_FPS, "grid": np.repeat(g[None], n, 0).astype(np.float16),
            "cut": np.zeros(n, bool), "aspect": np.float64(aspect)}


class TestWindowFeatures(unittest.TestCase):
    def test_outward_motion_has_positive_flux(self):
        x = ad.window_features(radial_flow(20, outward=True), 0.0, 4.0)
        self.assertGreater(x[0], 0.9)

    def test_inward_motion_has_negative_flux(self):
        x = ad.window_features(radial_flow(20, outward=False), 0.0, 4.0)
        self.assertLess(x[0], -0.9)

    def test_motion_late_in_window_has_late_centroid(self):
        f = radial_flow(20)
        f["grid"][:10] = 0
        x = ad.window_features(f, 0.0, 4.0)
        self.assertGreater(x[1], 0.7)

    def test_log_duration(self):
        x = ad.window_features(radial_flow(30), 1.0, 5.0)
        self.assertAlmostEqual(x[2], np.log(4.0))

    def test_too_short_window_gives_none(self):
        self.assertIsNone(ad.window_features(radial_flow(20), 0.0, 0.3))

    def test_still_window_gives_none(self):
        f = radial_flow(20)
        f["grid"][:] = 0
        self.assertIsNone(ad.window_features(f, 0.0, 4.0))


class TestDecision(unittest.TestCase):
    def test_bands_in_auto_mode(self):
        self.assertEqual(ad.decide(0.9)[0], ad.DISASSEMBLY)
        self.assertEqual(ad.decide(0.1)[0], ad.ASSEMBLY)
        self.assertEqual(ad.decide(0.5), (ad.UNCLEAR, "none"))
        self.assertEqual(ad.decide(None), (ad.UNCLEAR, "none"))

    def test_task_names_every_step(self):
        self.assertEqual(ad.decide(0.5, "disassembly"), (ad.DISASSEMBLY, "task setting"))
        self.assertEqual(ad.decide(None, "assembly"), (ad.ASSEMBLY, "task setting"))
        self.assertEqual(ad.decide(0.95, "assembly"), (ad.ASSEMBLY, "task setting"))

    def test_summary_verdicts(self):
        A, D, U = ad.ASSEMBLY, ad.DISASSEMBLY, ad.UNCLEAR
        self.assertEqual(ad.summarise([A, A, A, D, U])["verdict"], "mostly assembly")
        self.assertEqual(ad.summarise([D, D, D, U])["verdict"], "mostly disassembly")
        self.assertEqual(ad.summarise([A, D, A, D])["verdict"], "mixed or unclear")
        self.assertEqual(ad.summarise([A, A])["verdict"], "mixed or unclear")     # too few
        self.assertIn("--task", ad.summarise([A], "disassembly")["verdict"])

    def test_model_direction_matches_the_cues(self):
        m = ad.load_model()
        out = ad.p_remove([0.3, 0.6, np.log(2.0)], m)     # outward, late, short
        inn = ad.p_remove([-0.3, 0.4, np.log(8.0)], m)    # inward, early, long
        self.assertGreater(out, 0.5)
        self.assertLess(inn, 0.5)


class TestFlowPass(unittest.TestCase):
    def test_flow_grid_on_synthetic_video(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.avi")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
            rng = np.random.default_rng(0)
            tex = (rng.random((180, 320)) * 255).astype(np.uint8)
            for k in range(30):
                frame = np.roll(tex, 2 * k, axis=1)
                w.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            w.release()
            flow = ad.extract_flow_grid(path)
            self.assertEqual(flow["grid"].shape[1:], (ad.GRID_H, ad.GRID_W, 2))
            self.assertAlmostEqual(float(flow["aspect"]), 320 / 180, places=3)
            # whole-frame translation is camera motion: removed by the median
            self.assertLess(float(np.abs(flow["grid"].astype(np.float32)).mean()), 0.5)
            # cache round trip
            f2 = ad.flow_for_run(path, d)
            self.assertTrue(os.path.exists(os.path.join(d, "direction_flow.npz")))
            self.assertEqual(len(f2["t"]), len(flow["t"]))


class TestWording(unittest.TestCase):
    def test_reword(self):
        self.assertEqual(reword("Press RAM module into slot", "assembly"), "Press RAM module into slot")
        self.assertEqual(reword("Press RAM module into slot", "disassembly"), "Pull RAM module out of slot")
        self.assertEqual(reword("Press RAM module into slot", "unclear"), "RAM module handling")
        self.assertEqual(reword("Inspection or pause", "disassembly"), "Inspection or pause")

    def test_names_are_neutral_without_direction(self):
        class Seg:
            real_objects_used = ["ram stick"]
            interaction_types = ["grip"]
            avg_motion_energy = 4.0
        r = ActivityRecognizer()
        self.assertEqual(r.describe_segment(Seg(), 10, 0, 1).label, "RAM module handling")
        d = r.describe_segment(Seg(), 10, 0, 1, direction="disassembly")
        self.assertEqual(d.label, "Pull RAM module out of slot")
        self.assertEqual(d.key, "Press RAM module into slot")


class TestQuickMoveRule(unittest.TestCase):
    """A short, gripping, high-motion step is a part moved, not a screw turned."""

    class Seg:
        grip_ratio = 0.6
        avg_curvature = 0.8          # curved trajectory: used to give "Rotating"
        hands_present_ratio = 0.8
        avg_motion_energy = 4.0      # strong motion

    def key_for(self, duration, quick_move=ActivityRecognizer.QUICK_MOVE_SECONDS):
        s = self.Seg()
        s.duration = duration
        r = ActivityRecognizer()
        r.quick_move_seconds = quick_move
        return r.describe_segment(s, 10, 0, 3).key

    def test_short_step_is_a_move(self):
        self.assertEqual(self.key_for(2.5), "Grip and move a component")
        self.assertEqual(reword(self.key_for(2.5), "disassembly"), "Take a component out or away")

    def test_long_step_is_still_turning(self):
        self.assertEqual(self.key_for(20.0), "Rotating / tightening motion")

    def test_rule_can_be_disabled(self):
        self.assertEqual(self.key_for(2.5, quick_move=None), "Rotating / tightening motion")


class TestLabelRule(unittest.TestCase):
    def test_install_names(self):
        for lab in ("Insert the first RAM module", "Screw the motherboard in", "Place the CPU in the socket",
                    "Fetch and drive the next screw", "Slide the side cover on", "Connect CPU power cable"):
            self.assertEqual(direction_from_label(lab), ad.ASSEMBLY, lab)

    def test_remove_names(self):
        for lab in ("Remove the side panel", "Unscrew the CPU cooler", "Take the SSD out",
                    "Lift off the case panel", "Unclip and lift off the rear side panel",
                    "Unscrew the first bottom screw"):
            self.assertEqual(direction_from_label(lab), ad.DISASSEMBLY, lab)

    def test_other_names_are_left_out(self):
        for lab in ("Open the box", "Inspect CPU contact side", "Set the socket cover aside",
                    "Pick up the screwdriver", "Open the screw bag"):
            self.assertIsNone(direction_from_label(lab), lab)

    def test_packaging(self):
        self.assertTrue(is_packaging("Take out the RAM clamshell"))
        self.assertFalse(is_packaging("Take the RAM modules out"))


class TestStats(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertEqual(auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)

    def test_logistic_separates_separable_data(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] - X[:, 2] > 0).astype(float)
        m = fit_logistic(X, y)
        self.assertGreater(m["coef"][0], 0)
        self.assertLess(m["coef"][2], 0)
        p = np.array([ad.p_remove(x, m) for x in X])
        self.assertGreater(((p > 0.5) == (y > 0.5)).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
