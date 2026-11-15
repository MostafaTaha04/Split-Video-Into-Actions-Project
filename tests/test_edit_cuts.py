"""Tests for the hard-cut detector used to diagnose edited footage."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edit_cuts import find_cuts  # noqa: E402


def _signal(spikes, n=900, fps=30.0, base=3.0, seed=0):
    rng = np.random.default_rng(seed)
    d = base + rng.uniform(0, 1.5, n)
    for i, v in spikes:
        d[i] = v
    return np.arange(1, n + 1) / fps, d


class TestFindCuts(unittest.TestCase):
    def test_isolated_spikes_are_cuts(self):
        t, d = _signal([(150, 60.0), (600, 45.0)])
        self.assertEqual(find_cuts(t, d, 30.0), [round(t[150], 2), round(t[600], 2)])

    def test_small_or_sustained_change_is_not_a_cut(self):
        t, d = _signal([(150, 20.0)])           # below the absolute floor
        d[400:460] = 50.0                       # fast camera motion: high but not local
        self.assertEqual(find_cuts(t, d, 30.0), [])

    def test_close_cuts_merge_and_edges_ignored(self):
        t, d = _signal([(5, 80.0), (300, 70.0), (320, 70.0), (897, 80.0)])
        self.assertEqual(find_cuts(t, d, 30.0), [round(t[300], 2)])


if __name__ == "__main__":
    unittest.main()
