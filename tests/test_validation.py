"""The degree-preserving shuffle used by ablation 3."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from validation import shuffle_edges  # noqa: E402


class TestShuffle(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 60
        pairs = set()
        while len(pairs) < 400:
            a, b = rng.integers(0, n, 2)
            if a != b:
                pairs.add((int(a), int(b)))
        pairs = sorted(pairs)
        self.pre = np.array([p[0] for p in pairs])
        self.post = np.array([p[1] for p in pairs])

    def test_degrees_are_preserved_and_edges_actually_move(self):
        pre, post, done = shuffle_edges(self.pre, self.post, np.random.default_rng(1))
        self.assertEqual(Counter(pre.tolist()), Counter(self.pre.tolist()))
        self.assertEqual(Counter(post.tolist()), Counter(self.post.tolist()))
        self.assertGreater(done, 0)
        self.assertLess(len(set(zip(pre.tolist(), post.tolist())) & set(zip(self.pre.tolist(), self.post.tolist()))), 0.5 * len(pre))

    def test_no_self_loops_or_duplicates(self):
        pre, post, _ = shuffle_edges(self.pre, self.post, np.random.default_rng(2))
        self.assertFalse((pre == post).any())
        self.assertEqual(len(set(zip(pre.tolist(), post.tolist()))), len(pre))

    def test_deterministic_for_a_seed(self):
        a = shuffle_edges(self.pre, self.post, np.random.default_rng(3))
        b = shuffle_edges(self.pre, self.post, np.random.default_rng(3))
        np.testing.assert_array_equal(a[1], b[1])

    def test_inputs_are_not_modified(self):
        before = self.post.copy()
        shuffle_edges(self.pre, self.post, np.random.default_rng(4))
        np.testing.assert_array_equal(self.post, before)


if __name__ == "__main__":
    unittest.main()
