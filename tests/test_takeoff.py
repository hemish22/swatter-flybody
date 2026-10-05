"""The plan's mode rule, pinned on its boundary cases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from takeoff import SHORT_WINDOW_MS, classify  # noqa: E402


class TestClassify(unittest.TestCase):
    def test_nothing_fires(self):
        self.assertEqual(classify(None, None), "none")

    def test_gf_first_is_short(self):
        self.assertEqual(classify(10.0, 12.0), "short")
        self.assertEqual(classify(10.0, None), "short")

    def test_parallel_only_is_long(self):
        self.assertEqual(classify(None, 10.0), "long")

    def test_gf_during_the_raise_is_short_and_after_it_long(self):
        self.assertEqual(classify(10.0 + SHORT_WINDOW_MS, 10.0), "short")  # exactly at the bound
        self.assertEqual(classify(10.0 + SHORT_WINDOW_MS + 0.2, 10.0), "long")

    def test_window_is_the_published_bound(self):
        self.assertEqual(SHORT_WINDOW_MS, 6.87)


if __name__ == "__main__":
    unittest.main()
