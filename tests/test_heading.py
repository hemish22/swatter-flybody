"""Heading geometry and the asymmetry readout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from heading import asymmetry, plane_position  # noqa: E402


class TestGeometry(unittest.TestCase):
    def test_a_stimulus_on_the_right_sits_on_the_right_eye_axis_and_far_from_the_left(self):
        self.assertAlmostEqual(plane_position(90.0, "R")[0, 0], 0.0, places=5)
        self.assertAlmostEqual(abs(plane_position(90.0, "L")[0, 0]), 180.0, places=5)

    def test_left_eye_sees_phi_as_the_right_eye_sees_minus_phi(self):
        for phi in (0.0, 30.0, 135.0, 200.0):
            np.testing.assert_allclose(plane_position(phi, "L"), plane_position(-phi, "R"), atol=1e-9)

    def test_ahead_and_behind_are_equidistant_from_both_axes(self):
        self.assertAlmostEqual(abs(plane_position(0.0, "R")[0, 0]), 90.0)
        self.assertAlmostEqual(abs(plane_position(180.0, "L")[0, 0]), 90.0)


class TestAsymmetry(unittest.TestCase):
    side = np.array(["R", "L", "R", "L"])
    targets = np.arange(4)

    def spikes(self, events):
        s = np.zeros((200, 4), dtype=bool)
        for t, j in events:
            s[t, j] = True
        return s

    def test_right_dominant(self):
        out = asymmetry(self.spikes([(10, 0), (12, 2), (14, 1)]), self.targets, self.side, 0.2)
        self.assertEqual((out["right"], out["left"], out["first_side"]), (2, 1, "R"))
        self.assertAlmostEqual(out["A"], 1 / 3)

    def test_spikes_after_the_window_are_ignored(self):
        out = asymmetry(self.spikes([(10, 0), (10 + 150, 1)]), self.targets, self.side, 0.2)  # 30 ms later
        self.assertEqual((out["right"], out["left"]), (1, 0))

    def test_silent_trial(self):
        out = asymmetry(self.spikes([]), self.targets, self.side, 0.2)
        self.assertEqual((out["fired"], out["A"]), (False, 0.0))


if __name__ == "__main__":
    unittest.main()
