"""Tests for the Week 2 fit harness: the split is really held out, and selection avoids cliffs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))

from lif_fit import (  # noqa: E402
    GAIN_SCALES, TEST_RV_EXTRAP, TEST_RV_INTERP, THRESHOLDS, WEIGHT_SCALES, build_looms, score, select,
)
from loom import RV_MS  # noqa: E402
from loom_sweep import build_trials  # noqa: E402
from takeoff import classify  # noqa: E402

SHAPE = (len(GAIN_SCALES), len(WEIGHT_SCALES), len(THRESHOLDS))


def fake_rows(margins: np.ndarray) -> list[dict]:
    return [{"train": {"margin": float(m), "theta_spread": 2.0}} for m in margins.ravel()]


class TestSplit(unittest.TestCase):
    def test_test_set_shares_no_rv_and_no_azimuth_with_train(self):
        self.assertFalse(set(RV_MS) & set(TEST_RV_INTERP + TEST_RV_EXTRAP))
        train_az = {round(l.azimuth_deg % 360, 3) for l in build_trials()}
        test_az = {round(l.azimuth_deg % 360, 3) for l in build_looms(TEST_RV_INTERP, 22.5)}
        self.assertFalse(train_az & test_az)

    def test_train_builder_matches_the_gate_suite(self):
        a = [(l.kind, l.rv_ms, l.azimuth_deg) for l in build_looms(RV_MS, 0.0)]
        b = [(l.kind, l.rv_ms, l.azimuth_deg) for l in build_trials()]
        self.assertEqual(a, b)


class TestSelect(unittest.TestCase):
    def test_prefers_a_plateau_over_a_lone_perfect_cliff(self):
        m = np.zeros(SHAPE)
        m[1:4, 1:4, 1:4] = 1.0  # a 3x3x3 plateau of perfect settings, interior
        m[6, 4, 3] = 1.0  # a lone perfect cell whose neighbours are all zero
        pick = select(fake_rows(m), SHAPE)
        g, w, v = np.unravel_index(pick, SHAPE)
        self.assertTrue(1 <= g <= 3 and 1 <= w <= 3 and 1 <= v <= 3, (g, w, v))

    def test_grid_edge_is_never_chosen(self):
        m = np.ones(SHAPE)
        pick = select(fake_rows(m), SHAPE)
        g, w, v = np.unravel_index(pick, SHAPE)
        self.assertTrue(0 < g < SHAPE[0] - 1 and 0 < w < SHAPE[1] - 1 and 0 < v < SHAPE[2] - 1)


class TestScore(unittest.TestCase):
    def row(self, kind, rv, spikes, t=None, theta=None, par=None):
        return {"kind": kind, "rv_ms": rv, "gf_spikes": spikes, "gf_t_ms": t, "gf_theta_deg": theta,
                "parallel_t_ms": par, "mode": classify(t, par)}

    def test_margin_is_expanding_minus_strongest_control(self):
        rows = [self.row("expanding", 20.0, 1, 100.0, 10.0), self.row("expanding", 20.0, 1, 110.0, 12.0),
                self.row("receding", 20.0, 0), self.row("translating", 20.0, 1), self.row("dimming", 20.0, 0)]
        # translating fires in 1 of 1 trials, so the margin is 1 - 1 = 0
        s = score(rows, (20.0,))
        self.assertEqual(s["margin"], 0.0)
        self.assertEqual(s["theta_by_rv"]["20.0"], 11.0)

    def test_only_the_asked_rvs_are_scored(self):
        rows = [self.row("expanding", 20.0, 1, 1.0, 1.0), self.row("expanding", 40.0, 0),
                self.row("receding", 20.0, 0), self.row("translating", 20.0, 0), self.row("dimming", 20.0, 0)]
        self.assertEqual(score(rows, (20.0,))["margin"], 1.0)

    def test_short_fraction_and_lag_per_rv(self):
        rows = [self.row("expanding", 20.0, 1, 100.0, 10.0, par=98.0),   # lag -2, short
                self.row("expanding", 20.0, 1, 120.0, 10.0, par=100.0),  # GF 20 ms after the raise began: long
                self.row("expanding", 20.0, 0)]                          # no escape: not counted
        s = score(rows, (20.0,))
        self.assertEqual(s["short_fraction_by_rv"]["20.0"], 0.5)
        self.assertEqual(s["parallel_minus_gf_ms_by_rv"]["20.0"], -11.0)


if __name__ == "__main__":
    unittest.main()
