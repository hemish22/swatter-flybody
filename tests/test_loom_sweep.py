"""Tests for the sweep harness's pure parts: readout bookkeeping and the angular fallback drive."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))

from lif import EscapeGraph, LifParams  # noqa: E402
from loom import Loom  # noqa: E402
from loom_sweep import angular_drive, summarise, trial_readout  # noqa: E402

LPLC2 = ROOT / "data" / "ol" / "lplc2_inputs_R.npz"
P = LifParams()


def toy_graph() -> EscapeGraph:
    return EscapeGraph(
        bodies=np.arange(4), role=np.array(["source", "source", "target", "target"]),
        type=np.array(["LPLC2", "LPLC2", "DNp01", "DNp04"]), side=np.array(["R"] * 4),
        sign=np.ones(4, dtype=np.float32), pre=np.array([0]), post=np.array([2]), weight=np.array([1.0], dtype=np.float32))


class TestReadout(unittest.TestCase):
    def test_gf_time_angle_and_time_to_collision(self):
        loom = Loom(kind="expanding", rv_ms=20.0, azimuth_deg=0.0, dt_ms=5.0)
        spikes = np.zeros((1, int(400 / P.dt) + 1, 4), dtype=bool)
        spikes[0, int(300 / P.dt), 2] = True  # the GF fires at 300 ms
        spikes[0, int(310 / P.dt), 3] = True  # a parallel DN a little later
        row = trial_readout(spikes, toy_graph(), P, [loom])[0]
        self.assertAlmostEqual(row["gf_t_ms"], 300.0)
        self.assertAlmostEqual(row["gf_theta_deg"], float(np.interp(300.0, loom.time_ms(), loom.angular_size_deg())))
        self.assertAlmostEqual(row["gf_ttc_ms"], loom.t_c_ms - 300.0)  # time before collision, not trial clock
        self.assertEqual((row["gf_spikes"], row["parallel_spikes"]), (1, 1))

    def test_silent_trial_has_no_spike_fields(self):
        loom = Loom(kind="receding", rv_ms=20.0, azimuth_deg=0.0, dt_ms=5.0)
        row = trial_readout(np.zeros((1, 2001, 4), dtype=bool), toy_graph(), P, [loom])[0]
        self.assertIsNone(row["gf_t_ms"])
        self.assertEqual(row["gf_spikes"], 0)

    def test_criterion1_fractions_are_per_kind(self):
        rows = [dict(kind="expanding", rv_ms=20.0, gf_spikes=1, gf_t_ms=1.0, gf_theta_deg=10.0, gf_ttc_ms=5.0),
                dict(kind="expanding", rv_ms=20.0, gf_spikes=0, gf_t_ms=None, gf_theta_deg=None, gf_ttc_ms=None),
                dict(kind="receding", rv_ms=20.0, gf_spikes=0, gf_t_ms=None, gf_theta_deg=None, gf_ttc_ms=None),
                dict(kind="translating", rv_ms=20.0, gf_spikes=1, gf_t_ms=1.0, gf_theta_deg=10.0, gf_ttc_ms=5.0),
                dict(kind="dimming", rv_ms=20.0, gf_spikes=0, gf_t_ms=None, gf_theta_deg=None, gf_ttc_ms=None)]
        c = summarise(rows)["criterion1_gf_fraction_by_kind"]
        self.assertEqual(c, {"expanding": 0.5, "receding": 0.0, "translating": 1.0, "dimming": 0.0})


@unittest.skipUnless(LPLC2.exists(), "needs data/ol/lplc2_inputs_R.npz (make malecns-ol on the DGX)")
class TestAngularDrive(unittest.TestCase):
    def drive(self, kind, az=0.0, rv=20.0):
        return angular_drive([Loom(kind=kind, rv_ms=rv, azimuth_deg=az, dt_ms=5.0)], LPLC2)[0]

    def test_only_growth_drives(self):
        """Constant-size and shrinking stimuli must give exactly zero drive: that is the whole selectivity."""
        for kind in ("receding", "translating", "dimming"):
            self.assertEqual(float(self.drive(kind).max()), 0.0, kind)
        self.assertGreater(float(self.drive("expanding").max()), 0.0)

    def test_growth_rate_at_a_given_size_is_inverse_in_rv(self):
        """theta' = 2 sin^2(theta/2) / (r/v) at any size: why the GF spikes at a smaller angle for faster looms.

        Checked at a size inside each loom's own range (the r/v = 10 loom ends at
        12 degrees and the r/v = 80 one starts at 16, so no size is common to both
        in a 470 ms trial); the formula then gives 'faster is steeper at equal
        size' for free.
        """
        for rv, theta_at in ((10.0, 10.0), (40.0, 20.0), (80.0, 30.0)):
            loom = Loom(kind="expanding", rv_ms=rv, azimuth_deg=0.0, dt_ms=0.5, duration_ms=470.0)
            theta = loom.angular_size_deg()
            self.assertTrue(theta.min() < theta_at < theta.max(), (rv, theta.min(), theta.max()))
            rate = float(np.interp(theta_at, theta, np.gradient(theta, loom.dt_ms)))
            want = 2 * np.sin(np.radians(theta_at / 2)) ** 2 / rv * 180 / np.pi  # deg/ms
            self.assertAlmostEqual(rate, want, delta=0.02 * want)

    def test_only_neurons_near_the_disc_are_driven(self):
        d = self.drive("expanding").max(axis=0)
        self.assertGreater(float(d.max()), 100 * float(np.median(d)))


if __name__ == "__main__":
    unittest.main()
