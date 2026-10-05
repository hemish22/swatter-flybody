"""Tests for the lab stimulus generator.

These run in well under a second and need no connectome data, because the
stimulus generator is the one piece of the offline pipeline that is cheap to
test and expensive to get wrong: if the controls are broken, criteria 1 and 2
silently measure the wrong thing and the whole Week 1 gate is meaningless.

Each test that pins a specific bug is marked with what it caught.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from loom import (  # noqa: E402
    COLLISION_OVERSHOOT,
    RV_MS,
    STIMULUS_KINDS,
    EyeGeometry,
    Loom,
    build_suite,
)


class TestAngularSize(unittest.TestCase):
    def test_expanding_grows_monotonically(self):
        theta = Loom(kind="expanding", rv_ms=20).angular_size_deg()
        self.assertTrue(np.all(np.diff(theta) > 0), "expanding disc must grow")

    def test_expanding_stays_below_180(self):
        """Collision sits past the trial, so no sample is the degenerate 180 deg."""
        for rv in RV_MS:
            theta = Loom(kind="expanding", rv_ms=rv).angular_size_deg()
            self.assertLess(theta.max(), 180.0, f"r/v={rv} hits the degenerate limit")
            self.assertGreater(theta[0], 0.0)

    def test_receding_shrinks(self):
        """Caught a bug where abs() made receding identical to expanding.

        With the absolute value taken, the two stimuli had byte-identical theta
        and criterion 1 was being tested against a copy of the positive case.
        """
        theta = Loom(kind="receding", rv_ms=20).angular_size_deg()
        self.assertTrue(np.all(np.diff(theta) < 0), "receding disc must shrink")

    def test_receding_starts_where_expanding_starts(self):
        """The pair must share an initial angle so only the direction differs."""
        expanding = Loom(kind="expanding", rv_ms=40).angular_size_deg()
        receding = Loom(kind="receding", rv_ms=40).angular_size_deg()
        self.assertAlmostEqual(expanding[0], receding[0], places=6)

    def test_faster_looms_are_larger_at_every_time(self):
        """Criterion 2 needs faster looms to be ahead throughout the run-up."""
        theta = {rv: Loom(kind="expanding", rv_ms=rv).angular_size_deg() for rv in RV_MS}
        for earlier, later in zip(RV_MS, RV_MS[1:]):
            self.assertTrue(
                np.all(theta[later] > theta[earlier]),
                f"r/v={later} should loom faster than r/v={earlier}",
            )

    def test_faster_looms_have_higher_size_rate(self):
        rates = {}
        for rv in RV_MS:
            loom = Loom(kind="expanding", rv_ms=rv)
            theta = loom.angular_size_deg()
            rates[rv] = float(np.gradient(theta, loom.dt_ms).max())
        for earlier, later in zip(RV_MS, RV_MS[1:]):
            self.assertGreater(rates[later], rates[earlier])

    def test_controls_are_flat_and_visible(self):
        """A control must be big enough to see, or it passes by being invisible."""
        for kind in ("translating", "dimming"):
            for rv in RV_MS:
                loom = Loom(kind=kind, rv_ms=rv)
                theta = loom.angular_size_deg()
                self.assertAlmostEqual(float(theta.std()), 0.0, places=9, msg=f"{kind} not flat")
                self.assertGreater(theta[0], 1.0, f"{kind} r/v={rv} control is too small to detect")

    def test_collison_overshoot_keeps_samples_pre_collision(self):
        loom = Loom(kind="expanding", rv_ms=20, duration_ms=400)
        self.assertGreater(loom.t_c_ms, loom.duration_ms)
        self.assertAlmostEqual(loom.t_c_ms, 400 * COLLISION_OVERSHOOT)


class TestLuminance(unittest.TestCase):
    def setUp(self):
        self.eye = EyeGeometry.hexagonal_lattice()

    def test_only_covered_columns_go_dark(self):
        """Caught a units bug: the angular offset was radians, theta was degrees.

        Every offset looked smaller than the disc, so all 721 columns came back
        dark and the raster carried no spatial information at all.
        """
        loom = Loom(kind="expanding", rv_ms=20, azimuth_deg=0.0)
        dark = (loom.luminance(self.eye) < 0.5).sum(axis=1)
        expected = loom.angular_size_deg()[0] / self.eye.acceptance_deg
        self.assertAlmostEqual(dark[0], expected, delta=2.0)
        self.assertLess(dark[0], self.eye.n_columns * 0.5)

    def test_background_is_bright_away_from_the_stimulus(self):
        loom = Loom(kind="expanding", rv_ms=20, azimuth_deg=0.0)
        luminance = loom.luminance(self.eye)
        far = np.abs(self.eye.azimuth_deg) > 60.0
        self.assertTrue(np.all(luminance[:, far] > 0.9), "sky away from the disc must stay bright")

    def test_disc_centre_is_dark(self):
        loom = Loom(kind="expanding", rv_ms=20, azimuth_deg=0.0)
        centre = int(np.argmin(np.abs(self.eye.azimuth_deg)))
        self.assertLess(loom.luminance(self.eye)[-1, centre], 0.1)

    def test_dark_column_count_tracks_angular_size(self):
        loom = Loom(kind="expanding", rv_ms=20, azimuth_deg=0.0)
        theta = loom.angular_size_deg()
        dark = (loom.luminance(self.eye) < 0.5).sum(axis=1)
        for i in (0, len(theta) // 2, len(theta) - 1):
            self.assertAlmostEqual(dark[i], theta[i] / self.eye.acceptance_deg, delta=2.0)

    def test_expanding_darkens_and_receding_brightens(self):
        counts = {}
        for kind in ("expanding", "receding"):
            loom = Loom(kind=kind, rv_ms=20, azimuth_deg=0.0)
            counts[kind] = (loom.luminance(self.eye) < 0.5).sum(axis=1)
        self.assertGreater(counts["expanding"][-1], counts["expanding"][0])
        self.assertLess(counts["receding"][-1], counts["receding"][0])

    def test_dimming_fades_to_background(self):
        loom = Loom(kind="dimming", rv_ms=20, azimuth_deg=0.0)
        luminance = loom.luminance(self.eye)
        self.assertGreater(luminance[0].mean(), 0.5, "dimming control should start dark")
        self.assertAlmostEqual(float(luminance[-1].mean()), loom.background, places=6)

    def test_translating_keeps_constant_coverage(self):
        loom = Loom(kind="translating", rv_ms=20, azimuth_deg=0.0)
        dark = (loom.luminance(self.eye) < 0.5).sum(axis=1)
        self.assertLess(float(dark.std()), 3.0, "a constant-size disc must cover a constant area")

    def test_shifting_azimuth_shifts_the_dark_region(self):
        centre = int(np.argmin(np.abs(self.eye.azimuth_deg - 0.0)))
        off_centre = int(np.argmin(np.abs(self.eye.azimuth_deg - 60.0)))
        # r/v=10 reaches ~14 deg, so the disc is well clear of 60 deg. Using a
        # fast loom here would not test anything: a 90 deg-wide disc centred at 0
        # legitimately covers 60 deg too.
        dark_at_zero = Loom(kind="expanding", rv_ms=10, azimuth_deg=0.0).luminance(self.eye)[-1] < 0.5
        dark_at_sixty = Loom(kind="expanding", rv_ms=10, azimuth_deg=60.0).luminance(self.eye)[-1] < 0.5
        self.assertTrue(dark_at_zero[centre])
        self.assertFalse(dark_at_zero[off_centre])
        self.assertTrue(dark_at_sixty[off_centre])
        self.assertFalse(dark_at_sixty[centre])


class TestSuite(unittest.TestCase):
    def test_suite_shape(self):
        suite = build_suite()
        self.assertEqual(len(suite), len(RV_MS) * 8 * len(STIMULUS_KINDS))

    def test_suite_covers_every_combination(self):
        suite = build_suite()
        keys = {(s.kind, s.rv_ms, s.azimuth_deg) for s in suite}
        self.assertEqual(len(keys), len(suite), "duplicate trials in the suite")

    def test_azimuths_span_the_circle(self):
        """Criterion 4 needs a uniform stimulus field, not hand-picked angles."""
        azimuths = sorted({s.azimuth_deg for s in build_suite()})
        self.assertEqual(len(azimuths), 8)
        # Wrap the last gap through 360, not through zero.
        gaps = {round((b - a) % 360.0, 6) for a, b in zip(azimuths, azimuths[1:] + azimuths[:1])}
        self.assertEqual(gaps, {45.0}, "azimuths should be evenly spaced")

    def test_only_expanding_should_escape(self):
        for loom in build_suite():
            truth = loom.ground_truth()
            self.assertEqual(truth["should_escape"], loom.kind == "expanding")

    def test_controls_have_zero_size_rate(self):
        for loom in build_suite():
            if loom.kind == "expanding":
                continue
            if loom.kind == "receding":
                continue  # receding shrinks; that is its defining feature
            self.assertAlmostEqual(loom.ground_truth()["size_rate_deg_per_ms"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
