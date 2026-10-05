"""Tests for the eye lattice and the loom renderer.

Cheap and local: no flyvis weights, no GPU. Each test that pins a specific
failure says which one.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from eye import (  # noqa: E402
    EXTENT,
    N_COLUMNS,
    OMMATIDIAL_ANGLE_DEG,
    Eye,
    column_plane_deg,
    edge_movie,
    flyvis_columns,
    render_loom,
    unit_vector,
)
from loom import Loom  # noqa: E402


class TestLattice(unittest.TestCase):
    def setUp(self):
        self.eye = Eye.flyvis()

    def test_721_columns_in_flyvis_order(self):
        """Column i must be photoreceptor i of the network: u ascending, then v."""
        uv = flyvis_columns()
        self.assertEqual(len(uv), N_COLUMNS)
        self.assertEqual(N_COLUMNS, 721)
        self.assertEqual(tuple(uv[0]), (-EXTENT, 0))
        self.assertEqual(tuple(uv[-1]), (EXTENT, 0))
        order = np.lexsort((uv[:, 1], uv[:, 0]))
        self.assertTrue((order == np.arange(len(uv))).all())

    def test_neighbours_are_one_inter_ommatidial_angle_apart(self):
        plane = column_plane_deg(self.eye.uv)
        d = np.linalg.norm(plane[:, None] - plane[None], axis=2)
        np.fill_diagonal(d, np.inf)
        self.assertAlmostEqual(float(d.min()), OMMATIDIAL_ANGLE_DEG, places=6)
        # a column in the middle has exactly six neighbours at that distance
        centre = int(np.argmin(np.linalg.norm(plane, axis=1)))
        self.assertEqual(int((np.abs(d[centre] - OMMATIDIAL_ANGLE_DEG) < 1e-6).sum()), 6)

    def test_field_is_87_degrees_off_axis(self):
        ecc = np.degrees(np.arccos(self.eye.directions[:, 2]))
        self.assertAlmostEqual(float(ecc.max()), EXTENT * OMMATIDIAL_ANGLE_DEG, places=6)

    def test_plane_matches_flyvis_hex_to_pixel_orientation(self):
        """The T4/T5 preferred directions flyvis quotes are angles in ITS plane.

        If this embedding were turned or mirrored the radial wiring would point
        the wrong way and nothing downstream would crash.
        """
        try:
            from flyvis.utils.hex_utils import hex_to_pixel
        except Exception:  # flyvis not installed on this machine
            self.skipTest("flyvis not installed")
        uv = self.eye.uv
        x, y = hex_to_pixel(uv[:, 0], uv[:, 1])
        ref = np.stack([x, y], axis=1) * (OMMATIDIAL_ANGLE_DEG / math.sqrt(3.0))
        np.testing.assert_allclose(column_plane_deg(uv), ref, atol=1e-9)


class TestRenderer(unittest.TestCase):
    def setUp(self):
        self.eye = Eye.flyvis()

    def dark(self, kind, rv=20.0, az=30.0):
        x = render_loom(Loom(kind=kind, rv_ms=rv, azimuth_deg=az, dt_ms=5.0), self.eye)
        return (1.0 - x).sum(axis=1)  # total darkness per frame

    def test_expanding_gets_darker_receding_does_not(self):
        e, r = self.dark("expanding"), self.dark("receding")
        self.assertGreater(e[-1], 4 * e[0])
        self.assertLess(r[-1], r[0])

    def test_controls_are_flat_in_size(self):
        """translating and dimming must not grow; they differ from the loom only in lacking growth."""
        t, d = self.dark("translating"), self.dark("dimming")
        self.assertLess(abs(t[-1] - t[0]) / t[0], 0.1)
        self.assertGreater(d[0], 5 * d[-1])  # contrast fades to nothing

    def test_small_disc_is_dim_not_missing(self):
        """A 4.8 degree disc is under one acceptance angle: dimmer, but there.

        Catches the sampler turning into a hard 'covered or not' test, which
        would drop the start of the r/v=20 ms loom entirely.
        """
        x = render_loom(Loom(kind="expanding", rv_ms=20.0, azimuth_deg=0.0, dt_ms=5.0), self.eye)
        self.assertTrue(0.3 < x[0].min() < 0.98)
        self.assertLess(x[-1].min(), 0.05)

    def test_disc_is_centred_where_asked(self):
        x = render_loom(Loom(kind="expanding", rv_ms=20.0, azimuth_deg=90.0, dt_ms=5.0), self.eye)
        darkest = int(np.argmin(x[-1]))
        want = unit_vector(40.0, 90.0)[0]
        got = self.eye.directions[darkest]
        self.assertLess(math.degrees(math.acos(np.clip(want @ got, -1, 1))), OMMATIDIAL_ANGLE_DEG)

    def test_luminance_stays_in_unit_range(self):
        for kind in ("expanding", "receding", "translating", "dimming"):
            x = render_loom(Loom(kind=kind, rv_ms=10.0, azimuth_deg=0.0, dt_ms=5.0), self.eye)
            self.assertGreaterEqual(float(x.min()), -1e-6)
            self.assertLessEqual(float(x.max()), 1.0 + 1e-6)


class TestEdgeMovie(unittest.TestCase):
    def test_edge_moves_along_the_requested_direction(self):
        eye = Eye.flyvis()
        pos = column_plane_deg(eye.uv)
        for phi in (0, 90, 180, 270):
            m = edge_movie(eye, phi, 150.0, 0.005, 1.2)
            n = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
            along = pos @ n
            first = along[m[40] > 0.75].mean()
            later = along[m[160] > 0.75].mean()
            self.assertGreater(later, first, f"bar did not advance along {phi} deg")


if __name__ == "__main__":
    unittest.main()
