"""Tests for the MaleCNS optic-lobe helpers: the pure functions, on toy data.

The real run reads a 1 GB table on the DGX; these pin the logic that run depends
on, with tables small enough to check by hand.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from malecns_ol import (  # noqa: E402
    apply_map,
    candidate_matrices,
    centroid,
    cosine,
    infer_columns,
    to_flyvis_columns,
    weighted_median,
)


class TestColumnInference(unittest.TestCase):
    def test_weighted_median_follows_the_weight(self):
        self.assertEqual(weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 10.0])), 3.0)
        self.assertEqual(weighted_median(np.array([5.0, 1.0, 3.0]), np.array([1.0, 1.0, 1.0])), 3.0)

    def toy(self):
        n = pd.DataFrame(
            {
                "bodyId": [1, 2, 3, 10],
                "type": ["Mi1", "Mi1", "Tm1", "T4a"],
                "assignedOlHex1": [10.0, 10.0, 12.0, np.nan],
                "assignedOlHex2": [20.0, 20.0, 22.0, np.nan],
            }
        )
        e = pd.DataFrame({"body_pre": [1, 2, 3], "body_post": [10, 10, 10], "weight": [8, 6, 1]})
        return n, e

    def test_untagged_neuron_takes_the_median_of_tagged_partners(self):
        n, e = self.toy()
        inf = infer_columns(n, e).set_index("bodyId")
        self.assertEqual((inf.loc[10, "inf_hex1"], inf.loc[10, "inf_hex2"]), (10.0, 20.0))

    def test_a_neuron_is_never_its_own_partner(self):
        """Tagged neurons are scored by re-inferring them, so self-edges would leak the answer."""
        n, e = self.toy()
        e = pd.concat([e, pd.DataFrame({"body_pre": [1], "body_post": [1], "weight": [1000]})])
        inf = infer_columns(n, e).set_index("bodyId")
        self.assertNotIn(1, inf.index)  # its only other contact is the untagged T4a


class TestOrientation(unittest.TestCase):
    def test_104_candidates_all_unimodular(self):
        c = candidate_matrices()
        self.assertEqual(len(c), 104)
        self.assertTrue(all(abs(a * d - b * cc) == 1 for a, b, cc, d in c))

    def test_identity_is_a_candidate_and_preserves_filters(self):
        self.assertIn((1, 0, 0, 1), candidate_matrices())
        f = {("A", "B", 1, 0): 2.0, ("A", "B", 0, -1): 1.0}
        self.assertEqual(dict(apply_map((1, 0, 0, 1), f)), f)

    def test_rotation_is_found_by_cosine(self):
        """A filter rotated by a lattice symmetry is recovered by the matching inverse map."""
        f = {("A", "B", 1, 0): 2.0, ("A", "B", 0, 1): 1.0, ("A", "B", 2, 1): 0.5}
        m = (0, -1, 1, 1)  # a 60 degree rotation in axial coordinates
        rotated = dict(apply_map(m, f))
        best = max(candidate_matrices(), key=lambda c: cosine(f, apply_map(c, rotated)))
        self.assertAlmostEqual(cosine(f, apply_map(best, rotated)), 1.0)

    def test_centroid_is_weight_averaged(self):
        f = {("A", "B", 2, 0): 3.0, ("A", "B", 0, 0): 1.0}
        self.assertEqual(centroid(f, "A", "B"), (1.5, 0.0))
        self.assertIsNone(centroid(f, "A", "C"))


class TestLatticeCrop(unittest.TestCase):
    def test_origin_is_the_anchor_centroid_and_far_columns_are_cropped(self):
        n = pd.DataFrame(
            {
                "type": ["Mi1", "Mi1", "Mi1", "T4a"],
                "hex1": [10.0, 12.0, 11.0, 11.0 + 40.0],
                "hex2": [20.0, 20.0, 20.0, 20.0],
            }
        )
        out, info = to_flyvis_columns(n, (1, 0, 0, 1))
        self.assertEqual(info["origin_hex"], [11.0, 20.0])
        self.assertEqual((out["u"].iloc[2], out["v"].iloc[2]), (0.0, 0.0))
        self.assertEqual(list(out["in_lattice"]), [True, True, True, False])


if __name__ == "__main__":
    unittest.main()
