"""Tests for the extracted escape circuit, read from the manifest.

These read the small JSON manifest rather than the 1.05 GB feather tables, so
they cost nothing to run and can live in the local `make test`. They exist to
pin the claims in docs/week0_status.md: if a future re-release of MaleCNS moves
the numbers, these fail loudly instead of the findings quietly becoming wrong.

Regenerate the manifest on the DGX with `make extract` before trusting it.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))

from common import local_allowed, on_remote, remote_dir, dgx_host  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "subgraph" / "manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise unittest.SkipTest(
            f"{MANIFEST_PATH} not present; run `make extract` on the DGX to produce it"
        )
    return json.loads(MANIFEST_PATH.read_text())


@unittest.skipUnless(MANIFEST_PATH.exists(), "manifest not extracted yet")
class TestSubgraphScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()

    def test_annotated_neuron_count(self):
        """The 166,700 figure is the superclass-filtered body count."""
        self.assertEqual(self.manifest["annotated_neurons"], 166_700)

    def test_edge_and_synapse_counts_are_plausible(self):
        edges = self.manifest["edges"]
        synapses = self.manifest["synapses"]
        self.assertGreater(edges, 20_000_000)
        self.assertGreater(synapses, edges, "a synapse count below the edge count is impossible")

    def test_populations_are_present_and_bilateral(self):
        by_type = {row["type"]: row for row in self.manifest["population_summary"]}
        for name in ("LPLC2", "LC4", "DNp01"):
            self.assertIn(name, by_type)
            self.assertGreater(by_type[name]["neurons"], 0)

        detectors = by_type["LPLC2"]["soma_side"]
        self.assertEqual(set(detectors), {"L", "R"}, "LPLC2 must exist on both sides")
        self.assertEqual(by_type["LC4"]["soma_side"].keys(), {"L", "R"})

    def test_giant_fiber_is_a_pair(self):
        """The GF is two neurons. If this is not 2, the body annotation drifted."""
        by_type = {row["type"]: row for row in self.manifest["population_summary"]}
        self.assertEqual(by_type["DNp01"]["neurons"], 2)
        self.assertEqual(len(self.manifest["giant_fiber_bodies"]), 2)


@unittest.skipUnless(MANIFEST_PATH.exists(), "manifest not extracted yet")
class TestEscapeCircuit(unittest.TestCase):
    """The plan's central premise, asserted rather than assumed.

    "a looming object activates LPLC2 neurons ... and LC4 neurons ... Both synapse
    directly onto the giant fiber (GF) descending neuron (Ache et al. 2019)."
    """

    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.core = {row["target"]: row for row in cls.manifest["escape_circuit_core"]}

    def test_lplc2_and_lc4_both_synapse_onto_the_gf(self):
        gf = self.core["DNp01"]
        self.assertGreater(gf["from_LPLC2"], 0, "no LPLC2 -> GF synapses: the premise fails")
        self.assertGreater(gf["from_LC4"], 0, "no LC4 -> GF synapses: the premise fails")

    def test_detectors_are_a_major_input_to_the_gf(self):
        gf = self.core["DNp01"]
        self.assertGreater(
            gf["detector_share_of_incoming"],
            0.05,
            "LPLC2+LC4 should be a substantial share of the GF's chemical input",
        )

    def test_literature_parallel_dns_are_confirmed_in_the_data(self):
        """DNp02, DNp04, DNp11 must be confirmed by the data, not assumed."""
        for target in ("DNp02", "DNp04", "DNp11"):
            self.assertIn(target, self.core)
            row = self.core[target]
            self.assertGreater(
                row["from_LPLC2"] + row["from_LC4"],
                0,
                f"{target} is named in the literature but gets no detector input here",
            )

    def test_detector_share_is_lower_for_the_gf_than_for_dnp04(self):
        """DNp04 is the most detector-dominated DN, so it leads the parallel set.

        This is the data making the choice the plan asked for, rather than the
        plan asserting it: if a re-release makes some other DN more
        detector-dominated than the GF itself, revisit the parallel set.
        """
        shares = {t: r["detector_share_of_incoming"] for t, r in self.core.items()}
        self.assertGreater(shares["DNp04"], shares["DNp01"])
        self.assertLess(shares["DNp01"], 1.0)


class TestExecutionGuard(unittest.TestCase):
    """The local-vs-DGX rule, tested without needing a DGX."""

    def test_defaults(self):
        self.assertEqual(dgx_host(), "h4hgpu")
        self.assertEqual(remote_dir(), "~/swatter-flybody")

    def test_escape_hatches_are_off_by_default(self):
        self.assertFalse(on_remote())
        self.assertFalse(local_allowed())

    def test_guard_blocks_a_local_heavy_run(self):
        from common import require_remote_execution

        import io
        import contextlib

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            require_remote_execution("some_job.py", "an expensive thing")
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("refusing to run this on your laptop", stderr.getvalue())
        self.assertIn("dgx.sh", stderr.getvalue())

    def test_guard_allows_a_remote_run(self):
        import os
        from common import require_remote_execution

        os.environ["SWATTER_REMOTE"] = "1"
        try:
            require_remote_execution("some_job.py", "an expensive thing")
        finally:
            del os.environ["SWATTER_REMOTE"]


if __name__ == "__main__":
    unittest.main()
