"""Browser-engine parity: engine_ref (event-driven, streaming, from web/brain only) against the
Python reference outcomes recorded by export.py. Needs web/brain, which export.py builds on the DGX."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))

from engine_ref import Brain, outcome, run_trial  # noqa: E402

BRAIN = ROOT / "web" / "brain"
HAVE = (BRAIN / "parity.json").exists()


@unittest.skipUnless(HAVE, "web/brain not built (offline/export.py, DGX)")
class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = Brain.load(BRAIN)
        cls.par = json.loads((BRAIN / "parity.json").read_text())

    def test_manifest_matches_binary(self):
        a, m = self.brain.arrays, self.brain.manifest
        self.assertEqual(len(a["csr_indptr"]), m["neurons"] + 1)
        self.assertEqual(int(a["csr_indptr"][-1]), m["edges"])
        self.assertEqual(len(a["csr_post"]), m["edges"])
        self.assertTrue((np.diff(a["csr_indptr"]) >= 0).all())

    def test_fifty_stimuli_and_each_outcome_matches_the_reference(self):
        self.assertEqual(self.par["n"], 50)
        dt = self.par["dt_ms"]
        bad = []
        for tr in self.par["trials"]:
            eng = run_trial(self.brain, np.array(tr["frames"]))
            got = outcome(self.brain, eng, self.brain.manifest["roles"]["short_window_ms"])
            exp = tr["expected"]
            same = got["mode"] == exp["mode"] and got["heading_side"] == exp["heading_side"]
            for k in ("gf_first_ms", "parallel_first_ms", "escape_first_ms"):
                if (got[k] is None) != (exp[k] is None) or (got[k] is not None and abs(got[k] - exp[k]) > dt + 1e-9):
                    same = False
            if not same:
                bad.append((tr["id"], tr["kind"], tr["rv_ms"], tr["azimuth_deg"], got, exp))
        self.assertEqual(bad, [], f"{len(bad)} of 50 trials differ from the reference")


if __name__ == "__main__":
    unittest.main()
