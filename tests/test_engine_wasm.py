"""The Rust engine, built to WASM and run under wasmtime, against the 50 parity stimuli.

Needs web/brain/engine.wasm (build: `cd engine && cargo build --release --target wasm32-unknown-unknown`,
then copy the .wasm into web/brain/) and `pip install wasmtime`.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))
BRAIN = ROOT / "web" / "brain"

try:
    import wasmtime  # noqa: F401
    HAVE = (BRAIN / "engine.wasm").exists() and (BRAIN / "parity.json").exists()
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "web/brain/engine.wasm or wasmtime missing")
class TestWasmParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine_ref import Brain, Engine, outcome
        from wasm_host import WasmEngine

        cls.brain = Brain.load(BRAIN)
        cls.outcome, cls.Engine = staticmethod(outcome), Engine
        cls.wasm = WasmEngine(BRAIN)
        cls.par = json.loads((BRAIN / "parity.json").read_text())

    def test_wasm_matches_the_python_reference_outcomes(self):
        dt = self.par["dt_ms"]
        bad = []
        for tr in self.par["trials"]:
            eng = self.Engine(self.brain)
            eng.spikes = self.wasm.run(np.array(tr["frames"]))
            got = self.outcome(self.brain, eng, self.brain.manifest["roles"]["short_window_ms"])
            exp = tr["expected"]
            same = got["mode"] == exp["mode"] and got["heading_side"] == exp["heading_side"]
            for k in ("gf_first_ms", "parallel_first_ms", "escape_first_ms"):
                if (got[k] is None) != (exp[k] is None) or (got[k] is not None and abs(got[k] - exp[k]) > dt + 1e-9):
                    same = False
            if not same:
                bad.append((tr["id"], tr["kind"], tr["rv_ms"], tr["azimuth_deg"], got, exp))
        self.assertEqual(bad, [], f"{len(bad)} of 50 trials differ")

    def test_wasm_spike_trains_equal_the_numpy_engine_exactly_on_a_trial(self):
        from engine_ref import run_trial

        tr = self.par["trials"][8]
        ref = run_trial(self.brain, np.array(tr["frames"])).spikes
        self.assertEqual(self.wasm.run(np.array(tr["frames"])), ref)

    def test_a_second_run_on_the_same_instance_is_identical(self):
        fr = np.array(self.par["trials"][3]["frames"])
        self.assertEqual(self.wasm.run(fr), self.wasm.run(fr))


if __name__ == "__main__":
    unittest.main()
