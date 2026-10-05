"""Tests that run the real flyvis network. DGX only (SWATTER_REMOTE=1), on CPU.

They need the pretrained weights under data/flyvis (`flyvis download-pretrained`
with FLYVIS_ROOT_DIR=data/flyvis) and take about a minute, so they stay out of
the laptop's `make test`. They run on the CPU on purpose: a unit test should
not take someone else's GPU.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))

from common import on_remote  # noqa: E402

WEIGHTS = ROOT / "data" / "flyvis" / "results" / "flow" / "0000" / "000"


@unittest.skipUnless(on_remote() and WEIGHTS.exists(), "needs SWATTER_REMOTE=1 and the pretrained flyvis weights")
class TestOpticLobe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch

        torch.set_num_threads(8)  # shared host: do not take all 256 cores for a unit test
        from eye import Eye
        from optic_wrapper import OpticLobe

        cls.eye = Eye.flyvis()
        cls.lobe = OpticLobe(device="cpu", settle_s=0.4)

    def test_columns_are_in_the_eyes_order(self):
        """If this drifts, the renderer and the network disagree about which column is which."""
        self.assertTrue((self.lobe.columns_uv == self.eye.uv).all())
        self.assertEqual(self.lobe.n_columns, 721)

    def test_streaming_step_matches_whole_movie_run(self):
        rng = np.random.default_rng(0)
        x = (0.5 + 0.2 * rng.standard_normal((2, 6, 721))).clip(0, 1).astype(np.float32)
        whole = self.lobe.run(x, record=("T4a",), background=0.5)["T4a"]
        self.lobe.reset(2, 0.5)
        stepped = []
        for i in range(6):
            self.lobe.step(x[:, i])
            stepped.append(self.lobe.read(["T4a"])["T4a"])
        np.testing.assert_allclose(np.stack(stepped, 1), whole, atol=1e-4)

    def test_settling_on_the_first_frame_removes_the_onset_flash(self):
        """A disc already on screen at t=0 must not arrive as a stimulus onset.

        Settling on the bare sky made translating and dimming controls score
        above the expanding loom, because their ~14 degree disc 'appeared' at
        t=0. This pins that the first-frame settle starts them at rest.
        """
        from eye import render_loom
        from loom import Loom

        movie = render_loom(Loom(kind="dimming", rv_ms=20.0, azimuth_deg=0.0, dt_ms=5.0), self.eye)[None, :12]
        sky = self.lobe.run(movie, record=("L2",), background=1.0)["L2"]
        adapted = self.lobe.run(movie, record=("L2",), background="first_frame")["L2"]
        onset_sky = float(np.abs(sky - sky[:, :1]).max())
        onset_adapted = float(np.abs(adapted - adapted[:, :1]).max())
        self.assertGreater(onset_sky, 5 * onset_adapted)

    def test_t5a_prefers_leftward_motion_in_the_eye_plane(self):
        """T5a's preferred direction is 180 degrees in flyvis's plane, which is the renderer's plane."""
        from eye import edge_movie

        movies = np.stack([edge_movie(self.eye, phi, 300.0, 0.005, 0.6, "off") for phi in (180, 0)])
        a = self.lobe.run(movies, record=("T5a",), background=0.5)["T5a"]
        peak = (a - a[:, :1]).max(axis=1).mean(axis=1)
        self.assertGreater(peak[0], 1.5 * peak[1])


if __name__ == "__main__":
    unittest.main()
