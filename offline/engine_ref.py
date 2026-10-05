"""Reference browser engine, in numpy: the specification the Rust/WASM port is tested against.

It consumes exactly what `export.py` writes (`web/brain/brain.bin` + `manifest.json`)
and nothing else, and it is deliberately written the way the browser will run:

  - event-driven: a spike scatters its CSR row into a ring buffer of pending
    conductance, instead of the dense matrix product `lif.py` uses;
  - streaming: the game hands over one frame (every 5 ms) and gets the 25
    sub-steps of 0.2 ms back, never the whole trial;
  - self-contained: the LPLC2 drive is formed from the frame itself, so the
    browser needs no optic lobe and no stimulus library.

Frame = (rate, xR, yR, xL, yL): the swatter's angular growth rate in deg/ms and its
position on each eye's plane in degrees. LPLC2 neuron i on eye e is driven by

    relu(rate) * exp(-|rf_i - pos_e|^2 / (2 sigma^2)) * input_gain      [mV/ms]

and drive is linearly interpolated between frames, the same as `lif.upsample_drive`.

Parity target: `lif.LifNetwork` fed the same drive. The two sum in different
orders in float32, so spike times may differ by a sub-step on a borderline
crossing; the exported parity vectors compare outcomes and first-spike times to
one step, which is also what leaderboard verification compares (outcome, mode,
heading side), not raw floats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUBSTEPS = 25  # 5 ms frame / 0.2 ms step


@dataclass
class Brain:
    manifest: dict
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, directory: Path) -> "Brain":
        m = json.loads((directory / "manifest.json").read_text())
        raw = (directory / m["binary"]).read_bytes()
        arrays = {}
        for name, spec in m["arrays"].items():
            a = np.frombuffer(raw, dtype=np.dtype(spec["dtype"]).newbyteorder("<"), count=int(np.prod(spec["shape"])), offset=spec["offset"])
            arrays[name] = a.reshape(spec["shape"]).astype(a.dtype.newbyteorder("="))
        return cls(m, arrays)


class Engine:
    """One trial's state. `frame()` advances 5 ms; call `finish()` for the last sample."""

    def __init__(self, brain: Brain):
        m = brain.manifest
        p = m["lif"]
        a = brain.arrays
        self.p = p
        self.n = m["neurons"]
        self.indptr = a["csr_indptr"]
        self.post = a["csr_post"]
        self.w = a["csr_weight_mv"]  # signed, weight_scale already applied
        self.drive_neuron = a["drive_neuron"]  # graph index of each driven LPLC2 neuron
        self.drive_rf = a["drive_rf_deg"]  # (n_driven, 2)
        self.drive_eye = a["drive_eye"]  # 0 = right eye, 1 = left eye
        self.sigma = m["drive"]["rf_sigma_deg"]
        self.dt = p["dt_ms"]
        self.n_delay = p["n_delay"]
        self.n_refractory = p["n_refractory"]
        self.v = np.full(self.n, p["v0"], dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.int32)
        self.ring = np.zeros((self.n_delay, self.n), dtype=np.float32)
        self.step_index = 0
        self.prev_drive: np.ndarray | None = None
        self.spikes: list[tuple[int, int]] = []  # (step, neuron)

    def _drive(self, frame) -> np.ndarray:
        """(n_driven,) mV/ms, before input_gain."""
        rate = max(float(frame[0]), 0.0)
        pos = np.array([[frame[1], frame[2]], [frame[3], frame[4]]], dtype=np.float64)
        d2 = ((self.drive_rf.astype(np.float64) - pos[self.drive_eye]) ** 2).sum(axis=1)
        return (rate * np.exp(-d2 / (2.0 * self.sigma**2))).astype(np.float32)

    def _substep(self, x: np.ndarray) -> None:
        """One 0.2 ms step with per-neuron external input `x` (mV/ms, gain applied)."""
        p, t = self.p, self.step_index
        slot = t % self.n_delay
        self.g += self.ring[slot] + x * np.float32(self.dt)
        self.ring[slot] = 0.0
        active = self.refr <= 0
        dv = (p["v0"] - self.v + self.g) * np.float32(self.dt / p["t_mbr"])
        self.v = np.where(active, self.v + dv, np.float32(p["v_rst"])).astype(np.float32)
        self.g = self.g - self.g * np.float32(self.dt / p["tau"])
        fired = (self.v > p["v_th"]) & active
        self.v[fired] = p["v_rst"]
        self.refr = np.where(fired, self.n_refractory, self.refr - 1).astype(np.int32)
        for j in np.flatnonzero(fired):
            lo, hi = self.indptr[j], self.indptr[j + 1]
            np.add.at(self.ring[slot], self.post[lo:hi], self.w[lo:hi])  # arrives n_delay steps from now
            self.spikes.append((t, int(j)))
        self.step_index += 1

    def _x(self, drive: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n, dtype=np.float32)
        x[self.drive_neuron] = drive * np.float32(self.p["input_gain"])
        return x

    def frame(self, frame) -> None:
        """Take the next frame. The first call only latches it; each later one runs the 25 steps
        between the previous frame and this one (so the engine is one frame, 5 ms, behind)."""
        cur = self._drive(frame)
        if self.prev_drive is not None:
            for i in range(SUBSTEPS):
                f = np.float32(i / SUBSTEPS)
                self._substep(self._x(self.prev_drive * (1 - f) + cur * f))
        self.prev_drive = cur

    def finish(self) -> None:
        """The trial's last sample, at the final frame's own drive."""
        if self.prev_drive is not None:
            self._substep(self._x(self.prev_drive))


def run_trial(brain: Brain, frames: np.ndarray) -> Engine:
    eng = Engine(brain)
    for f in frames:
        eng.frame(f)
    eng.finish()
    return eng


def outcome(brain: Brain, eng: Engine, short_window_ms: float) -> dict:
    """First-spike times of the giant fiber (either side) and the parallel DNs, and the heading side."""
    m = brain.manifest
    dt = eng.dt
    roles = m["roles"]
    gf = set(roles["gf"])
    par = set(roles["parallel"])
    tgt = roles["gf"] + roles["parallel"]
    side = {int(i): s for i, s in zip(tgt, roles["target_side"])}

    def first(ids):
        ts = [t for t, j in eng.spikes if j in ids]
        return None if not ts else min(ts) * dt

    t_gf, t_par = first(gf), first(par)
    any_t = [t for t, j in eng.spikes if j in side]
    out = {"gf_first_ms": t_gf, "parallel_first_ms": t_par, "escape_first_ms": None, "A": 0.0, "heading_side": None}
    if any_t:
        t0 = min(any_t)
        out["escape_first_ms"] = t0 * dt
        win = [j for t, j in eng.spikes if j in side and t0 <= t < t0 + int(round(20.0 / dt))]
        r = sum(side[j] == "R" for j in win)
        l = sum(side[j] == "L" for j in win)
        out["A"] = (r - l) / max(1, r + l)
        out["heading_side"] = "left" if r > l else ("right" if l > r else None)  # heading is AWAY from the stronger side
    if t_gf is None and t_par is None:
        out["mode"] = "none"
    elif t_par is None:
        out["mode"] = "short"
    elif t_gf is None:
        out["mode"] = "long"
    else:
        out["mode"] = "short" if t_gf <= t_par + short_window_ms else "long"
    return out
