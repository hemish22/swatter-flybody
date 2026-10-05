"""Reference LIF simulator for the escape circuit, and the optic-lobe interface.

Point-neuron leaky integrate-and-fire, in the form and with the constants of
Shiu et al. 2024 (the Drosophila whole-brain LIF model the plan cross-checks
against), written for batches of trials on a GPU:

    dv/dt = (v0 - v + g) / t_mbr          membrane, mV
    dg/dt = -g / tau                      synaptic conductance as a voltage, mV
    spike when v > v_th: v <- v_rst, held for t_rfc
    a spike reaches every target t_dly later and adds
        sign(pre) * w_syn * (synapse count)   to g

    v0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms,
    t_rfc = 2.2 ms, t_dly = 1.8 ms, w_syn = 0.275 mV per synapse.

The constants are from memory of that paper and NOT yet cross-checked against
its Brian2 code (the plan's "LIF cross-check" row). Until that is done this is
"a LIF model in the style of Shiu et al.", and the docs say so.

Scale to keep in mind: one spike's peak PSP is ~0.043 mV per synapse (0.275 mV
conductance jump x 0.157, the peak of the 5 ms / 20 ms double exponential), so
crossing the 7 mV gap from rest takes ~162 synapses on one spike. Most
LPLC2 -> DN pairs are tens of synapses (4,862 onto the GF from 185 neurons), so
the escape DNs fire on many LPLC2 spikes arriving together, not on one.

Deterministic by construction: no noise, fixed-step Euler, no data-dependent
branching, so a trial is a function of its drive. The leaderboard re-sim and the
browser parity test (plan, Week 3) both depend on that.

What is NOT here, and is stated so it is not mistaken for a feature:
  - electrical synapses (the GF's fastest inputs are electrical and EM records
    them poorly; the chemical path into the GF is incomplete);
  - any neuron the 495-neuron graph does not contain;
  - noise, adaptation, graded release.

The fitted parameters, frozen after the Week 2 fit, are exactly three:
  `input_gain`   optic-lobe drive -> current onto the LPLC2 neurons
  `weight_scale` multiplies every synaptic weight (the plan's "global gain")
  `v_th`         spike threshold
Nothing else is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class LifParams:
    # constants (Shiu et al. 2024; see module docstring)
    v0: float = -52.0
    v_rst: float = -52.0
    t_mbr: float = 20.0
    tau: float = 5.0
    t_rfc: float = 2.2
    t_dly: float = 1.8
    w_syn: float = 0.275
    # the three fitted parameters
    v_th: float = -45.0
    weight_scale: float = 1.0
    input_gain: float = 0.01  # mV/ms of conductance per unit of optic drive; provisional
    # integration step
    dt: float = 0.2  # ms

    @property
    def n_refractory(self) -> int:
        return int(round(self.t_rfc / self.dt))

    @property
    def n_delay(self) -> int:
        return max(1, int(round(self.t_dly / self.dt)))


class LifNetwork:
    """A fixed network: signed weights in mV, dense (the graph is ~500 neurons)."""

    def __init__(self, weights_mv: np.ndarray, params: LifParams = LifParams(), device: str | torch.device | None = None):
        """weights_mv[post, pre]: mV added to the post neuron per pre spike."""
        self.params = params
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.n = int(weights_mv.shape[0])
        self.w = torch.as_tensor(weights_mv * params.weight_scale, dtype=torch.float32, device=self.device)

    @classmethod
    def from_edges(cls, n: int, pre, post, count, sign, params: LifParams = LifParams(), device=None) -> "LifNetwork":
        """Build from synapse counts: w = sign(pre) * w_syn * count, summed over duplicate edges."""
        w = np.zeros((n, n), dtype=np.float32)
        np.add.at(w, (np.asarray(post), np.asarray(pre)), np.asarray(count, dtype=np.float32) * np.asarray(sign)[np.asarray(pre)] * params.w_syn)
        return cls(w, params, device)

    @torch.no_grad()
    def run(
        self,
        drive: np.ndarray | torch.Tensor,
        record_v: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Simulate B trials of T steps.

        drive: (B, T, n) external input in mV/ms of conductance, at this
            network's own step (`params.dt`). Zero where a neuron has no optic input.
        record_v: neuron indices whose membrane trace to return.

        Returns spikes (B, T, n) bool and, if asked, v (B, T, len(record_v)).
        """
        p = self.params
        x = torch.as_tensor(drive, dtype=torch.float32, device=self.device)
        B, T, n = x.shape
        if n != self.n:
            raise ValueError(f"drive has {n} neurons, network has {self.n}")
        v = torch.full((B, n), p.v0, device=self.device)
        g = torch.zeros((B, n), device=self.device)
        refr = torch.zeros((B, n), dtype=torch.int32, device=self.device)
        d = p.n_delay
        # buf[k] holds the spikes that arrive k steps from now, k = 0..d-1
        buf = torch.zeros((d, B, n), device=self.device)
        spikes = torch.zeros((B, T, n), dtype=torch.bool, device=self.device)
        rec = None if record_v is None else torch.as_tensor(record_v, device=self.device)
        vtrace = None if rec is None else torch.zeros((B, T, len(rec)), device=self.device)
        for t in range(T):
            arriving = buf[t % d]
            g = g + arriving @ self.w.T + x[:, t] * p.dt
            buf[t % d] = 0.0
            active = refr <= 0
            dv = (p.v0 - v + g) * (p.dt / p.t_mbr)
            v = torch.where(active, v + dv, torch.full_like(v, p.v_rst))
            g = g - g * (p.dt / p.tau)
            fired = (v > p.v_th) & active
            v = torch.where(fired, torch.full_like(v, p.v_rst), v)
            refr = torch.where(fired, torch.full_like(refr, p.n_refractory), refr - 1)
            # a spike now arrives d steps from now, which is this slot's next visit
            buf[t % d] = fired.float()
            spikes[:, t] = fired
            if vtrace is not None:
                vtrace[:, t] = v[:, rec]
        out = {"spikes": spikes.cpu().numpy()}
        if vtrace is not None:
            out["v"] = vtrace.cpu().numpy()
        return out


def upsample_drive(drive: np.ndarray, dt_in_ms: float, dt_out_ms: float, duration_ms: float) -> np.ndarray:
    """Linear interpolation of (B, T_in, n) drive sampled every `dt_in_ms` onto the LIF grid.

    The optic lobe runs at 5 ms and the LIF at 0.2 ms; plan: "the LIF stages
    interpolate its outputs". Linear, not zero-order hold, so the drive has no
    5 ms staircase for the membrane to resonate with.
    """
    t_in = np.arange(drive.shape[1]) * dt_in_ms
    t_out = np.arange(int(round(duration_ms / dt_out_ms)) + 1) * dt_out_ms
    t_out = t_out[t_out <= t_in[-1] + 1e-9]
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, len(t_in) - 2)
    frac = ((t_out - t_in[idx]) / dt_in_ms)[None, :, None]
    return (drive[:, idx] * (1.0 - frac) + drive[:, idx + 1] * frac).astype(np.float32)


@dataclass
class EscapeGraph:
    """The data/ol/escape_graph.npz arrays, with the lookups the stages need."""

    bodies: np.ndarray
    role: np.ndarray
    type: np.ndarray
    side: np.ndarray
    sign: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "EscapeGraph":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in ("bodies", "role", "type", "side", "sign", "pre", "post", "weight")})

    @property
    def n(self) -> int:
        return int(self.bodies.size)

    def index_of_type(self, t: str) -> np.ndarray:
        return np.flatnonzero(self.type == t)

    def network(self, params: LifParams = LifParams(), device=None) -> LifNetwork:
        return LifNetwork.from_edges(self.n, self.pre, self.post, self.weight, self.sign, params, device)

    def source_slots(self, lplc2_bodies: np.ndarray) -> np.ndarray:
        """Graph index of each body in `lplc2_bodies` (the order `lplc2_inputs_*.npz` uses)."""
        where = {int(b): i for i, b in enumerate(self.bodies)}
        return np.array([where[int(b)] for b in lplc2_bodies])

    def embed_drive(self, drive: np.ndarray, lplc2_bodies: np.ndarray) -> np.ndarray:
        """(B, T, n_lplc2) optic drive -> (B, T, n_graph), zero on every other neuron."""
        out = np.zeros(drive.shape[:2] + (self.n,), dtype=np.float32)
        out[..., self.source_slots(lplc2_bodies)] = drive
        return out


def with_params(params: LifParams, **kw) -> LifParams:
    return replace(params, **kw)
