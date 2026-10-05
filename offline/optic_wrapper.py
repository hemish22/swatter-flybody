"""The optic lobe: flyvis dynamics behind one small interface.

Everything downstream (the LPLC2 / LC4 input stage, the sweep harness, the
export to the browser) talks to `OpticLobe`, not to flyvis. That is what lets
the wiring underneath change without touching the stages after it:

    backend="flyvis"    the pretrained flyvis network on its own (FlyWire/FIB)
                        wiring. Works today; the Week 1 gate baseline.
    backend="malecns"   the same dynamics on the MaleCNS optic-lobe wiring
                        (see `malecns_ol.py`; not wired in yet).

Input is luminance on the 721-column lattice of `eye.py`, in [0, 1], one frame
per integration step. Output is activity per cell type, as (batch, time, 721)
arrays in the same column order, NaN where a type has no cell in that column
(flyvis thins Lawf1/Lawf2 and a few others).

Integration step. flyvis is an explicit-Euler network whose step is chosen by
the caller, not baked in: it warns above 20 ms (the step the pretrained
ensemble was trained at) and its fastest time constant is 19 ms, so 5 ms is
stable and four times finer than training. `docs/optic_lobe.md` records how much
the T4/T5 and Tm responses move between 20, 10 and 5 ms; the LIF stages
interpolate whatever step is chosen here down to 0.5-1 ms.

Heavy only when a sweep is run (it takes the GPU); importing and a single trial
are cheap. Select the GPU with CUDA_VISIBLE_DEVICES; flyvis picks cuda if it
can see one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Where flyvis keeps its pretrained ensemble. Must be set before flyvis is
# imported, so it is set here and not left to the caller.
_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FLYVIS_ROOT_DIR", str(_ROOT / "data" / "flyvis"))

import torch  # noqa: E402

# Motion detectors: the planned input to the radial-motion template if the
# Week 1 gate takes the partial path (docs/build-plan.md, go/no-go table).
MOTION_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

# Integration step in seconds. See the module docstring.
DEFAULT_DT_S = 0.005

ENSEMBLE = "flow/0000"


class OpticLobe:
    """flyvis network + the bookkeeping to read per-type activity on the lattice."""

    def __init__(
        self,
        backend: str = "flyvis",
        member: int = 0,
        dt_s: float = DEFAULT_DT_S,
        device: str | None = None,
        settle_s: float = 1.0,
    ):
        self.backend = backend
        self.member = member
        self.dt_s = float(dt_s)
        self.settle_s = float(settle_s)
        self.net = self._load(backend, member)
        self.device = torch.device(device) if device else next(self.net.parameters()).device
        self.net.to(self.device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self._index_columns()
        self._state = None

    # -- construction ------------------------------------------------------ #

    @staticmethod
    def _load(backend: str, member: int):
        if backend == "flyvis":
            import flyvis

            view = flyvis.NetworkView(flyvis.results_dir / f"{ENSEMBLE}/{member:03d}")
            return view.init_network()
        if backend == "malecns":
            raise NotImplementedError("the MaleCNS backend is built in malecns_ol.py and not wired in yet")
        raise ValueError(f"unknown backend {backend!r}")

    def _index_columns(self) -> None:
        """Map every cell type to (node indices, column index per node).

        Column order is flyvis's (u, v) order, which `eye.flyvis_columns` also
        produces, so a column index means the same thing in the renderer and here.
        """
        nodes = self.net.connectome.nodes
        u = np.asarray(nodes.u[:])
        v = np.asarray(nodes.v[:])
        types = np.asarray(nodes.type[:]).astype(str)
        r = types == "R1"
        self.n_columns = int(r.sum())
        col_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u[r], v[r]))}
        self.columns_uv = np.stack([u[r], v[r]], axis=1)
        self._type_nodes: dict[str, np.ndarray] = {}
        self._type_cols: dict[str, np.ndarray] = {}
        for t in dict.fromkeys(types.tolist()):
            idx = np.flatnonzero(types == t)
            self._type_nodes[t] = idx
            self._type_cols[t] = np.array([col_of[(int(u[i]), int(v[i]))] for i in idx])
        self.cell_types = tuple(self._type_nodes)

    # -- dynamics ---------------------------------------------------------- #

    def _movie(self, lum: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(lum, dtype=torch.float32, device=self.device)
        if x.ndim == 2:  # (T, n) -> single trial
            x = x[None]
        if x.shape[-1] != self.n_columns:
            raise ValueError(f"expected {self.n_columns} columns, got {x.shape[-1]}")
        return x[:, :, None, :]  # (B, T, 1 channel, hexals)

    @torch.no_grad()
    def _settle(self, batch: int, background: float | torch.Tensor, settle_s: float | None = None):
        """Steady state under a held field, the state a trial starts from.

        `background` is a scalar (uniform field) or a (batch, 721) frame held
        constant, which is the fly having watched the first frame of the trial
        for `settle_s`. Done one step at a time keeping only the last state:
        `Network.steady_state` returns the last of a list of ALL states, which
        at batch 32 and a 5 ms step is 200 copies of the 1.5M-edge state and
        does not fit on a 40 GB A100.
        """
        settle_s = self.settle_s if settle_s is None else settle_s
        state = None
        frame = None
        if torch.is_tensor(background):
            frame = background.to(self.device, torch.float32)[:, None, None, :]
        for _ in range(int(round(settle_s / self.dt_s))):
            self.net.stimulus.zero(batch, 1)
            if frame is None:
                self.net.stimulus.add_pre_stim(float(background))
            else:
                self.net.stimulus.add_input(frame)
            state = self.net(self.net.stimulus(), self.dt_s, state=state, as_states=True)[-1]
        return state

    def reset(self, batch: int, background: float = 1.0, settle_s: float | None = None) -> None:
        self._state = self._settle(batch, background, settle_s)

    @torch.no_grad()
    def step(self, lum_t: np.ndarray | torch.Tensor) -> None:
        """Advance one integration step on a (batch, 721) frame. For streaming use."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        x = self._movie(torch.as_tensor(lum_t)[:, None, :])
        self.net.stimulus.zero(x.shape[0], 1)
        self.net.stimulus.add_input(x)
        self._state = self.net(self.net.stimulus(), self.dt_s, state=self._state, as_states=True)[-1]

    def read(self, types: Iterable[str]) -> dict[str, np.ndarray]:
        """Current activity of the named types as (batch, 721), NaN where absent."""
        act = self._state.nodes.activity
        return {t: self._to_columns(act[:, self._type_nodes[t]], t) for t in types}

    @torch.no_grad()
    def run(
        self,
        lum: np.ndarray | torch.Tensor,
        record: Sequence[str] = MOTION_TYPES,
        background: float | str | None = 1.0,
        chunk: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run whole movies (B, T, 721) from steady state; activity of `record` types.

        `background`: a number settles the network on a uniform field of that
        luminance; "first_frame" settles each trial on its own first frame, so
        a disc that is already on screen at t=0 does not arrive as an onset
        flash (the loom controls start with a ~14 degree disc, the expanding
        loom with a 5 degree one, and onset transients swamp the comparison).

        Returns {type: (B, T, 721)}. Trials share nothing, so a batch is just
        throughput; `chunk` bounds the batch if memory is tight.
        """
        movie = self._movie(lum)
        B, T = movie.shape[:2]
        if chunk and B > chunk:
            parts = [self.run(movie[i : i + chunk, :, 0], record, background) for i in range(0, B, chunk)]
            return {t: np.concatenate([p[t] for p in parts], axis=0) for t in record}
        if background is None:
            state = None
        elif background == "first_frame":
            state = self._settle(B, movie[:, 0, 0, :])
        else:
            state = self._settle(B, float(background))
        self.net.stimulus.zero(B, T)
        self.net.stimulus.add_input(movie)
        act = self.net(self.net.stimulus(), self.dt_s, state=state)  # (B, T, n_nodes)
        return {t: self._to_columns(act[:, :, self._type_nodes[t]], t) for t in record}

    # -- helpers ----------------------------------------------------------- #

    def _to_columns(self, a: torch.Tensor, t: str) -> np.ndarray:
        a = a.detach().float().cpu().numpy()
        cols = self._type_cols[t]
        if cols.size == self.n_columns and np.array_equal(cols, np.arange(self.n_columns)):
            return a
        out = np.full(a.shape[:-1] + (self.n_columns,), np.nan, dtype=np.float32)
        out[..., cols] = a
        return out

    @property
    def n_cells(self) -> int:
        return int(self.net.n_nodes)
