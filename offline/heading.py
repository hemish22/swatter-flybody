"""Criterion 4: takeoff heading from left/right escape-DN asymmetry, both eyes driven.

Plan: "v1 reads takeoff heading from left/right asymmetry in LPLC2, LC4 and DN
activity, biased away from the stronger side"; a labelled simplification (real
flies set heading through pre-takeoff leg posture). This is that rule, read off
the DNs, with no fitted parameters: the three LIF parameters are the ones chosen
by `lif_fit.py`.

Geometry (assumptions, all stated):
  - the loom is in the horizontal plane at body azimuth phi (0 = ahead, +90 = the
    fly's right), at the same angular-size time course as the lab looms;
  - each eye's optical axis points laterally (right eye +90, left eye -90). Real
    axes are not measured here; this fixes where a stimulus lands on each eye;
  - a stimulus at angle `ecc` from an eye's axis lands at plane coordinates
    (ecc, 0) if it is ahead of the lateral plane and (-ecc, 0) if behind. Which
    sign is "anterior" in the flyvis plane is not pinned by anything in this repo;
    it only swaps front and back, never left and right, so criterion 4 does not
    depend on it;
  - the left eye is the mirror image: it sees phi as the right eye sees -phi, and
    its LPLC2 neurons keep their own measured receptive fields (`lplc2_inputs_L`).

Readout: the escape DNs (giant fiber and the four parallel DN types, both sides).
Window = 20 ms from the first spike of any of them; A = (R - L) / (R + L) over
spike counts in that window, 0 if nothing fired. A > 0 means the right side is
stronger, so the heading is biased left (away). `first_side` is the side of the
first DN to spike, as a cross-check.

Usage (DGX, one GPU, about a minute):
    CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/heading.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402

WINDOW_MS = 20.0
LATERAL_MIN_SIN = float(np.sin(np.radians(22.5)))  # |sin phi| above this counts as a lateral stimulus
RVS = (10.0, 20.0, 40.0, 80.0)


def plane_position(phi_deg, side: str) -> np.ndarray:
    """Eye-plane coordinates (deg) of a horizontal-plane stimulus at body azimuth phi, for one eye."""
    phi = np.radians(np.atleast_1d(np.asarray(phi_deg, dtype=float)))
    lateral = np.sin(phi) * (1.0 if side == "R" else -1.0)  # component along this eye's axis
    ecc = np.degrees(np.arccos(np.clip(lateral, -1.0, 1.0)))
    sign = np.where(np.cos(phi) >= 0.0, 1.0, -1.0)
    return np.stack([sign * ecc, np.zeros_like(ecc)], axis=1)


def asymmetry(spikes: np.ndarray, targets: np.ndarray, side: np.ndarray, dt_ms: float) -> dict:
    """A, counts and first-DN side for one trial's spikes (T, n)."""
    first = np.flatnonzero(spikes[:, targets].any(axis=1))
    if first.size == 0:
        return {"fired": False, "A": 0.0, "right": 0, "left": 0, "first_side": None, "first_ms": None}
    t0 = int(first[0])
    win = spikes[t0 : t0 + int(round(WINDOW_MS / dt_ms))]
    right = int(win[:, targets[side[targets] == "R"]].sum())
    left = int(win[:, targets[side[targets] == "L"]].sum())
    at0 = spikes[t0, targets]
    first_sides = set(side[targets][at0])
    return {
        "fired": True,
        "A": (right - left) / max(1, right + left),
        "right": right,
        "left": left,
        "first_side": first_sides.pop() if len(first_sides) == 1 else "both",
        "first_ms": t0 * dt_ms,
    }


def heading_trials(graph, params, azimuths) -> list[dict]:
    """Bilateral expanding looms at each body azimuth and r/v through `graph`; one asymmetry row per trial."""
    from lif import upsample_drive
    from loom import Loom
    from loom_sweep import DURATION_MS, OPTIC_DT_MS, angular_drive

    targets = np.flatnonzero(graph.role == "target")
    net = graph.network(params, "cuda")
    rows = []
    for rv in RVS:
        looms = [Loom(kind="expanding", rv_ms=rv, azimuth_deg=a, duration_ms=DURATION_MS, dt_ms=OPTIC_DT_MS) for a in azimuths]
        total = 0
        for side in ("R", "L"):
            npz = Path(f"data/ol/lplc2_inputs_{side}.npz")
            drive = angular_drive(looms, npz, positions=plane_position(azimuths, side))
            total = total + graph.embed_drive(upsample_drive(drive, OPTIC_DT_MS, params.dt, DURATION_MS), np.load(npz)["bodies"])
        spikes = net.run(total * params.input_gain)["spikes"]
        for i, a in enumerate(azimuths):
            rows.append({"rv_ms": rv, "azimuth_deg": a, **asymmetry(spikes[i], targets, graph.side, params.dt)})
    return rows


def summarise(rows: list[dict]) -> dict:
    lateral = [r for r in rows if abs(np.sin(np.radians(r["azimuth_deg"]))) >= LATERAL_MIN_SIN and r["fired"]]
    ahead_behind = [r for r in rows if abs(np.sin(np.radians(r["azimuth_deg"]))) < 1e-9 and r["fired"]]
    right_side = lambda r: np.sin(np.radians(r["azimuth_deg"])) > 0
    # A > 0 (right DNs stronger) must go with a stimulus on the right
    correct = [np.sign(r["A"]) == (1 if right_side(r) else -1) for r in lateral if r["A"] != 0]
    first_ok = [r["first_side"] == ("R" if right_side(r) else "L") for r in lateral]
    return {
        "n_trials": len(rows), "n_fired": sum(r["fired"] for r in rows),
        "lateral_trials_fired": len(lateral),
        "lateral_heading_away_fraction": float(np.mean(correct)) if correct else None,
        "lateral_first_dn_ipsilateral_fraction": float(np.mean(first_ok)) if first_ok else None,
        "lateral_mean_abs_A": float(np.mean([abs(r["A"]) for r in lateral])) if lateral else None,
        "ahead_behind_mean_abs_A": float(np.mean([abs(r["A"]) for r in ahead_behind])) if ahead_behind else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=Path, default=Path("docs/lif_fit.json"))
    ap.add_argument("--graph", type=Path, default=Path("data/ol/escape_graph.npz"))
    ap.add_argument("--save", type=Path, default=Path("docs/heading.json"))
    args = ap.parse_args()
    require_remote_execution("heading.py", "bilateral loom trials through the LIF circuit on a GPU")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: set CUDA_VISIBLE_DEVICES to a free GPU (nvidia-smi)")
    from lif import EscapeGraph, LifParams, with_params

    fit = json.loads(args.fit.read_text())
    ch = fit["chosen"]
    params = with_params(LifParams(), input_gain=ch["input_gain"], weight_scale=ch["weight_scale"], v_th=ch["v_th"])
    graph = EscapeGraph.load(args.graph)
    azimuths = [float(a) for a in np.arange(0, 360, 22.5)]  # the plan's 8 plus 8 in between
    rows = heading_trials(graph, params, azimuths)
    summary = summarise(rows)
    summary["params"] = {"input_gain": ch["input_gain"], "weight_scale": ch["weight_scale"], "v_th": ch["v_th"]}
    banner("criterion 4: heading from left/right DN asymmetry")
    print(f"{'az':>6} | " + " ".join(f"{'rv ' + str(int(rv)):>9}" for rv in RVS) + "   (A = (R-L)/(R+L); '-' no escape DN spike)")
    for a in azimuths:
        cells = []
        for rv in RVS:
            r = next(x for x in rows if x["rv_ms"] == rv and x["azimuth_deg"] == a)
            cells.append(f"{r['A']:+9.2f}" if r["fired"] else f"{'-':>9}")
        print(f"{a:6.1f} | " + " ".join(cells))
    print(json.dumps(summary, indent=2))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"summary": summary, "trials": rows}, indent=2, default=float))
    print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
