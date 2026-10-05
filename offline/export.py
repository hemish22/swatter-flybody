"""Export the frozen brain for the browser engine, plus 50 parity vectors.

Writes `web/brain/`:

    brain.bin       little-endian arrays, 4-byte aligned, offsets in the manifest
                    csr_indptr  int32   (n + 1)       presynaptic-major row pointers
                    csr_post    uint16  (edges)       postsynaptic neuron
                    csr_weight_mv float32 (edges)     signed mV per spike, weight_scale applied
                    drive_neuron uint16 (m)           graph index of each driven LPLC2 neuron
                    drive_rf_deg float32 (m, 2)       receptive-field centre on its eye's plane
                    drive_eye   uint8   (m)           0 right eye, 1 left eye
    manifest.json   shapes, dtypes, offsets, LIF constants and the three fitted
                    parameters, drive spec, neuron roles, provenance
    parity.json     50 fixed stimuli as engine frames, with the outcome the Python
                    reference (`lif.py` + `loom_sweep.angular_drive`) produces

Plan said float16 weights; float32 is used because the whole brain is ~42k edges
(about 0.5 MB) and float16 would add a rounding the parity test would then have to
excuse. Revisit only if the download size ever matters.

The engine contract (`engine_ref.py`) is: frames in, spikes out. The brain is the
LIF circuit only; the optic lobe is not shipped (docs/week1_gate.md), and the
LPLC2 drive is formed from the frame's growth rate and position.

Parameters in the file are PROVISIONAL (docs/lif_fit.md): they are the held-out
fit's choice, before the Brian2 check and the criterion-3 decision.

Usage (DGX): SWATTER_REMOTE=1 .venv/bin/python offline/export.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402

RF_SIGMA_DEG = 15.0  # loom_sweep.angular_drive's default; the one drive constant
SHORT_WINDOW_MS = 6.87
LPLC2 = {"R": Path("data/ol/lplc2_inputs_R.npz"), "L": Path("data/ol/lplc2_inputs_L.npz")}


def parity_stimuli() -> list[dict]:
    """The 50 fixed stimuli. Spread over r/v, lateral azimuths, controls and edge cases."""
    out = []
    for rv in (10.0, 14.0, 20.0, 28.0, 40.0, 57.0, 80.0):
        for az in (45.0, 90.0, 135.0, 225.0, 270.0, 315.0):
            out.append({"kind": "expanding", "rv_ms": rv, "azimuth_deg": az})
    out += [
        {"kind": "receding", "rv_ms": 20.0, "azimuth_deg": 90.0},
        {"kind": "receding", "rv_ms": 20.0, "azimuth_deg": 270.0},
        {"kind": "translating", "rv_ms": 20.0, "azimuth_deg": 90.0},
        {"kind": "dimming", "rv_ms": 20.0, "azimuth_deg": 90.0},
        {"kind": "expanding", "rv_ms": 20.0, "azimuth_deg": 0.0},
        {"kind": "expanding", "rv_ms": 20.0, "azimuth_deg": 180.0},
        {"kind": "expanding", "rv_ms": 7.0, "azimuth_deg": 90.0},
        {"kind": "expanding", "rv_ms": 120.0, "azimuth_deg": 90.0},
    ]
    assert len(out) == 50
    for i, s in enumerate(out):
        s["id"] = i
    return out


def frames_for(loom, plane_r: np.ndarray, plane_l: np.ndarray) -> np.ndarray:
    """(T, 5) engine frames: growth rate (deg/ms, relu'd as the drive does) and the disc's plane position per eye."""
    if loom.kind == "translating":
        rate = np.zeros(loom.n_steps)  # constant size: no growth, whatever the position
    else:
        rate = np.clip(np.gradient(loom.angular_size_deg(), loom.dt_ms), 0.0, None)
    t = len(rate)
    return np.column_stack([rate, np.tile(plane_r, (t, 1)), np.tile(plane_l, (t, 1))])


def pack(arrays: dict[str, np.ndarray]) -> tuple[bytes, dict]:
    blob, spec, off = bytearray(), {}, 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        spec[name] = {"dtype": a.dtype.str.replace(">", "<"), "shape": list(a.shape), "offset": off}
        raw = a.astype(a.dtype.newbyteorder("<")).tobytes()
        blob += raw
        pad = (-len(blob)) % 4
        blob += b"\x00" * pad
        off = len(blob)
    return bytes(blob), spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("web/brain"))
    args = ap.parse_args()
    require_remote_execution("export.py", "50 reference trials through the LIF circuit")
    import torch

    torch.set_num_threads(8)
    from engine_ref import Brain, Engine, outcome
    from heading import plane_position
    from lif import EscapeGraph, LifParams, upsample_drive, with_params
    from loom import Loom
    from loom_sweep import DURATION_MS, OPTIC_DT_MS, angular_drive, rf_centres

    fit = json.loads(Path("docs/lif_fit.json").read_text())["chosen"]
    params = with_params(LifParams(), input_gain=fit["input_gain"], weight_scale=fit["weight_scale"], v_th=fit["v_th"])
    graph_path = Path("data/ol/escape_graph.npz")
    graph = EscapeGraph.load(graph_path)
    n = graph.n

    # --- the brain: CSR by presynaptic neuron, signed mV, weight_scale baked in
    w = np.zeros((n, n), dtype=np.float32)  # [post, pre], summing duplicate edges as lif.py does
    np.add.at(w, (graph.post, graph.pre), graph.weight * graph.sign[graph.pre] * params.w_syn * params.weight_scale)
    pre_idx, post_idx = np.nonzero(w.T)  # w.T is [pre, post]: rows are presynaptic
    order = np.lexsort((post_idx, pre_idx))
    pre_s, post_s = pre_idx[order], post_idx[order]
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, pre_s + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)

    drive_neuron, drive_rf, drive_eye = [], [], []
    for eye, side in enumerate(("R", "L")):
        z = np.load(LPLC2[side])
        drive_neuron.append(graph.source_slots(z["bodies"]))
        drive_rf.append(rf_centres(LPLC2[side]))
        drive_eye.append(np.full(len(z["bodies"]), eye))
    arrays = {
        "csr_indptr": indptr,
        "csr_post": post_s.astype(np.uint16),
        "csr_weight_mv": w[post_s, pre_s].astype(np.float32),
        "drive_neuron": np.concatenate(drive_neuron).astype(np.uint16),
        "drive_rf_deg": np.concatenate(drive_rf).astype(np.float32),
        "drive_eye": np.concatenate(drive_eye).astype(np.uint8),
    }
    blob, spec = pack(arrays)

    tgt = np.flatnonzero(graph.role == "target")
    gf = np.flatnonzero(graph.type == "DNp01")
    par = np.flatnonzero(np.isin(graph.type, ("DNp04", "DNp103", "DNp02", "DNp11")))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {
        "version": 1,
        "status": "PROVISIONAL: parameters from docs/lif_fit.md, before the Brian2 check and the criterion-3 decision",
        "binary": "brain.bin",
        "arrays": spec,
        "neurons": n,
        "edges": int(len(post_s)),
        "lif": {"v0": params.v0, "v_rst": params.v_rst, "v_th": params.v_th, "t_mbr": params.t_mbr, "tau": params.tau,
                "t_rfc": params.t_rfc, "t_dly": params.t_dly, "w_syn": params.w_syn, "dt_ms": params.dt,
                "n_delay": params.n_delay, "n_refractory": params.n_refractory,
                "weight_scale": params.weight_scale, "input_gain": params.input_gain},
        "drive": {"rf_sigma_deg": RF_SIGMA_DEG, "frame_ms": OPTIC_DT_MS, "substeps": int(round(OPTIC_DT_MS / params.dt)),
                  "formula": "relu(rate_deg_per_ms) * exp(-|rf - pos|^2 / (2 sigma^2)) * input_gain, linear between frames",
                  "eye_axes": "lateral: right eye +90, left eye -90 body azimuth (heading.py; unverified assumption)"},
        "roles": {"gf": gf.tolist(), "parallel": par.tolist(), "target_side": [str(graph.side[i]) for i in np.concatenate([gf, par])],
                  "short_window_ms": SHORT_WINDOW_MS},
        "neuron_type": graph.type.tolist(),
        "neuron_side": graph.side.tolist(),
        "neuron_role": graph.role.tolist(),
        "provenance": {"escape_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(), "repo_commit": commit,
                       "fit": "docs/lif_fit.json chosen"},
    }
    # roles lists above: target_side is aligned with gf + parallel, in that order
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "brain.bin").write_bytes(blob)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # --- parity vectors from the Python reference
    stim = parity_stimuli()
    looms = [Loom(kind=s["kind"], rv_ms=s["rv_ms"], azimuth_deg=s["azimuth_deg"], duration_ms=DURATION_MS, dt_ms=OPTIC_DT_MS) for s in stim]
    azimuths = [s["azimuth_deg"] for s in stim]
    total = 0
    for side in ("R", "L"):
        d = angular_drive(looms, LPLC2[side], rf_sigma_deg=RF_SIGMA_DEG, positions=np.vstack([plane_position(a, side) for a in azimuths]))
        total = total + graph.embed_drive(upsample_drive(d, OPTIC_DT_MS, params.dt, DURATION_MS), np.load(LPLC2[side])["bodies"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spikes = graph.network(params, device).run(total * params.input_gain)["spikes"]

    brain = Brain.load(args.out)
    out = []
    for i, (s, loom) in enumerate(zip(stim, looms)):
        eng = Engine(brain)  # only used as a container for the reference spike list and constants
        eng.spikes = [(int(t), int(j)) for t, j in zip(*np.nonzero(spikes[i]))]
        frames = frames_for(loom, plane_position(s["azimuth_deg"], "R")[0], plane_position(s["azimuth_deg"], "L")[0])
        out.append({**s, "frames": frames.tolist(), "expected": outcome(brain, eng, SHORT_WINDOW_MS),
                    "expected_gf_spike_steps": sorted({int(t) for t, j in eng.spikes if j in set(gf.tolist())})[:5]})
    (args.out / "parity.json").write_text(json.dumps({"n": len(out), "dt_ms": params.dt, "trials": out}))

    banner("export")
    print(f"neurons {n}, edges {len(post_s)}, brain.bin {len(blob) / 1024:.0f} KiB, parity.json {(args.out / 'parity.json').stat().st_size / 1024:.0f} KiB")
    print("modes:", {m: sum(o["expected"]["mode"] == m for o in out) for m in ("short", "long", "none")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
