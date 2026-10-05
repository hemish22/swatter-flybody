"""Criteria 1 and 2 end to end: eye -> optic lobe -> LPLC2 -> LIF -> giant fiber.

The Week 1 gate asks whether the simulated fly's giant fiber (DNp01, two
neurons) spikes for an expanding dark disc and stays silent for the three
controls, and whether the angular size at which it spikes is consistent across
r/v. This runs the full chain on the plan's suite (4 r/v x 8 azimuths x 4
stimulus kinds = 128 trials) and tabulates exactly that.

Chain, one place for each piece:

    loom.Loom + eye.render_loom     stimulus on the 721-column lattice
    optic_wrapper.OpticLobe         flyvis T4/T5, settled on each first frame
    optic_gate.Lplc2Drive           MaleCNS LPLC2 wiring over T4/T5 minus the
                                    LPi inhibitory path (gain 1)
    lif.LifNetwork                  495-neuron escape graph, Shiu-style LIF

Monocular, on purpose: the lab stimulus is shown to the right eye, so only the
91 right-hand LPLC2 neurons are driven; the left LPLC2 population is in the
graph and receives nothing. That is how the published single-side experiments
are run, and it is stated here because the giant fiber is bilateral.

`input_gain` is the one number set by a rule and not a fit: the most-driven LPLC2
neuron's peak drive at expanding r/v = 20 (median over azimuths) is scaled to
`TARGET_OVERDRIVE` times the steady drive that just reaches threshold. The rule
looks only at expanding trials, so it cannot be tuned to make a control quiet.
The real fit is Week 2; a sweep of the gain is reported here as sensitivity.

Usage (DGX, one GPU, about two minutes):
    CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/loom_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402
from eye import Eye, render_loom  # noqa: E402
from lif import EscapeGraph, LifParams, upsample_drive, with_params  # noqa: E402
from loom import RV_MS, STIMULUS_KINDS, Loom  # noqa: E402
from optic_gate import TYPES, Lplc2Drive  # noqa: E402

# The most-driven LPLC2 neuron reaches this multiple of the threshold current at
# expanding r/v = 20. 1.5 puts it clearly over threshold without saturating.
TARGET_OVERDRIVE = 1.5
GAIN_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
DURATION_MS = 400.0
OPTIC_DT_MS = 5.0


def build_trials(rvs=RV_MS):
    azimuths = tuple(float(a) for a in np.linspace(0, 360, 8, endpoint=False))
    return [
        Loom(kind=k, rv_ms=rv, azimuth_deg=az, duration_ms=DURATION_MS, dt_ms=OPTIC_DT_MS)
        for k in STIMULUS_KINDS
        for rv in rvs
        for az in azimuths
    ]


def angular_drive(looms, lplc2_npz: Path, rf_sigma_deg: float = 15.0) -> np.ndarray:
    """The plan's No-go fallback: LPLC2 driven directly by the stimulus's angular growth.

        drive_i(t) = relu(d theta / dt) * exp(-d_i^2 / (2 sigma^2))

    where d_i is the angle between the disc and neuron i's receptive-field centre
    (the centroid of its T4/T5 inputs on the eye plane). Growth rate, not size:
    a disc of constant size, translating or fading, drives nothing, and a
    receding one drives nothing. That is selective BY CONSTRUCTION, so a pass here
    says nothing about the optic lobe; it only says the LIF circuit and the
    connectome weights downstream behave sensibly. The UI has to say so.
    """
    from eye import LAB_ECCENTRICITY_DEG, column_plane_deg, flyvis_columns

    z = np.load(lplc2_npz)
    w = z["weights"].sum(axis=1)  # (n, 721)
    plane = column_plane_deg(flyvis_columns())
    centre = (w[:, :, None] * plane[None]).sum(axis=1) / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    out = np.zeros((len(looms), looms[0].n_steps, len(centre)), dtype=np.float32)
    for i, loom in enumerate(looms):
        rate = np.clip(np.gradient(loom.angular_size_deg(), loom.dt_ms), 0.0, None)  # deg/ms
        disc = LAB_ECCENTRICITY_DEG * np.array([np.cos(np.radians(loom.azimuth_deg)), np.sin(np.radians(loom.azimuth_deg))])
        if loom.kind == "translating":
            continue  # constant size: zero growth, drive is exactly zero whatever the position
        d = np.linalg.norm(centre - disc[None], axis=1)
        out[i] = rate[:, None] * np.exp(-(d**2) / (2 * rf_sigma_deg**2))[None]
    return out


def calibrate_input_gain(drive: np.ndarray, looms, params: LifParams) -> float:
    """Rule above. drive is (N, T, n_lplc2) in optic units."""
    peaks = [drive[i].max() for i, l in enumerate(looms) if l.kind == "expanding" and l.rv_ms == 20.0]
    gap = params.v_th - params.v0
    return TARGET_OVERDRIVE * gap / (params.tau * float(np.median(peaks)))


def trial_readout(spikes: np.ndarray, graph: EscapeGraph, params: LifParams, looms) -> list[dict]:
    """Per trial: giant-fiber first spike, angular size there, parallel-DN and LPLC2 counts."""
    gf = graph.index_of_type("DNp01")
    parallel = np.flatnonzero(np.isin(graph.type, ("DNp04", "DNp103", "DNp02", "DNp11")))
    lplc2 = graph.index_of_type("LPLC2")
    out = []
    for i, loom in enumerate(looms):
        gf_t = np.flatnonzero(spikes[i][:, gf].any(axis=1))
        row = {
            "kind": loom.kind,
            "rv_ms": loom.rv_ms,
            "azimuth_deg": loom.azimuth_deg,
            "gf_spikes": int(spikes[i][:, gf].sum()),
            "parallel_spikes": int(spikes[i][:, parallel].sum()),
            "lplc2_spikes": int(spikes[i][:, lplc2].sum()),
            "gf_t_ms": None,
            "gf_theta_deg": None,
            "gf_ttc_ms": None,
        }
        if gf_t.size:
            t = float(gf_t[0] * params.dt)
            theta = loom.angular_size_deg()
            row["gf_t_ms"] = t
            row["gf_theta_deg"] = float(np.interp(t, loom.time_ms(), theta))
            row["gf_ttc_ms"] = float(loom.t_c_ms - t)
        out.append(row)
    return out


def summarise(rows: list[dict]) -> dict:
    by_rv = []
    for rv in sorted({r["rv_ms"] for r in rows}):
        entry = {"rv_ms": rv}
        for kind in STIMULUS_KINDS:
            sel = [r for r in rows if r["kind"] == kind and r["rv_ms"] == rv]
            entry[f"{kind}_gf_fraction"] = float(np.mean([r["gf_spikes"] > 0 for r in sel]))
            entry[f"{kind}_gf_spikes_median"] = float(np.median([r["gf_spikes"] for r in sel]))
        exp = [r for r in rows if r["kind"] == "expanding" and r["rv_ms"] == rv and r["gf_t_ms"] is not None]
        entry["expanding_gf_t_ms_median"] = float(np.median([r["gf_t_ms"] for r in exp])) if exp else None
        entry["expanding_gf_theta_deg_median"] = float(np.median([r["gf_theta_deg"] for r in exp])) if exp else None
        entry["expanding_gf_ttc_ms_median"] = float(np.median([r["gf_ttc_ms"] for r in exp])) if exp else None
        by_rv.append(entry)
    thetas = [e["expanding_gf_theta_deg_median"] for e in by_rv if e[f"expanding_gf_fraction"] >= 0.5]
    crit1 = {
        k: float(np.mean([r["gf_spikes"] > 0 for r in rows if r["kind"] == k])) for k in STIMULUS_KINDS
    }
    return {
        "by_rv": by_rv,
        "criterion1_gf_fraction_by_kind": crit1,
        # criterion 2: "GF spike at a consistent angular size"; reported as the spread of the median angle
        "criterion2_theta_deg_by_rv": [e["expanding_gf_theta_deg_median"] for e in by_rv],
        "criterion2_theta_spread_ratio": float(max(thetas) / min(thetas)) if len(thetas) >= 2 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--graph", type=Path, default=Path("data/ol/escape_graph.npz"))
    ap.add_argument("--lplc2", type=Path, default=Path("data/ol/lplc2_inputs_R.npz"))
    ap.add_argument("--readout", default="lpi:1", help="LPLC2 drive readout from optic_gate (default: LPi gain 1)")
    ap.add_argument("--drive", choices=("optic", "angular"), default="optic",
                    help="optic: through the optic lobe. angular: the plan's No-go fallback, from stimulus geometry")
    ap.add_argument("--save", type=Path, default=Path("docs/loom_sweep.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/ol/sweep_drive.npz"),
                    help="optic drive cache: LIF-only experiments reuse it instead of rerunning the optic lobe")
    ap.add_argument("--refresh", action="store_true", help="recompute the optic drive even if cached")
    ap.add_argument("--cpu", action="store_true", help="allow a CPU run (slow, and takes the shared host's cores)")
    args = ap.parse_args()
    require_remote_execution("loom_sweep.py", "128 trials through the optic lobe and LIF on a GPU")
    import torch

    if not torch.cuda.is_available() and not args.cpu:
        raise SystemExit("no GPU visible: set CUDA_VISIBLE_DEVICES to a free GPU (nvidia-smi), or pass --cpu\n"
                         "(an unplanned CPU fallback took six minutes and most of a shared host's cores)")
    if args.cpu:
        torch.set_num_threads(8)

    from optic_wrapper import OpticLobe

    t0 = time.time()
    looms = build_trials()
    key = f"member{args.member}_{args.readout}"
    if args.drive == "angular":
        drive = angular_drive(looms, args.lplc2)
        args.save = args.save.with_name(args.save.stem + "_angular.json")
        banner("LPLC2 driven from stimulus angular growth (No-go fallback): the optic lobe is NOT in this loop")
    elif args.cache.exists() and not args.refresh and str(np.load(args.cache)["key"]) == key:
        drive = np.load(args.cache)["drive"]
        banner(f"optic drive from cache {args.cache} ({key})")
    else:
        eye = Eye.flyvis()
        lobe = OpticLobe(member=args.member, dt_s=OPTIC_DT_MS / 1000.0)
        movies = np.stack([render_loom(l, eye) for l in looms])
        act = lobe.run(movies, record=TYPES, background="first_frame", chunk=32)
        drive = Lplc2Drive(args.lplc2)(act, args.readout)  # (N, T, 91), optic units
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, drive=drive, key=key)
        banner(f"optic lobe done ({time.time() - t0:.0f}s): {len(looms)} trials, readout {args.readout}")

    graph = EscapeGraph.load(args.graph)
    base = LifParams()
    gain0 = calibrate_input_gain(drive, looms, base)
    print(f"input gain by rule (most-driven LPLC2 at expanding r/v=20 -> {TARGET_OVERDRIVE}x threshold): {gain0:.5f} mV/ms per unit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lif_drive = graph.embed_drive(upsample_drive(drive, OPTIC_DT_MS, base.dt, DURATION_MS), np.load(args.lplc2)["bodies"])

    result = {"member": args.member, "readout": args.readout if args.drive == "optic" else "angular", "drive": args.drive, "target_overdrive": TARGET_OVERDRIVE,
              "input_gain_by_rule": gain0, "by_gain_scale": {}}
    for scale in GAIN_SCALES:
        params = with_params(base, input_gain=gain0 * scale)
        net = graph.network(params, device)
        spikes = np.concatenate(
            [net.run(lif_drive[i : i + 32] * params.input_gain)["spikes"] for i in range(0, len(looms), 32)]
        )
        rows = trial_readout(spikes, graph, params, looms)
        result["by_gain_scale"][str(scale)] = summarise(rows)
        if scale == 1.0:
            result["trials"] = rows
        print(f"gain x{scale}: GF spike fraction " + "  ".join(
            f"{k[:5]} {v:.2f}" for k, v in result["by_gain_scale"][str(scale)]["criterion1_gf_fraction_by_kind"].items()), flush=True)

    s = result["by_gain_scale"]["1.0"]
    banner("criterion 1 and 2 at the rule-set gain: fraction of trials with a GF spike; angle at the first spike")
    print(f"{'r/v':>5} | " + " ".join(f"{k[:5]:>9}" for k in STIMULUS_KINDS) + " | exp t(ms)  theta(deg)  ttc(ms)")
    for e in s["by_rv"]:
        cells = " ".join(f"{e[f'{k}_gf_fraction']:9.2f}" for k in STIMULUS_KINDS)
        f = lambda x: "   -  " if x is None else f"{x:6.1f}"
        print(f"{e['rv_ms']:5.0f} | {cells} | {f(e['expanding_gf_t_ms_median'])}   {f(e['expanding_gf_theta_deg_median'])}   {f(e['expanding_gf_ttc_ms_median'])}")
    print("theta spread ratio (max/min of median angle across r/v, want ~1):", s["criterion2_theta_spread_ratio"])

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {args.save}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
