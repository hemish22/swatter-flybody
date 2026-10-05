"""Week 1 gate, optic-lobe half: does the optic lobe carry a looming signal?

The go / partial / no-go table in docs/build-plan.md turns on one question:
after the optic lobe, is there something that fires for an expanding dark disc
and stays quiet for a receding one, a dimming one and a translating one? The
point-neuron LPLC2 of the LIF stage is not built yet, so this script asks the
question one stage earlier:

    eye (721 columns, loom rendered on the real lattice)
        -> flyvis optic lobe, settled on each trial's first frame
        -> T4/T5 activity
        -> MaleCNS LPLC2 wiring (synapse counts per T4/T5 type and column)
        -> linear drive of each of the 91 LPLC2 neurons

so the readout is the connectome's own receptive field, not a hand-drawn one.
An earlier version used a hand-built radial-motion template over T4/T5 (the
plan's "partial" fallback); it never separated translating from expanding and
is kept only as logs/optic_gate_v*.log. What it did teach, and what is built in
here as a result:

  - Trials are settled on their own first frame, not on the bare sky. Settling
    on the sky makes every disc already on screen at t=0 arrive as an onset
    flash, which scored dimming and translating above expanding for reasons that
    had nothing to do with looming.
  - T4/T5 preferred directions are measured, not assumed (`--calibrate` output
    in the log). The renderer's lattice orientation reproduces flyvis's own
    moving-edge protocol to within ~2 degrees for T4a/b, T5a-d; T4c sits 30
    degrees off nominal and T4d is barely direction-selective (DSI 0.14) in
    ensemble member 000 under flyvis's own protocol too, so that is the network
    and not the renderer.

Usage (DGX; takes a GPU for about a minute):
    CUDA_VISIBLE_DEVICES=7 SWATTER_REMOTE=1 .venv/bin/python offline/optic_gate.py
    ... offline/optic_gate.py --dt-check     # also T4/T5 vs integration step
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402
from eye import Eye, column_plane_deg, edge_movie, render_loom  # noqa: E402
from loom import RV_MS, STIMULUS_KINDS, Loom  # noqa: E402

TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
# T4 sees ON edges, T5 OFF edges.
POLARITY = {t: "on" if t.startswith("T4") else "off" for t in TYPES}

# Readouts, in the order they were added. "lpi:1" is the one the verdict is read at.
MODES = ("linear", "and", "lpi:1", "lpi:0.5", "lpi:2", "lpi:4")
TRANSLATION_SPEEDS_DEG_S = (25.0, 50.0, 100.0, 200.0)  # 200 is loom.py's control

CALIBRATION_PHIS = np.arange(0, 360, 45)
CALIBRATION_SPEED_DEG_S = 150.0


# --------------------------------------------------------------------------- #
# calibration: where does each T4/T5 type actually point?
# --------------------------------------------------------------------------- #


def calibrate_preferred_directions(lobe, eye: Eye) -> dict[str, dict[str, float]]:
    """Preferred direction (degrees, in the eye's tangent plane) and tuning depth per type."""
    out: dict[str, dict[str, float]] = {}
    for pol in ("on", "off"):
        movies = np.stack([edge_movie(eye, p, CALIBRATION_SPEED_DEG_S, lobe.dt_s, 1.2, pol) for p in CALIBRATION_PHIS])
        act = lobe.run(movies, record=TYPES, background=0.5)
        for t in TYPES:
            if POLARITY[t] != pol:
                continue
            a = act[t]
            peak = np.nanmean((a - a[:, :1]).max(axis=1), axis=1)  # (phis,)
            w = np.clip(peak - peak.min(), 0.0, None)
            vec = (w * np.exp(1j * np.radians(CALIBRATION_PHIS))).sum()
            out[t] = {
                "pd_deg": float(np.degrees(np.angle(vec)) % 360.0),
                # length of the vector sum over the total: 0 = untuned, 1 = one direction only
                "tuning": float(np.abs(vec) / max(w.sum(), 1e-9)),
                "peak": float(peak.max()),
            }
    return out


# --------------------------------------------------------------------------- #
# the readout: MaleCNS LPLC2 wiring over flyvis T4/T5
# --------------------------------------------------------------------------- #


class Lplc2Drive:
    """Linear drive of each MaleCNS LPLC2 neuron from the optic lobe's T4/T5.

    W[i, k, c] is the number of synapses LPLC2 neuron i receives from T4/T5 type
    k in lattice column c (`malecns_ol.py`). Drive is the synapse-weighted sum
    of each input's rise above its pre-stimulus level, rectified because a cell
    cannot pass on less than none:

        drive_i(t) = sum_{k,c} W[i,k,c] * relu(a_kc(t) - a_kc(0))

    Everything is excitatory (T4/T5 are cholinergic) and there is no threshold,
    no lobula-plate inhibition and no spiking: this is the wiring on its own,
    the lower bound on LPLC2's selectivity. It answers "does the connectome
    receptive field, fed by the optic lobe, already prefer expansion?" before
    the LIF stage adds its nonlinearity. Scale is arbitrary (raw synapse
    counts), so only ratios between conditions mean anything.
    """

    def __init__(self, npz: Path):
        z = np.load(npz)
        if tuple(z["types"]) != TYPES:
            raise ValueError(f"type order in {npz} is {tuple(z['types'])}, expected {TYPES}")
        self.weights = z["weights"].astype(np.float32)  # (n_lplc2, 8, 721)
        self.bodies = z["bodies"]
        self.has_lpi = "lpi_t4t5" in z
        if self.has_lpi:
            self.lpi_in = z["lpi_t4t5"].astype(np.float32)  # (n_lpi, 8, 721)
            self.lpi_total = np.maximum(z["lpi_t4t5_total"].astype(np.float32), 1.0)
            self.lpi_out = z["lpi_to_lplc2"].astype(np.float32)  # (n_lplc2, n_lpi)

    def groups(self, act: dict[str, np.ndarray]) -> np.ndarray:
        """(B, T, n_lplc2, 4) drive per direction group a-d, T4 and T5 of a group pooled."""
        rise = np.stack([np.clip(act[t] - act[t][:, :1], 0.0, None) for t in TYPES], axis=2)  # (B, T, 8, 721)
        per_type = np.einsum("btkc,ikc->btik", rise, self.weights)  # (B, T, n, 8)
        return per_type[..., :4] + per_type[..., 4:]

    def __call__(self, act: dict[str, np.ndarray], mode: str = "linear") -> np.ndarray:
        """act[type] is (B, T, 721). Returns (B, T, n_lplc2).

        mode "linear": the sum above. mode "and": the geometric mean of the four
        direction groups' drives, so a neuron whose receptive field gets no
        motion in one direction channel is silent however hard the other three
        are driven. That is the coincidence LPLC2 needs from a looming object
        (outward motion on every side) and a translating one cannot supply.
        It is a second readout of the same wiring, not a different wiring.

        mode "lpi:<gain>": the linear drive minus the inhibitory two-hop path.
        Each LPi cell carries the mean rise of its T4/T5 input (its synapse-
        weighted rise over its TOTAL T4/T5 input, so inputs outside the lattice
        count as silent), and each of its synapses onto an LPLC2 neuron subtracts
        that, times `gain`, in the same synapse x rise units as the excitation.
        Gain 1 means an inhibitory synapse is worth an excitatory one. It is not
        fitted: a per-synapse efficacy for glutamate against acetylcholine is not
        in the connectome, so the gain is reported as a sweep and the verdict is
        read at gain 1.
        """
        if mode == "linear":
            return self.groups(act).sum(axis=-1)
        if mode == "and":
            return np.prod(np.clip(self.groups(act), 0.0, None), axis=-1) ** 0.25
        if mode.startswith("lpi:"):
            gain = float(mode.split(":")[1])
            rise = np.stack([np.clip(act[t] - act[t][:, :1], 0.0, None) for t in TYPES], axis=2)
            exc = np.einsum("btkc,ikc->bti", rise, self.weights)
            lpi_act = np.einsum("btkc,jkc->btj", rise, self.lpi_in) / self.lpi_total
            inh = np.einsum("btj,ij->bti", lpi_act, self.lpi_out)
            return np.clip(exc - gain * inh, 0.0, None)
        raise ValueError(mode)


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #


def run_suite(lobe, eye: Eye, readout: Lplc2Drive, rvs=RV_MS, azimuths=None, duration_ms=400.0):
    """Every (kind, r/v, azimuth) trial through eye -> optic lobe -> LPLC2 drive."""
    azimuths = azimuths if azimuths is not None else tuple(float(a) for a in np.linspace(0, 360, 8, endpoint=False))
    dt_ms = lobe.dt_s * 1000.0
    looms = [
        Loom(kind=k, rv_ms=rv, azimuth_deg=az, duration_ms=duration_ms, dt_ms=dt_ms)
        for k in STIMULUS_KINDS
        for rv in rvs
        for az in azimuths
    ]
    movies = np.stack([render_loom(l, eye) for l in looms])
    act = lobe.run(movies, record=TYPES, background="first_frame", chunk=32)
    # most-driven LPLC2 neuron at each time, (N, T), per readout mode
    peaks = {m: readout(act, m).max(axis=2) for m in MODES if readout.has_lpi or not m.startswith("lpi")}
    return looms, peaks


def summarise(looms, peak, dt_ms: float) -> dict:
    """Selectivity and size/time-at-threshold, one table per r/v."""
    # One absolute threshold for every condition and r/v: half the median
    # expanding peak at the middle r/v. Chosen from the expanding trials only
    # so it cannot be tuned to make the controls look quiet.
    exp_mid = [p.max() for l, p in zip(looms, peak) if l.kind == "expanding" and l.rv_ms == 20.0]
    threshold = 0.5 * float(np.median(exp_mid))
    rows = []
    for rv in sorted({l.rv_ms for l in looms}):
        row = {"rv_ms": rv}
        for kind in STIMULUS_KINDS:
            sel = [(l, p) for l, p in zip(looms, peak) if l.kind == kind and l.rv_ms == rv]
            peaks = np.array([p.max() for _, p in sel])
            row[f"{kind}_peak_median"] = float(np.median(peaks))
            row[f"{kind}_fraction_above_threshold"] = float(np.mean(peaks > threshold))
        ctrl = max(row[f"{k}_peak_median"] for k in STIMULUS_KINDS if k != "expanding")
        row["selectivity_expanding_over_best_control"] = float(row["expanding_peak_median"] / max(ctrl, 1e-12))
        # criterion 2 precursor: when and at what angular size does the readout cross threshold
        crossings, sizes = [], []
        for l, p in [(l, p) for l, p in zip(looms, peak) if l.kind == "expanding" and l.rv_ms == rv]:
            above = np.flatnonzero(p > threshold)
            if above.size:
                crossings.append(float(l.time_ms()[above[0]]))
                sizes.append(float(l.angular_size_deg()[above[0]]))
        row["threshold_crossing_ms_median"] = float(np.median(crossings)) if crossings else None
        row["angular_size_at_crossing_deg_median"] = float(np.median(sizes)) if sizes else None
        row["trials_crossing"] = f"{len(crossings)}/{len([1 for l in looms if l.kind == 'expanding' and l.rv_ms == rv])}"
        rows.append(row)
    return {"threshold": threshold, "by_rv": rows}


def translation_sweep(lobe, eye: Eye, readout: Lplc2Drive, modes, duration_ms=400.0) -> dict:
    """Median peak drive of a translating disc at several speeds, per readout.

    Diagnostic only. loom.py's control (200 deg/s) stays the pass/fail one; this
    shows how far down in speed the translating disc has to go before it stops
    out-driving the loom, so the choice of control speed can be judged with the
    number in front of you instead of being tuned to the table.
    """
    azimuths = tuple(float(a) for a in np.linspace(0, 360, 8, endpoint=False))
    dt_ms = lobe.dt_s * 1000.0
    out = {}
    for speed in TRANSLATION_SPEEDS_DEG_S:
        arc = speed * duration_ms / 1000.0 / 2.0  # half the total sweep
        looms = [Loom(kind="translating", rv_ms=20.0, azimuth_deg=az, duration_ms=duration_ms, dt_ms=dt_ms) for az in azimuths]
        movies = np.stack([render_loom(l, eye, translate_arc_deg=arc) for l in looms])
        act = lobe.run(movies, record=TYPES, background="first_frame")
        out[speed] = {m: float(np.median(readout(act, m).max(axis=2).max(axis=1))) for m in modes}
    return out


def dt_check(lobe_cls, eye: Eye, member: int) -> dict:
    """How much do T4/T5 responses to one expanding loom move with the integration step?"""
    out = {}
    ref = None
    for dt_s in (0.005, 0.010, 0.020):
        lobe = lobe_cls(member=member, dt_s=dt_s)
        loom = Loom(kind="expanding", rv_ms=20.0, azimuth_deg=0.0, duration_ms=400.0, dt_ms=dt_s * 1000.0)
        act = lobe.run(render_loom(loom, eye)[None], record=TYPES, background="first_frame")
        # sample every 20 ms so the three steps land on the same grid
        stride = int(round(0.020 / dt_s))
        sub = {t: act[t][0, ::stride] for t in TYPES}
        out[dt_s] = sub
        if ref is None:
            ref = sub
    rep = {}
    for dt_s, sub in out.items():
        n = min(sub[TYPES[0]].shape[0], ref[TYPES[0]].shape[0])
        num = sum(float(np.nansum((sub[t][:n] - ref[t][:n]) ** 2)) for t in TYPES)
        den = sum(float(np.nansum((ref[t][:n] - ref[t][:1]) ** 2)) for t in TYPES)
        rep[f"{dt_s * 1000:.0f}ms_vs_5ms_relative_rms"] = math.sqrt(num / max(den, 1e-12))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", type=int, default=0, help="flyvis ensemble member (0 = best by validation loss)")
    ap.add_argument("--dt-ms", type=float, default=5.0)
    ap.add_argument("--dt-check", action="store_true")
    ap.add_argument("--lplc2", type=Path, default=Path("data/ol/lplc2_inputs_R.npz"),
                    help="LPLC2 <- T4/T5 synapse counts from malecns_ol.py")
    ap.add_argument("--save", type=Path, default=Path("docs/optic_gate.json"))
    args = ap.parse_args()

    require_remote_execution("optic_gate.py", "runs 128 trials through the 45k-neuron optic lobe on a GPU")
    from optic_wrapper import OpticLobe

    t0 = time.time()
    eye = Eye.flyvis()
    lobe = OpticLobe(member=args.member, dt_s=args.dt_ms / 1000.0)
    banner(f"optic lobe: {lobe.backend} member {args.member:03d}, dt {args.dt_ms} ms, {lobe.n_cells:,} cells on {lobe.device}")

    pds = calibrate_preferred_directions(lobe, eye)
    print("measured preferred directions (eye-plane degrees; nominal flyvis in brackets)")
    nominal = {"T4a": 180, "T4b": 0, "T4c": 90, "T4d": 270, "T5a": 180, "T5b": 0, "T5c": 90, "T5d": 270}
    for t in TYPES:
        print(f"  {t}  pd {pds[t]['pd_deg']:6.1f}  [{nominal[t]:3d}]  tuning {pds[t]['tuning']:.2f}  peak {pds[t]['peak']:.2f}")

    readout = Lplc2Drive(args.lplc2)
    looms, peaks = run_suite(lobe, eye, readout)
    result = {m: summarise(looms, peak, args.dt_ms) for m, peak in peaks.items()}
    result["preferred_directions"] = pds
    result["member"] = args.member
    result["dt_ms"] = args.dt_ms

    for mode, res in ((m, result[m]) for m in peaks):
        banner(f"criterion 1 precursor, LPLC2 readout '{mode}' (threshold {res['threshold']:.3g}, set from expanding r/v=20 only)")
        print(f"{'r/v':>5} | " + " ".join(f"{k[:5]:>9}" for k in STIMULUS_KINDS) + " | sel(exp/best ctrl) | crossing: ms, deg, trials")
        for r in res["by_rv"]:
            cells = " ".join(f"{r[f'{k}_peak_median']:9.3g}" for k in STIMULUS_KINDS)
            ms, deg = r["threshold_crossing_ms_median"], r["angular_size_at_crossing_deg_median"]
            print(f"{r['rv_ms']:5.0f} | {cells} | {r['selectivity_expanding_over_best_control']:18.2f} | "
                  f"{ms}, {None if deg is None else round(deg, 1)}, {r['trials_crossing']}")

    result["translation_speed"] = translation_sweep(lobe, eye, readout, list(peaks))
    banner("diagnostic: translating-disc peak drive vs speed (loom.py's control is 200 deg/s); expanding for reference")
    print(f"{'deg/s':>6} | " + " ".join(f"{m:>9}" for m in peaks))
    for speed, row in result["translation_speed"].items():
        print(f"{speed:6.0f} | " + " ".join(f"{row[m]:9.3g}" for m in peaks))
    for rv in (20.0, 40.0):
        ref = next(r for r in result["linear"]["by_rv"] if r["rv_ms"] == rv)
        print(f"expand r/v={rv:<4.0f}| " + " ".join(f"{next(r for r in result[m]['by_rv'] if r['rv_ms'] == rv)['expanding_peak_median']:9.3g}" for m in peaks))

    if args.dt_check:
        result["dt_check"] = dt_check(OpticLobe, eye, args.member)
        banner("integration step: T4/T5 response to one expanding loom, relative RMS vs the 5 ms run")
        for k, v in result["dt_check"].items():
            print(f"  {k}: {v:.3f}")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {args.save}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
