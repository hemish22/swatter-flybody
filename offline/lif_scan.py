"""Best case for criterion 1: scan the three fitted LIF parameters on the cached optic drive.

Week 2 fits `input_gain`, `weight_scale` and `v_th` once and freezes them. This
asks the cheaper question first: is there ANY setting of them under which the
giant fiber spikes for expanding discs and stays quiet for the controls? If the
best margin over a coarse grid is already poor, fitting cannot rescue it and the
gate verdict does not depend on the fit.

Margin = (fraction of expanding trials with a GF spike) - (the largest such
fraction among receding, translating and dimming). 1 is perfect, 0 is no
separation, negative means a control fires more than the loom.

Needs `data/ol/sweep_drive.npz` from loom_sweep.py. One GPU, a few minutes.
    CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/lif_scan.py
    ... offline/lif_scan.py --angular     # the No-go fallback drive instead (no cache needed)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402
from lif import EscapeGraph, LifParams, upsample_drive, with_params  # noqa: E402
from loom import STIMULUS_KINDS  # noqa: E402
from loom_sweep import DURATION_MS, OPTIC_DT_MS, angular_drive, build_trials, calibrate_input_gain, trial_readout  # noqa: E402

GAIN_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
WEIGHT_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
ANGULAR = "--angular" in sys.argv
if ANGULAR:  # the angular drive is weaker where the GF needs many LPLC2 spikes at once; scan higher
    GAIN_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
    WEIGHT_SCALES = (1.0, 2.0, 4.0, 8.0)
THRESHOLDS = (-47.0, -45.0, -43.0)


def main() -> int:
    require_remote_execution("lif_scan.py", "75 LIF runs of 128 trials on a GPU")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: set CUDA_VISIBLE_DEVICES to a free GPU")
    looms = build_trials()
    lplc2_npz = Path("data/ol/lplc2_inputs_R.npz")
    drive = angular_drive(looms, lplc2_npz) if ANGULAR else np.load("data/ol/sweep_drive.npz")["drive"]
    graph = EscapeGraph.load(Path("data/ol/escape_graph.npz"))
    base = LifParams()
    gain0 = calibrate_input_gain(drive, looms, base)
    lplc2_bodies = np.load(lplc2_npz)["bodies"]
    lif_drive = graph.embed_drive(upsample_drive(drive, OPTIC_DT_MS, base.dt, DURATION_MS), lplc2_bodies)

    rows = []
    for gs, ws, vth in itertools.product(GAIN_SCALES, WEIGHT_SCALES, THRESHOLDS):
        p = with_params(base, input_gain=gain0 * gs, weight_scale=ws, v_th=vth)
        net = graph.network(p, "cuda")
        spikes = np.concatenate([net.run(lif_drive[i : i + 32] * p.input_gain)["spikes"] for i in range(0, len(looms), 32)])
        t = trial_readout(spikes, graph, p, looms)
        frac = {k: float(np.mean([r["gf_spikes"] > 0 for r in t if r["kind"] == k])) for k in STIMULUS_KINDS}
        # also per r/v, so a setting that wins by firing only for the slowest loom is visible
        exp_by_rv = {rv: float(np.mean([r["gf_spikes"] > 0 for r in t if r["kind"] == "expanding" and r["rv_ms"] == rv]))
                     for rv in sorted({l.rv_ms for l in looms})}
        theta_by_rv = {rv: (float(np.median([r["gf_theta_deg"] for r in t if r["kind"] == "expanding" and r["rv_ms"] == rv and r["gf_theta_deg"] is not None]))
                            if any(r["gf_theta_deg"] is not None for r in t if r["kind"] == "expanding" and r["rv_ms"] == rv) else None)
                       for rv in sorted({l.rv_ms for l in looms})}
        margin = frac["expanding"] - max(frac[k] for k in STIMULUS_KINDS if k != "expanding")
        rows.append({"gain_scale": gs, "weight_scale": ws, "v_th": vth, "fraction": frac,
                     "expanding_by_rv": exp_by_rv, "theta_by_rv": theta_by_rv, "margin": margin})
        print(f"gain x{gs:<4} w x{ws:<4} vth {vth:5.0f}  exp {frac['expanding']:.2f} rec {frac['receding']:.2f} "
              f"tra {frac['translating']:.2f} dim {frac['dimming']:.2f}  margin {margin:+.2f}", flush=True)

    rows.sort(key=lambda r: r["margin"], reverse=True)
    banner("best five by margin")
    for r in rows[:5]:
        print(f"gain x{r['gain_scale']} w x{r['weight_scale']} vth {r['v_th']:.0f}: margin {r['margin']:+.2f}  "
              f"expanding by r/v {r['expanding_by_rv']}  theta {({k: None if v is None else round(v, 1) for k, v in r['theta_by_rv'].items()})}  controls {({k: round(v, 2) for k, v in r['fraction'].items() if k != 'expanding'})}")
    out = Path("docs/lif_scan_angular.json" if ANGULAR else "docs/lif_scan.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
