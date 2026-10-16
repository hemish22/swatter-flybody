"""Cross-check of the LIF engine against the Brian2 model it is modelled on (the plan's "LIF cross-check" row).

The reference is the model code of Shiu et al. 2024, `model.py` in github.com/philshiu/Drosophila_brain_model (read
2026-10-16, key lines quoted below). We feed the same 495-neuron graph, the same weights and the same per-step LPLC2
drive as the engine, through three Brian2 variants, and compare the outcome of each of the 50 parity trials with the
one `web/brain/parity.json` records for the engine:

  shiu_dt0.1         Shiu's equations and reset exactly, method='linear', Brian's default 0.1 ms step.
  shiu_dt0.2         the same at the engine's 0.2 ms step: separates the step size from the equations.
  ours_euler_dt0.2   the engine's own scheme written in Brian2 (forward Euler, g keeps decaying in the refractory
                     period, g not reset on a spike): shows how far the harness itself is from the engine.

Shiu's lines that matter (verbatim from the file):
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    reset  'v = v_rst; w = 0; g = 0 * mV'          method='linear', refractory='rfc', synapses 'g += w', delay t_dly
Differences from `lif.py`, all of them in what a neuron does around its own spike:
  1. g is frozen while the neuron is refractory (here it decays and keeps integrating);
  2. g is set to 0 on a spike (here it carries on);
  3. exact (linear) integration at 0.1 ms (here Euler at 0.2 ms).
The external drive is injected into g as a jump of `x * dt` every 0.2 ms, the way the engine does it; Shiu drive their
inputs with Poisson jumps on v, which is a different input, not a different neuron.

Run on the DGX (it is light, but the repo rule is that jobs run there):
    tmux new -s swatter-b2 'cd ~/swatter-flybody && .venv-b2/bin/python offline/brian2_check.py 2>&1 | tee logs/brian2_check.log'
Needs web/brain (brain.bin, manifest.json, parity.json) and `pip install brian2`. Writes docs/brian2_check.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline"))

from engine_ref import Brain, Engine, outcome, run_trial  # noqa: E402

VARIANTS = {
    "shiu_dt0.1": dict(dt=0.1, method="linear", freeze=True, reset_g=True),
    "shiu_dt0.2": dict(dt=0.2, method="linear", freeze=True, reset_g=True),
    "ours_euler_dt0.2": dict(dt=0.2, method="euler", freeze=False, reset_g=False),
}
STEP_MS = 0.2  # the engine's step, and the grid the drive and the parity outcomes live on


def drive_steps(brain: Brain, frames: np.ndarray) -> np.ndarray:
    """(steps, n) external input in mV/ms at every 0.2 ms step, built exactly the way Engine.frame/finish does."""
    e = Engine(brain)
    out: list[np.ndarray] = []
    prev = None
    for fr in frames:
        cur = e._drive(fr)
        if prev is not None:
            for i in range(25):
                f = np.float32(i / 25)
                out.append(e._x(prev * (1 - f) + cur * f))
        prev = cur
    out.append(e._x(prev))
    return np.stack(out)


def build(brain: Brain, cfg: dict):
    from brian2 import NeuronGroup, Network, SpikeMonitor, Synapses, ms, mV, network_operation, start_scope

    start_scope()
    m = brain.manifest
    p = m["lif"]
    n = m["neurons"]
    a = brain.arrays
    ns = dict(v_0=p["v0"] * mV, v_rst=p["v_rst"] * mV, v_th=p["v_th"] * mV, t_mbr=p["t_mbr"] * ms, tau=p["tau"] * ms)
    unless = " (unless refractory)"
    eqs = f"""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau : volt{unless if cfg['freeze'] else ''}
        rfc : second
    """
    reset = "v = v_rst; g = 0 * mV" if cfg["reset_g"] else "v = v_rst"
    neu = NeuronGroup(n, eqs, method=cfg["method"], threshold="v > v_th", reset=reset, refractory="rfc", namespace=ns, dt=cfg["dt"] * ms)
    neu.v = ns["v_0"]
    neu.g = 0 * mV
    neu.rfc = p["t_rfc"] * ms
    pre = np.repeat(np.arange(n), np.diff(a["csr_indptr"])).astype(np.int64)  # Brian wants int64 index arrays, the file stores uint16
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p["t_dly"] * ms, dt=cfg["dt"] * ms)
    syn.connect(i=pre, j=a["csr_post"].astype(np.int64))
    syn.w = a["csr_weight_mv"].astype(np.float64) * mV
    mon = SpikeMonitor(neu)
    state = {"drive": np.zeros((0, n))}

    @network_operation(dt=STEP_MS * ms, when="start")
    def inject(t):
        k = int(round(float(t / ms) / STEP_MS))
        if k < len(state["drive"]):
            neu.g_[:] = neu.g_[:] + state["drive"][k] * STEP_MS * 1e-3  # mV/ms * ms = mV, in volts
    net = Network(neu, syn, mon, inject)
    net.store("init")
    return net, mon, state


def run_variant(name: str, cfg: dict, brain: Brain, par: dict, ref_spike_counts: list[int]) -> dict:
    from brian2 import ms

    net, mon, state = build(brain, cfg)
    window = brain.manifest["roles"]["short_window_ms"]
    rows = []
    t0 = time.time()
    for tr in par["trials"]:
        net.restore("init")
        frames = np.array(tr["frames"])
        state["drive"] = drive_steps(brain, frames)
        net.run(len(state["drive"]) * STEP_MS * ms)
        spikes = [(int(round(float(t / ms) / STEP_MS)), int(i)) for i, t in zip(mon.i, mon.t)]
        eng = SimpleNamespace(spikes=spikes, dt=STEP_MS)
        got = outcome(brain, eng, window)
        exp = tr["expected"]
        rows.append(dict(
            id=tr["id"], kind=tr["kind"], rv_ms=tr["rv_ms"], azimuth_deg=tr["azimuth_deg"],
            gf_exp=exp["gf_first_ms"], gf_got=got["gf_first_ms"], par_exp=exp["parallel_first_ms"], par_got=got["parallel_first_ms"],
            mode_exp=exp["mode"], mode_got=got["mode"], side_exp=exp["heading_side"], side_got=got["heading_side"],
            spikes=len(spikes), spikes_engine=ref_spike_counts[tr["id"]],
        ))
    both = [r for r in rows if r["gf_exp"] is not None and r["gf_got"] is not None]
    d = np.array([r["gf_got"] - r["gf_exp"] for r in both])
    dp = np.array([r["par_got"] - r["par_exp"] for r in rows if r["par_exp"] is not None and r["par_got"] is not None])
    summary = dict(
        variant=name, seconds=round(time.time() - t0, 1), trials=len(rows),
        gf_fires_engine=sum(r["gf_exp"] is not None for r in rows), gf_fires_brian=sum(r["gf_got"] is not None for r in rows),
        gf_fire_mismatch=sum((r["gf_exp"] is None) != (r["gf_got"] is None) for r in rows),
        gf_delta_ms=dict(n=int(d.size), median=float(np.median(d)) if d.size else None, p5=float(np.percentile(d, 5)) if d.size else None,
                         p95=float(np.percentile(d, 95)) if d.size else None, max_abs=float(np.abs(d).max()) if d.size else None,
                         within_one_step=int((np.abs(d) <= STEP_MS + 1e-9).sum())),
        parallel_delta_ms=dict(n=int(dp.size), median=float(np.median(dp)) if dp.size else None, max_abs=float(np.abs(dp).max()) if dp.size else None),
        mode_agree=sum(r["mode_exp"] == r["mode_got"] for r in rows), side_agree=sum(r["side_exp"] == r["side_got"] for r in rows),
        spike_ratio_median=float(np.median([r["spikes"] / max(1, r["spikes_engine"]) for r in rows])),
    )
    return {"summary": summary, "trials": rows}


def main() -> None:
    brain = Brain.load(ROOT / "web" / "brain")
    par = json.loads((ROOT / "web" / "brain" / "parity.json").read_text())
    ref_counts = [len(run_trial(brain, np.array(tr["frames"])).spikes) for tr in par["trials"]]
    import brian2
    from brian2 import prefs
    prefs.codegen.target = "numpy"  # no compiler needed; the graph is small
    print(f"brian2 {brian2.__version__}, {len(par['trials'])} parity trials, engine spikes/trial median {int(np.median(ref_counts))}", flush=True)
    result = {"brian2": brian2.__version__, "variants": {}}
    for name, cfg in VARIANTS.items():
        r = run_variant(name, cfg, brain, par, ref_counts)
        result["variants"][name] = r
        s = r["summary"]
        print(f"\n{name}  ({s['seconds']} s)", flush=True)
        print(f"  GF fires: engine {s['gf_fires_engine']}, brian {s['gf_fires_brian']}, mismatch {s['gf_fire_mismatch']}", flush=True)
        g = s["gf_delta_ms"]
        print(f"  GF first spike, brian minus engine (ms): n {g['n']}, median {g['median']}, 5-95% [{g['p5']}, {g['p95']}], max |d| {g['max_abs']}, within one step {g['within_one_step']}", flush=True)
        print(f"  parallel first spike: median {s['parallel_delta_ms']['median']}, max |d| {s['parallel_delta_ms']['max_abs']}", flush=True)
        print(f"  mode agree {s['mode_agree']}/{s['trials']}, heading side agree {s['side_agree']}/{s['trials']}, spikes per trial vs engine (median ratio) {s['spike_ratio_median']:.2f}", flush=True)
    (ROOT / "docs" / "brian2_check.json").write_text(json.dumps(result, indent=1))
    print("\nwrote docs/brian2_check.json", flush=True)


if __name__ == "__main__":
    main()
