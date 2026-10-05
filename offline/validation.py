"""Validation v1 (Week 2): plots 1-4 and the offline ablations of docs/validation.md.

Everything is at the parameters chosen by `lif_fit.py` (docs/lif_fit.json) and
under the angular-growth drive, so every result inherits the caveat that
selectivity is constructed (docs/week1_gate.md).

Parts (each writes data/ol/val_<part>.json or .npz; run them on different GPUs):

    main      the lab suite (4 kinds x 4 r/v x 8 azimuths) with the giant fiber's
              membrane recorded: feeds plots 1, 2 and 3
    lplc2     ablation 1 as a dose-response: silence a random fraction of LPLC2
              neurons (their drive is removed, so they never spike), 5 seeds each
    shuffle   ablation 3: degree-preserving shuffle of the escape graph's edges
              (in- and out-degree of every neuron kept, weights travel with their
              edge, sign stays with the presynaptic neuron), 20 shuffles, parameters
              NOT refitted
    shuffle_heading
              the same 20 shuffles, scored on criterion 4 (heading side), which
              needs side-specific wiring that a shuffle destroys
    report    figures in docs/figures/ and docs/validation.json (no GPU)

Ablation 4 (the distance-threshold bot) is a closed-form comparison in `report`:
the loom's response depends on r/v alone, so the circuit's trigger distance
scales with the swatter's size and a fixed-distance bot's does not.

Usage (DGX):
    for p in main lplc2 shuffle; do
      CUDA_VISIBLE_DEVICES=<gpu> SWATTER_REMOTE=1 .venv/bin/python offline/validation.py --part $p &
    done; wait
    SWATTER_REMOTE=1 .venv/bin/python offline/validation.py --part report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402

LPLC2_NPZ = Path("data/ol/lplc2_inputs_R.npz")
FIG_DIR = Path("docs/figures")
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
SEEDS = 5
N_SHUFFLES = 20
SWATTER_HALF_WIDTH_MM = 50.0  # loom.py default, the plan's swatter


def shuffle_edges(pre: np.ndarray, post: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10):
    """Degree-preserving edge shuffle by double edge swaps: (a->b),(c->d) becomes (a->d),(c->b).

    Every neuron keeps its in- and out-degree; weights stay attached to the edge.
    A swap that would create a self-loop or a duplicate edge is skipped.
    """
    pre, post = pre.copy(), post.copy()
    present = set(zip(pre.tolist(), post.tolist()))
    n = len(pre)
    done = 0
    for _ in range(swaps_per_edge * n):
        i, j = rng.integers(0, n, size=2)
        a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if i == j or a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present.difference_update({(a, b), (c, d)})
        present.update({(a, d), (c, b)})
        post[i], post[j] = d, b
        done += 1
    return pre, post, done


def _setup():
    import torch  # noqa: F401
    from lif import EscapeGraph, LifParams, upsample_drive, with_params
    from loom_sweep import DURATION_MS, OPTIC_DT_MS, angular_drive, build_trials

    ch = json.loads(Path("docs/lif_fit.json").read_text())["chosen"]
    params = with_params(LifParams(), input_gain=ch["input_gain"], weight_scale=ch["weight_scale"], v_th=ch["v_th"])
    graph = EscapeGraph.load(Path("data/ol/escape_graph.npz"))
    looms = build_trials()
    bodies = np.load(LPLC2_NPZ)["bodies"]
    drive = graph.embed_drive(upsample_drive(angular_drive(looms, LPLC2_NPZ), OPTIC_DT_MS, params.dt, DURATION_MS), bodies)
    return graph, params, looms, drive, graph.source_slots(bodies)


def _run(graph, params, drive, looms, record=None):
    from loom_sweep import trial_readout

    net = graph.network(params, "cuda")
    outs = [net.run(drive[i : i + 32] * params.input_gain, record_v=record) for i in range(0, len(looms), 32)]
    spikes = np.concatenate([o["spikes"] for o in outs])
    v = np.concatenate([o["v"] for o in outs]) if record is not None else None
    return trial_readout(spikes, graph, params, looms), v


def _summ(rows):
    from loom import RV_MS
    from loom_sweep import summarise

    s = summarise(rows)
    expanding = {str(rv): float(np.mean([r["gf_spikes"] > 0 for r in rows if r["kind"] == "expanding" and r["rv_ms"] == rv])) for rv in RV_MS}
    return {
        "fraction_by_kind": s["criterion1_gf_fraction_by_kind"],
        "expanding_by_rv": expanding,
        "theta_by_rv": {str(rv): t for rv, t in zip(RV_MS, s["criterion2_theta_deg_by_rv"])},
        "theta_spread": s["criterion2_theta_spread_ratio"],
    }


def part_main():
    graph, params, looms, drive, _ = _setup()
    gf_r = int(np.flatnonzero((graph.type == "DNp01") & (graph.side == "R"))[0])
    rows, v = _run(graph, params, drive, looms, record=np.array([gf_r]))
    np.savez_compressed("data/ol/val_main.npz", v=v[..., 0].astype(np.float32))
    Path("data/ol/val_main.json").write_text(json.dumps({"rows": rows, "dt_ms": params.dt, "v_th": params.v_th, "v_rest": params.v0}, default=float))
    print("main:", _summ(rows)["fraction_by_kind"])


def part_lplc2():
    graph, params, looms, drive, slots = _setup()
    keep = [i for i, l in enumerate(looms) if l.kind in ("expanding", "receding", "translating", "dimming")]
    out = []
    for f in FRACTIONS:
        for seed in range(SEEDS if 0.0 < f < 1.0 else 1):
            rng = np.random.default_rng(1000 * seed + int(f * 100))
            off = rng.choice(slots, size=int(round(f * len(slots))), replace=False)
            d = drive.copy()
            d[:, :, off] = 0.0
            rows, _ = _run(graph, params, d, [looms[i] for i in keep])
            out.append({"fraction_silenced": f, "seed": seed, **_summ(rows)})
            print(f"silenced {f:.2f} seed {seed}: expanding GF fraction {out[-1]['fraction_by_kind']['expanding']:.2f}", flush=True)
    Path("data/ol/val_lplc2.json").write_text(json.dumps(out, default=float))


def part_shuffle():
    graph, params, looms, drive, _ = _setup()
    out = []
    for k in range(N_SHUFFLES):
        pre, post, done = shuffle_edges(graph.pre, graph.post, np.random.default_rng(k))
        rows, _ = _run(replace(graph, pre=pre, post=post), params, drive, looms)
        out.append({"shuffle": k, "swaps": done, **_summ(rows)})
        print(f"shuffle {k}: expanding {out[-1]['fraction_by_kind']['expanding']:.2f}  controls "
              f"{ {kk: round(v, 2) for kk, v in out[-1]['fraction_by_kind'].items() if kk != 'expanding'} }", flush=True)
    Path("data/ol/val_shuffle.json").write_text(json.dumps(out, default=float))


def part_shuffle_heading():
    from heading import heading_trials, summarise

    graph, params, _, _, _ = _setup()
    azimuths = [float(a) for a in np.arange(0, 360, 22.5)]
    out = []
    for k in range(N_SHUFFLES):
        pre, post, done = shuffle_edges(graph.pre, graph.post, np.random.default_rng(k))
        s = summarise(heading_trials(replace(graph, pre=pre, post=post), params, azimuths))
        out.append({"shuffle": k, **s})
        print(f"shuffle {k}: lateral away {s['lateral_heading_away_fraction']}  ipsilateral-first {s['lateral_first_dn_ipsilateral_fraction']}  "
              f"fired {s['n_fired']}/{s['n_trials']}", flush=True)
    Path("data/ol/val_shuffle_heading.json").write_text(json.dumps(out, default=float))


# --------------------------------------------------------------------------- report


def part_report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from loom import RV_MS, STIMULUS_KINDS

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main = json.loads(Path("data/ol/val_main.json").read_text())
    rows, dt = main["rows"], main["dt_ms"]
    v = np.load("data/ol/val_main.npz")["v"]
    fit = json.loads(Path("docs/lif_fit.json").read_text())
    heading = json.loads(Path("docs/heading.json").read_text())
    lplc2 = json.loads(Path("data/ol/val_lplc2.json").read_text())
    shuf = json.loads(Path("data/ol/val_shuffle.json").read_text())
    shuf_h = json.loads(Path("data/ol/val_shuffle_heading.json").read_text())

    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    blues = ("#9ec5f0", "#5b9be0", "#2a78d6", "#14407c")  # r/v ordered 10 -> 80: one hue, light to dark
    kind_color = {"expanding": "#2a78d6", "receding": "#eb6834", "translating": "#1baf7a", "dimming": "#eda100"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "text.color": ink, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6, "figure.dpi": 130})
    t = np.arange(v.shape[1]) * dt

    # ---- plot 1: GF voltage by r/v (small multiples) + angle at first spike vs r/v
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    for ax, rv, c in zip(axes[:4], RV_MS, blues):
        idx = [i for i, r in enumerate(rows) if r["kind"] == "expanding" and r["rv_ms"] == rv]
        for i in idx:
            ax.plot(t, v[i], color=c, lw=0.9, alpha=0.8)
        ax.axhline(main["v_th"], color=muted, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylim(-53, -44)
        ax.set_title(f"r/v = {rv:.0f} ms", fontsize=9, color=ink, loc="left")
        ax.set_xlabel("time since onset (ms)")
        ax.set_xlim(0, 400)
    axes[0].set_ylabel("right giant fiber, mV")
    axes[0].text(398, main["v_th"] + 0.25, "threshold", ha="right", va="bottom", fontsize=7.5, color=muted)
    ax = axes[4]
    for k, rv in enumerate(RV_MS):
        th = [r["gf_theta_deg"] for r in rows if r["kind"] == "expanding" and r["rv_ms"] == rv and r["gf_theta_deg"] is not None]
        ax.scatter([rv] * len(th), th, s=14, color=blues[k], edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([rv * 0.85, rv * 1.18], [np.median(th)] * 2, color=ink, lw=1.4)
    ax.set_xscale("log")
    ax.set_xticks(RV_MS)
    ax.set_xticklabels([f"{x:.0f}" for x in RV_MS])
    ax.minorticks_off()
    ax.set_xlabel("r/v (ms)")
    ax.set_ylabel("angle at first GF spike (deg)")
    ax.set_title("size at spike vs r/v", fontsize=9, color=ink, loc="left")
    fig.suptitle("Plot 1. Giant fiber voltage (8 azimuths per panel) and the angular size at its first spike; bar = median", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot1_gf_voltage_and_angle.png")
    plt.close(fig)

    # ---- plot 2: selectivity
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.2
    for k, kind in enumerate(STIMULUS_KINDS):
        fr = [np.mean([r["gf_spikes"] > 0 for r in rows if r["kind"] == kind and r["rv_ms"] == rv]) for rv in RV_MS]
        xs = np.arange(len(RV_MS)) + (k - 1.5) * (width + 0.02)
        ax.bar(xs, fr, width, color=kind_color[kind], label=kind)
        for x, f in zip(xs, fr):
            if f == 0:  # a zero bar is invisible; say so
                ax.text(x, 0.02, "0", ha="center", va="bottom", fontsize=8, color=kind_color[kind], fontweight="bold")
    ax.set_xticks(range(len(RV_MS)))
    ax.set_xticklabels([f"{x:.0f}" for x in RV_MS])
    ax.set_xlabel("r/v (ms)")
    ax.set_ylabel("fraction of trials with a GF spike")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=8)
    ax.text(0.0, -0.42, "Selective by construction: the drive is relu(d theta/dt), so the three controls get none.", transform=ax.transAxes, fontsize=7.5, color=muted)
    fig.suptitle("Plot 2. Selectivity (8 azimuths per bar)", x=0.01, ha="left", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot2_selectivity.png", bbox_inches="tight")
    plt.close(fig)

    # ---- plot 3: short-mode fraction vs r/v (train and held-out), and the lag that fixes it
    ch = fit["chosen"]
    pts = {}
    for split in ("train", "test_interp", "test_extrap"):
        for rv, f in ch[split]["short_fraction_by_rv"].items():
            pts[float(rv)] = (f, ch[split]["parallel_minus_gf_ms_by_rv"][rv])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    rvs = sorted(pts)
    axes[0].plot(rvs, [pts[r][0] for r in rvs], "o-", color="#2a78d6", lw=1.6, ms=5)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_xlabel("r/v (ms)")
    axes[0].set_ylabel("short-mode fraction")
    axes[0].set_xticks([7, 14, 28, 57, 120])
    axes[0].set_xticklabels(["7", "14", "28", "57", "120"])
    axes[0].minorticks_off()
    axes[0].set_title("every escape is short", fontsize=9, loc="left", color=ink)
    axes[1].plot(rvs, [pts[r][1] for r in rvs], "o-", color="#2a78d6", lw=1.6, ms=5)
    axes[1].axhline(0, color=muted, lw=0.8)
    axes[1].axhline(6.87, color=muted, lw=0.8, ls=(0, (4, 3)))
    axes[1].text(7, 6.5, "long-mode window, 6.87 ms", fontsize=7.5, color=muted, va="top")
    axes[1].set_xscale("log")
    axes[1].set_xticks([7, 14, 28, 57, 120])
    axes[1].set_xticklabels(["7", "14", "28", "57", "120"])
    axes[1].minorticks_off()
    axes[1].set_ylim(-4, 8)
    axes[1].set_xlabel("r/v (ms)")
    axes[1].set_ylabel("parallel DN minus GF, first spike (ms)")
    axes[1].set_title("DNp103 leads the GF by 1-2 ms", fontsize=9, loc="left", color=ink)
    fig.suptitle("Plot 3. Takeoff mode vs r/v (train and held-out r/v). Criterion 3 not met: no r/v trend.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot3_mode_vs_rv.png")
    plt.close(fig)

    # ---- plot 4: heading asymmetry vs stimulus azimuth (a side index; there is no heading angle to histogram)
    h = heading["trials"]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for k, rv in enumerate(RV_MS):
        pts_h = [(r["azimuth_deg"] + (k - 1.5) * 2.2, r["A"]) for r in h if r["rv_ms"] == rv and r["fired"]]  # small x offset per r/v so the four stay visible
        ax.plot([p[0] for p in pts_h], [p[1] for p in pts_h], "o", color=blues[k], ms=5, markeredgecolor="white", markeredgewidth=0.6, label=f"r/v {rv:.0f} ms")
    ax.axhline(0, color=muted, lw=0.8)
    ax.axvline(90, color=grid, lw=1)
    ax.axvline(270, color=grid, lw=1)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("stimulus body azimuth (deg; 0 ahead, 90 = fly's right)")
    ax.set_ylabel("escape-DN asymmetry A = (R-L)/(R+L)")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8)
    ax.text(0.0, -0.36, "A > 0: right DNs stronger, heading biased left (away from a stimulus on the right). Points are offset in x by r/v.", transform=ax.transAxes, fontsize=7.5, color=muted)
    fig.suptitle("Plot 4. Heading side vs stimulus azimuth (not a polar histogram: the rule yields a side, not an angle)", x=0.01, ha="left", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plot4_heading_side.png", bbox_inches="tight")
    plt.close(fig)

    # ---- ablation 1: LPLC2 dose-response
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    fs = sorted({d["fraction_silenced"] for d in lplc2})
    for k, rv in enumerate(RV_MS):
        mean = [np.mean([d["expanding_by_rv"][str(rv)] for d in lplc2 if d["fraction_silenced"] == f]) for f in fs]
        ax.plot(fs, mean, "o-", color=blues[k], lw=1.6, ms=4.5, label=f"r/v {rv:.0f} ms")
    ax.set_xlabel("fraction of LPLC2 neurons silenced")
    ax.set_ylabel("expanding trials with a GF spike")
    ax.set_ylim(-0.02, 1.08)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=8)
    fig.suptitle("Ablation 1. Silencing LPLC2 (mean of 5 random sets)", x=0.01, ha="left", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation1_lplc2.png", bbox_inches="tight")
    plt.close(fig)

    # ---- ablation 3: what a degree-preserving shuffle does to each criterion
    real_away = heading["summary"]["lateral_heading_away_fraction"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    panels = (
        ("expanding trials with a GF spike", [s["fraction_by_kind"]["expanding"] for s in shuf], 1.0, (0, 1.08)),
        ("GF angle spread across r/v (max/min)", [s["theta_spread"] for s in shuf], fit["chosen"]["train"]["theta_spread"], (1, 4)),
        ("lateral heading points away", [s["lateral_heading_away_fraction"] for s in shuf_h], real_away, (0, 1.08)),
    )
    rng_j = np.random.default_rng(0)
    for ax, (label, vals, real, ylim) in zip(axes, panels):
        ax.scatter(rng_j.uniform(-0.12, 0.12, len(vals)), vals, s=16, color="#9ec5f0", edgecolor="#2a78d6", linewidth=0.7, zorder=3, label="20 shuffles")
        ax.scatter([0.45], [real], s=40, color="#14407c", zorder=3, label="real wiring")
        ax.set_xlim(-0.4, 0.8)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_title(label, fontsize=8.5, loc="left", color=ink)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Ablation 3. Degree-preserving shuffle of the escape graph, parameters not refitted", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation3_shuffle.png")
    plt.close(fig)

    # ---- ablation 4: bot vs circuit trigger distance
    r = SWATTER_HALF_WIDTH_MM
    th = {float(k): v for k, v in fit["chosen"]["train"]["theta_by_rv"].items()}
    ttc_ms = {rv: 480.0 - fit["chosen"]["train"]["t_ms_by_rv"][str(rv)] for rv in RV_MS}
    speed = {rv: r / rv for rv in RV_MS}  # mm per ms
    dist = {rv: speed[rv] * ttc_ms[rv] for rv in RV_MS}
    d0 = dist[20.0]  # bot calibrated to the circuit at r/v = 20
    bot_ttc = {rv: d0 / speed[rv] for rv in RV_MS}

    # ---- numbers
    def band(xs):
        return {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))}

    summary = {
        "plot1_theta_by_rv": th,
        "ablation1_lplc2": {str(f): {"expanding_gf_fraction": band([d["fraction_by_kind"]["expanding"] for d in lplc2 if d["fraction_silenced"] == f]),
                                    "expanding_by_rv": {str(rv): float(np.mean([d["expanding_by_rv"][str(rv)] for d in lplc2 if d["fraction_silenced"] == f])) for rv in RV_MS}}
                            for f in fs},
        "ablation3_shuffle": {
            "n": len(shuf),
            "expanding_gf_fraction": band([s["fraction_by_kind"]["expanding"] for s in shuf]),
            "controls_max_gf_fraction": float(max(max(v for k, v in s["fraction_by_kind"].items() if k != "expanding") for s in shuf)),
            "theta_spread": band([s["theta_spread"] for s in shuf if s["theta_spread"] is not None]),
            "unshuffled_theta_spread": fit["chosen"]["train"]["theta_spread"],
            "n_with_no_expanding_spike": int(sum(s["fraction_by_kind"]["expanding"] == 0 for s in shuf)),
            "n_expanding_all_rv_perfect": int(sum(all(v == 1.0 for v in s["expanding_by_rv"].values()) for s in shuf)),
            "heading_lateral_away_fraction": {**band([s["lateral_heading_away_fraction"] for s in shuf_h]), "real": real_away},
            "heading_ipsilateral_first_fraction": {**band([s["lateral_first_dn_ipsilateral_fraction"] for s in shuf_h]),
                                                    "real": heading["summary"]["lateral_first_dn_ipsilateral_fraction"]},
        },
        "ablation4_bot": {
            "swatter_half_width_mm": r,
            "circuit_trigger_distance_mm_by_rv": dist,
            "circuit_time_to_collision_ms_by_rv": ttc_ms,
            "bot_distance_mm": d0,
            "bot_time_to_collision_ms_by_rv": bot_ttc,
            "circuit_distance_ratio_fastest_over_slowest": dist[10.0] / dist[80.0],
        },
    }
    Path("docs/validation.json").write_text(json.dumps(summary, indent=2, default=float))
    banner("validation numbers")
    print(json.dumps(summary, indent=2, default=float))
    print(f"figures in {FIG_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=("main", "lplc2", "shuffle", "shuffle_heading", "report"))
    args = ap.parse_args()
    require_remote_execution("validation.py", "LIF ablations on a GPU")
    if args.part != "report":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("no GPU visible: set CUDA_VISIBLE_DEVICES to a free GPU (nvidia-smi)")
    {"main": part_main, "lplc2": part_lplc2, "shuffle": part_shuffle, "shuffle_heading": part_shuffle_heading, "report": part_report}[args.part]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
