"""Week 2 fit of the three LIF parameters, with a held-out set.

`lif_scan.py --angular` picked its best of 60 settings on the same 128 trials it
scored them on. This separates the two:

    train      the plan's lab suite: r/v 10, 20, 40, 80 ms x azimuths 0, 45, ... 315
    test       r/v 14, 28, 57 ms (inside the trained range) at azimuths 22.5, 67.5, ...
               and, reported apart, r/v 7 and 120 ms (outside it)

No trial, r/v or azimuth is shared between the two. `input_gain` is anchored on
the train suite by the same rule `loom_sweep.py` uses and applied unchanged to
the test set; nothing on the test side is ever recalibrated or selected on.

Fit rule (fixed here, before looking at any test number):

  1. Score each setting on train by margin = (fraction of expanding trials with a
     giant-fiber spike) - (the largest such fraction among the three controls).
  2. Do not take the single best setting: take the one whose whole 3x3x3
     neighbourhood in the (gain, weight, v_th) grid has the best worst-case
     margin, ties broken by the neighbourhood mean, then by the smaller spread of
     the GF angle across r/v. A setting on the edge of a cliff is not a fit.
     Grid-edge settings have no full neighbourhood and are not eligible.
  3. Report that setting on test. Also report, for every setting that was perfect
     on train, how many were still perfect on test: that is how well "best on
     train" predicts "good on test", independent of the one setting picked.

The result is provisional, not frozen: the LIF constants are not yet cross-checked
against Shiu et al.'s Brian2 code, and criteria 3 and 4 have not been run.

The angular drive is selective by construction (docs/week1_gate.md), so a pass on
margin shows the connectome circuit and parameters are sane, not that the fly's
visual system discriminates.

Settings are independent, so the grid is sharded over GPUs (about 8 s per
setting on one GPU, ~50 min for the grid; four shards ~13 min):

    for k in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=<gpu k> SWATTER_REMOTE=1 .venv/bin/python offline/lif_fit.py --shard $k/4 &
    done; wait
    SWATTER_REMOTE=1 .venv/bin/python offline/lif_fit.py --merge 4      # selection and report, no GPU

With no flags it runs the whole grid on one GPU and reports.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, require_remote_execution  # noqa: E402
from lif import EscapeGraph, LifParams, upsample_drive, with_params  # noqa: E402
from loom import RV_MS, STIMULUS_KINDS  # noqa: E402
from loom_sweep import DURATION_MS, OPTIC_DT_MS, angular_drive, build_trials, calibrate_input_gain, trial_readout  # noqa: E402

TEST_RV_INTERP = (14.0, 28.0, 57.0)
TEST_RV_EXTRAP = (7.0, 120.0)
GAIN_SCALES = (1.0, 1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 11.3, 16.0)
WEIGHT_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
THRESHOLDS = (-48.0, -47.0, -46.0, -45.0, -44.0, -43.0)
CONTROLS = tuple(k for k in STIMULUS_KINDS if k != "expanding")


def split_azimuths(offset_deg: float) -> tuple[float, ...]:
    return tuple(float(a) + offset_deg for a in np.linspace(0, 360, 8, endpoint=False))


def build_looms(rvs, azimuth_offset_deg: float):
    """build_trials() with a chosen r/v list and an azimuth offset (train uses 0)."""
    from loom import Loom

    return [
        Loom(kind=k, rv_ms=rv, azimuth_deg=az, duration_ms=DURATION_MS, dt_ms=OPTIC_DT_MS)
        for k in STIMULUS_KINDS
        for rv in rvs
        for az in split_azimuths(azimuth_offset_deg)
    ]


def score(rows: list[dict], rvs) -> dict:
    """Margin, per-kind GF fractions and, per r/v, the GF angle and time of the expanding trials."""
    sel = [r for r in rows if r["rv_ms"] in rvs]
    frac = {k: float(np.mean([r["gf_spikes"] > 0 for r in sel if r["kind"] == k])) for k in STIMULUS_KINDS}
    theta, t_ms = {}, {}
    for rv in rvs:
        e = [r for r in sel if r["kind"] == "expanding" and r["rv_ms"] == rv and r["gf_t_ms"] is not None]
        theta[str(rv)] = float(np.median([r["gf_theta_deg"] for r in e])) if e else None
        t_ms[str(rv)] = float(np.median([r["gf_t_ms"] for r in e])) if e else None
    by_rv = {
        str(rv): float(np.mean([r["gf_spikes"] > 0 for r in sel if r["kind"] == "expanding" and r["rv_ms"] == rv]))
        for rv in rvs
    }
    thetas = [v for v in theta.values() if v is not None]
    # criterion 3 inputs: among expanding trials that escape at all, the short-mode fraction and the
    # median lag (parallel first spike minus GF first spike; negative = parallel DNs lead) per r/v
    short, lag = {}, {}
    for rv in rvs:
        e = [r for r in sel if r["kind"] == "expanding" and r["rv_ms"] == rv and r["mode"] != "none"]
        short[str(rv)] = float(np.mean([r["mode"] == "short" for r in e])) if e else None
        both = [r["parallel_t_ms"] - r["gf_t_ms"] for r in e if r["parallel_t_ms"] is not None and r["gf_t_ms"] is not None]
        lag[str(rv)] = float(np.median(both)) if both else None
    return {
        "short_fraction_by_rv": short,
        "parallel_minus_gf_ms_by_rv": lag,
        "fraction": frac,
        "margin": frac["expanding"] - max(frac[k] for k in CONTROLS),
        "expanding_by_rv": by_rv,
        "theta_by_rv": theta,
        "t_ms_by_rv": t_ms,
        "theta_spread": float(max(thetas) / min(thetas)) if len(thetas) >= 2 else None,
    }


def run_trials(graph: EscapeGraph, params: LifParams, lif_drive: np.ndarray, looms, device: str) -> list[dict]:
    net = graph.network(params, device)
    spikes = np.concatenate(
        [net.run(lif_drive[i : i + 32] * params.input_gain)["spikes"] for i in range(0, len(looms), 32)]
    )
    return trial_readout(spikes, graph, params, looms)


def select(rows: list[dict], shape: tuple[int, int, int]) -> int:
    """Index of the setting chosen by the neighbourhood rule in the module docstring."""
    from scipy.ndimage import minimum_filter, uniform_filter

    m = np.array([r["train"]["margin"] for r in rows]).reshape(shape)
    spread = np.array([r["train"]["theta_spread"] if r["train"]["theta_spread"] is not None else 99.0 for r in rows]).reshape(shape)
    # edges padded with -1: a margin can't go below that, so edge cells never look robust
    worst = minimum_filter(m, size=3, mode="constant", cval=-1.0)
    mean = uniform_filter(m, size=3, mode="constant", cval=-1.0)
    key = np.stack([worst.ravel(), mean.ravel(), -spread.ravel()], axis=1)
    interior = np.zeros(shape, dtype=bool)
    interior[1:-1, 1:-1, 1:-1] = True
    key[~interior.ravel()] = -np.inf
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    return int(order[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, default=Path("data/ol/escape_graph.npz"))
    ap.add_argument("--lplc2", type=Path, default=Path("data/ol/lplc2_inputs_R.npz"))
    ap.add_argument("--save", type=Path, default=Path("docs/lif_fit.json"))
    ap.add_argument("--shard", help="K/N: evaluate only settings with index %% N == K, write data/ol/fit_shard_K.json")
    ap.add_argument("--merge", type=int, metavar="N", help="combine the N shard files, select and report (no GPU)")
    args = ap.parse_args()
    require_remote_execution("lif_fit.py", "~380 LIF settings x ~290 trials on a GPU")
    import torch

    if args.merge is None and not torch.cuda.is_available():
        raise SystemExit("no GPU visible: set CUDA_VISIBLE_DEVICES to a free GPU (nvidia-smi)")

    t0 = time.time()
    graph = EscapeGraph.load(args.graph)
    bodies = np.load(args.lplc2)["bodies"]
    base = LifParams()

    train_looms = build_trials()
    test_looms = build_looms(TEST_RV_INTERP + TEST_RV_EXTRAP, azimuth_offset_deg=22.5)
    assert not ({(l.rv_ms, l.azimuth_deg) for l in train_looms} & {(l.rv_ms, l.azimuth_deg % 360) for l in test_looms})

    train_drive = angular_drive(train_looms, args.lplc2)
    test_drive = angular_drive(test_looms, args.lplc2)
    gain0 = calibrate_input_gain(train_drive, train_looms, base)  # train expanding r/v = 20 only
    print(f"input gain by rule (train expanding r/v=20): {gain0:.5f}", flush=True)

    grid = list(itertools.product(GAIN_SCALES, WEIGHT_SCALES, THRESHOLDS))
    if args.merge is not None:
        rows = [None] * len(grid)
        for k in range(args.merge):
            for i, r in json.loads(Path(f"data/ol/fit_shard_{k}.json").read_text()).items():
                rows[int(i)] = r
        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            raise SystemExit(f"{len(missing)} settings missing from the shards (first: {missing[:5]})")
        return report(rows, grid, gain0, args.save, t0)

    def embed(d):
        return graph.embed_drive(upsample_drive(d, OPTIC_DT_MS, base.dt, DURATION_MS), bodies)

    train_lif, test_lif = embed(train_drive), embed(test_drive)

    shard_k, shard_n = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    rows = {}
    for idx, (gs, ws, vth) in enumerate(grid):
        if idx % shard_n != shard_k:
            continue
        p = with_params(base, input_gain=gain0 * gs, weight_scale=ws, v_th=vth)
        tr = score(run_trials(graph, p, train_lif, train_looms, "cuda"), RV_MS)
        te_rows = run_trials(graph, p, test_lif, test_looms, "cuda")
        rows[idx] = {
            "gain_scale": gs, "weight_scale": ws, "v_th": vth,
            "train": tr,
            "test_interp": score(te_rows, TEST_RV_INTERP),
            "test_extrap": score(te_rows, TEST_RV_EXTRAP),
        }
        if len(rows) % 10 == 0:
            print(f"{len(rows)} settings, {time.time() - t0:.0f}s", flush=True)

    if args.shard:
        out = Path(f"data/ol/fit_shard_{shard_k}.json")
        out.write_text(json.dumps(rows, default=float))
        print(f"wrote {out}  ({time.time() - t0:.0f}s)")
        return 0
    return report([rows[i] for i in range(len(grid))], grid, gain0, args.save, t0)


def report(rows: list[dict], grid: list, gain0: float, save: Path, t0: float) -> int:
    shape = (len(GAIN_SCALES), len(WEIGHT_SCALES), len(THRESHOLDS))
    pick = select(rows, shape)
    chosen = rows[pick]

    perfect = [r for r in rows if r["train"]["margin"] >= 1.0]
    held = [r for r in perfect if r["test_interp"]["margin"] >= 1.0]
    best_train = max(rows, key=lambda r: r["train"]["margin"])

    banner("fit: neighbourhood-robust setting on train, scored on held-out test")
    def show(name, r):
        print(f"{name}: gain x{r['gain_scale']} (input_gain {gain0 * r['gain_scale']:.5f})  weight x{r['weight_scale']}  v_th {r['v_th']:.0f}")
        for split in ("train", "test_interp", "test_extrap"):
            s = r[split]
            print(f"  {split:<11} margin {s['margin']:+.2f}  expanding by r/v {s['expanding_by_rv']}  controls "
                  f"{ {k: round(s['fraction'][k], 2) for k in CONTROLS} }")
            print(f"  {'':<11} GF angle (deg) by r/v {({k: None if v is None else round(v, 1) for k, v in s['theta_by_rv'].items()})}"
                  f"  GF time (ms) {({k: None if v is None else round(v) for k, v in s['t_ms_by_rv'].items()})}")
    show("chosen", chosen)
    print()
    show("best single setting on train (for comparison)", best_train)
    print(f"\nsettings perfect on train: {len(perfect)} of {len(rows)}; still perfect on held-out interpolation: {len(held)}")

    result = {
        "protocol": {
            "train_rv_ms": list(RV_MS), "train_azimuths_deg": list(split_azimuths(0.0)),
            "test_interp_rv_ms": list(TEST_RV_INTERP), "test_extrap_rv_ms": list(TEST_RV_EXTRAP),
            "test_azimuths_deg": [a % 360 for a in split_azimuths(22.5)],
            "grid": {"gain_scales": GAIN_SCALES, "weight_scales": WEIGHT_SCALES, "v_th": THRESHOLDS},
            "input_gain_by_rule": gain0,
        },
        "chosen": {**chosen, "input_gain": gain0 * chosen["gain_scale"]},
        "best_single_on_train": best_train,
        "n_settings": len(rows), "n_perfect_train": len(perfect), "n_perfect_train_and_test_interp": len(held),
        "settings": rows,
    }
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {save}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
