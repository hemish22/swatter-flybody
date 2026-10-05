"""MaleCNS optic lobe on the flyvis lattice: columns, orientation, LPLC2 inputs.

The plan's "ideally via the MaleCNS port" is this file. It does not run flyvis
dynamics on MaleCNS wiring (that is `optic_wrapper.py`'s `malecns` backend, a
later step whose reference, Grigoriy-V/fly-brain, has measured quality problems
of its own: direction selectivity 0.152 against flyvis's 0.391, a spatially
broken OFF pathway, missing CT1/Am wiring). What the game needs from MaleCNS in
the optic lobe is smaller and better pinned down, and it is what the LIF stage
has to be fed with:

    which flyvis T4/T5 columns does each MaleCNS LPLC2 neuron listen to,
    and with how many synapses.

That needs three things, built here in order:

 1. A column for every optic-lobe neuron. 15 types carry MaleCNS's
    `assignedOlHex1/2` tag (Mi1, Tm1, Tm9, L1, ...). T4/T5 do not, so their
    column is the synapse-weighted median of their tagged partners' columns,
    both directions. This is a lookup of MaleCNS data, validated by holding
    out tagged neurons and re-inferring them from the others.
 2. The map from MaleCNS hex axes (hex1, hex2) to flyvis (u, v): an integer
    unimodular matrix, chosen by the cosine similarity of the (source type,
    target type, column offset) synapse filters between the two connectomes,
    then checked on the direction-selective inputs (Mi1/Mi9 -> T4a-d, Tm9 ->
    T5a-d), where a wrong handedness shows as the centroids pointing the
    wrong way.
 3. LPLC2 <- T4/T5 synapse counts, binned onto the 721 flyvis columns by that
    map, checked for the one property a looming detector's wiring must have:
    each direction channel sits on the side of the receptive field its motion
    points away from.

Reads the 1 GB connectome table, so it is DGX-side like fetch_malecns.py.

Usage:
    SWATTER_REMOTE=1 .venv/bin/python offline/malecns_ol.py --side R
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, data_dir, require_remote_execution  # noqa: E402
from eye import EXTENT, Eye, column_plane_deg, flyvis_columns  # noqa: E402

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
OL_SUPERCLASSES = ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal")
BAD_STATUS = ("Out of scope", "Orphan", "Orphan-artifact", "Unimportant")

# Edge weight below which a synapse row is not used for column inference and
# the orientation fit. 5 is the community default and the noise floor for 42%
# postsynaptic completion (fly-brain config.toml). The LPLC2 receptive fields
# use every row (>= 1): they are counts of specific contacts, not a statistic.
INFERENCE_MIN_WEIGHT = 5

DS_PAIRS = [("Mi1", f"T4{g}") for g in "abcd"] + [("Mi9", f"T4{g}") for g in "abcd"] + [
    ("Tm9", f"T5{g}") for g in "abcd"
]
MOTION = tuple(f"T4{g}" for g in "abcd") + tuple(f"T5{g}" for g in "abcd")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_neurons(raw: Path, side: str) -> pd.DataFrame:
    ann = pd.read_feather(
        raw / ANNOTATIONS,
        columns=["bodyId", "type", "somaSide", "superclass", "statusLabel", "assignedOlHex1", "assignedOlHex2"],
    )
    ol = ann[ann["superclass"].isin(OL_SUPERCLASSES) & ann["type"].notna()]
    ol = ol[~ol["statusLabel"].isin(BAD_STATUS)]
    return ol[ol["somaSide"] == side].reset_index(drop=True)


def load_edges(raw: Path, bodies: np.ndarray, min_weight: int) -> pd.DataFrame:
    """Edges with both ends in `bodies`. The table is sorted by weight, so the cut is cheap."""
    tbl = pf.read_table(raw / WEIGHTS, memory_map=True)
    tbl = tbl.filter(pc.greater_equal(tbl["weight"], min_weight))
    keep = pc.and_(
        pc.is_in(tbl["body_pre"], value_set=pa.array(bodies)),
        pc.is_in(tbl["body_post"], value_set=pa.array(bodies)),
    )
    return tbl.filter(keep).to_pandas()


# --------------------------------------------------------------------------- #
# 1. columns
# --------------------------------------------------------------------------- #


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def infer_columns(neurons: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Per-neuron column from tagged partners, both directions, synapse-weighted median.

    Returns bodyId, inf_hex1, inf_hex2, n_tagged_partners for every neuron that
    has at least one tagged partner. A neuron is never its own partner, so
    applying this to tagged neurons is a clean hold-out.
    """
    tagged = neurons["assignedOlHex1"].notna()
    h1 = neurons.set_index("bodyId")["assignedOlHex1"]
    h2 = neurons.set_index("bodyId")["assignedOlHex2"]
    tagged_ids = set(neurons.loc[tagged, "bodyId"])
    a = edges.rename(columns={"body_pre": "body", "body_post": "partner"})
    b = edges.rename(columns={"body_post": "body", "body_pre": "partner"})
    p = pd.concat([a, b], ignore_index=True)
    p = p[p["partner"].isin(tagged_ids) & (p["body"] != p["partner"])]
    p = p.assign(h1=p["partner"].map(h1).astype(float), h2=p["partner"].map(h2).astype(float))
    rows = []
    for body, g in p.groupby("body", sort=False):
        w = g["weight"].to_numpy(dtype=float)
        rows.append(
            (body, weighted_median(g["h1"].to_numpy(), w), weighted_median(g["h2"].to_numpy(), w), len(g))
        )
    return pd.DataFrame(rows, columns=["bodyId", "inf_hex1", "inf_hex2", "n_tagged_partners"])


def assign_columns(neurons: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    inf = infer_columns(neurons, edges)
    n = neurons.merge(inf, on="bodyId", how="left")
    tagged = n["assignedOlHex1"].notna()
    # Hold-out score: tagged neurons re-inferred from the other tagged neurons.
    chk = n[tagged & n["inf_hex1"].notna()]
    dist = np.abs(chk["inf_hex1"] - chk["assignedOlHex1"]) + np.abs(chk["inf_hex2"] - chk["assignedOlHex2"])
    holdout = {
        "n_tagged": int(tagged.sum()),
        "n_inferable": int(len(chk)),
        "exact": float(np.mean(dist == 0)),
        "within_1": float(np.mean(dist <= 1)),
        "within_2": float(np.mean(dist <= 2)),
        "exact_by_type": chk.assign(e=(dist == 0).to_numpy()).groupby("type")["e"].mean().round(3).to_dict(),
    }
    n["hex1"] = np.where(tagged, n["assignedOlHex1"], n["inf_hex1"])
    n["hex2"] = np.where(tagged, n["assignedOlHex2"], n["inf_hex2"])
    n["hex_source"] = np.where(tagged, "tagged", np.where(n["inf_hex1"].notna(), "inferred", "none"))
    return n, holdout


# --------------------------------------------------------------------------- #
# 2. orientation: MaleCNS (hex1, hex2) -> flyvis (u, v)
# --------------------------------------------------------------------------- #


def flyvis_filters(json_path: Path) -> dict:
    """{(src, tar, du, dv): synapses per target cell} from flyvis's fib25-fib19 connectome."""
    d = json.load(open(json_path))
    out: dict = defaultdict(float)
    for e in d["edges"]:
        for (du, dv), n in e["offsets"]:
            out[(e["src"], e["tar"], int(du), int(dv))] += n
    return dict(out)


def malecns_filters(n: pd.DataFrame, edges: pd.DataFrame, types: set[str]) -> dict:
    """Same shape from MaleCNS, offsets in raw hex axes, per source cell."""
    idx = n[n["hex1"].notna() & n["type"].isin(types)].set_index("bodyId")
    e = edges[edges["body_pre"].isin(idx.index) & edges["body_post"].isin(idx.index)].copy()
    e["src"] = e["body_pre"].map(idx["type"])
    e["tar"] = e["body_post"].map(idx["type"])
    e["du"] = (e["body_post"].map(idx["hex1"]) - e["body_pre"].map(idx["hex1"])).astype(int)
    e["dv"] = (e["body_post"].map(idx["hex2"]) - e["body_pre"].map(idx["hex2"])).astype(int)
    count = idx.groupby("type").size()
    g = e.groupby(["src", "tar", "du", "dv"])["weight"].sum()
    return {k: float(v) / float(count[k[0]]) for k, v in g.items()}


def candidate_matrices() -> list[tuple[int, int, int, int]]:
    """Integer 2x2 maps with det +-1 and entries in [-2, 2]: the 12 lattice
    symmetries and every small basis change (104 in all)."""
    return [m for m in itertools.product(range(-2, 3), repeat=4) if abs(m[0] * m[3] - m[1] * m[2]) == 1]


def apply_map(m, filt: dict, pairs: set | None = None) -> dict:
    a, b, c, d = m
    out: dict = defaultdict(float)
    for (s, t, du, dv), v in filt.items():
        if pairs is None or (s, t) in pairs:
            out[(s, t, a * du + b * dv, c * du + d * dv)] += v
    return out


def cosine(x: dict, y: dict) -> float:
    keys = list(set(x) | set(y))
    a = np.array([x.get(k, 0.0) for k in keys])
    b = np.array([y.get(k, 0.0) for k in keys])
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def centroid(filt: dict, s: str, t: str):
    w = [(du, dv, v) for (a, b, du, dv), v in filt.items() if a == s and b == t]
    tot = sum(v for *_, v in w)
    if not tot:
        return None
    return (round(sum(du * v for du, _, v in w) / tot, 2), round(sum(dv * v for _, dv, v in w) / tot, 2))


def choose_orientation(fv: dict, mc: dict) -> tuple[tuple, list[dict]]:
    pairs = {(s, t) for (s, t, _, _) in fv} & {(s, t) for (s, t, _, _) in mc}
    fv_c = {k: v for k, v in fv.items() if (k[0], k[1]) in pairs}
    scored = sorted(((cosine(fv_c, apply_map(m, mc, pairs)), m) for m in candidate_matrices()), reverse=True)
    table = [{"M": list(m), "cosine": round(c, 4)} for c, m in scored]
    return scored[0][1], table


# --------------------------------------------------------------------------- #
# 3. crop to the 721 flyvis columns
# --------------------------------------------------------------------------- #


def to_flyvis_columns(n: pd.DataFrame, m: tuple, anchor_type: str = "Mi1") -> tuple[pd.DataFrame, dict]:
    """Add u, v on the flyvis lattice; the origin is the centroid of `anchor_type`'s columns.

    The MaleCNS lobe has ~890 columns against flyvis's 721, so the lattice is
    cropped to the flyvis hexagon (radius 15). Where the optical axis sits in
    the lobe is not in the data; the centre of the Mi1 sheet is the neutral
    choice and is recorded so a different one can be tried.
    """
    a, b, c, d = m
    ok = n["hex1"].notna()
    uv = np.full((len(n), 2), np.nan)
    h = n.loc[ok, ["hex1", "hex2"]].to_numpy(dtype=float)
    ref = n[(n["type"] == anchor_type) & ok][["hex1", "hex2"]].to_numpy(dtype=float)
    centre = np.round(ref.mean(axis=0))
    rel = np.round(h - centre).astype(int)
    uv[ok.to_numpy()] = np.stack([a * rel[:, 0] + b * rel[:, 1], c * rel[:, 0] + d * rel[:, 1]], axis=1)
    out = n.copy()
    out["u"], out["v"] = uv[:, 0], uv[:, 1]
    inside = out["u"].notna() & (np.abs(out["u"]) <= EXTENT) & (np.abs(out["v"]) <= EXTENT) & (
        np.abs(out["u"] + out["v"]) <= EXTENT
    )
    out["in_lattice"] = inside
    info = {"anchor_type": anchor_type, "origin_hex": centre.tolist(), "M": list(m)}
    return out, info


def column_index() -> dict[tuple[int, int], int]:
    return {(int(u), int(v)): i for i, (u, v) in enumerate(flyvis_columns())}


# --------------------------------------------------------------------------- #
# 4. LPLC2 <- T4/T5
# --------------------------------------------------------------------------- #


def lplc2_inputs(n: pd.DataFrame, raw: Path) -> tuple[dict, dict]:
    """Synapse counts onto each LPLC2 neuron from T4a-d / T5a-d, binned to flyvis columns."""
    lp = n[n["type"] == "LPLC2"]
    mot = n[n["type"].isin(MOTION) & n["in_lattice"]]
    edges = load_edges(raw, np.concatenate([lp["bodyId"].to_numpy(), n[n["type"].isin(MOTION)]["bodyId"].to_numpy()]), 1)
    edges = edges[edges["body_post"].isin(lp["bodyId"]) & edges["body_pre"].isin(n[n["type"].isin(MOTION)]["bodyId"])]
    total_syn = int(edges["weight"].sum())
    col = column_index()
    mot = mot.assign(col=[col[(int(u), int(v))] for u, v in zip(mot["u"], mot["v"])])
    pre = mot.set_index("bodyId")
    e = edges[edges["body_pre"].isin(pre.index)].copy()
    e["type"] = e["body_pre"].map(pre["type"])
    e["col"] = e["body_pre"].map(pre["col"]).astype(int)
    kept_syn = int(e["weight"].sum())

    bodies = lp["bodyId"].to_numpy()
    row_of = {int(b): i for i, b in enumerate(bodies)}
    type_index = {t: i for i, t in enumerate(MOTION)}
    w = np.zeros((len(bodies), len(MOTION), Eye.flyvis().n_columns), dtype=np.float32)
    for r in e.itertuples():
        w[row_of[int(r.body_post)], type_index[r.type], r.col] += r.weight
    info = {
        "n_lplc2": int(len(bodies)),
        "synapses_from_t4t5_total": total_syn,
        "synapses_from_t4t5_inside_lattice": kept_syn,
        "fraction_inside_lattice": kept_syn / max(total_syn, 1),
        "by_type": {t: int(w[:, i].sum()) for t, i in type_index.items()},
    }
    return {"bodies": bodies, "weights": w, "types": np.array(MOTION)}, info


def lpi_pathway(n: pd.DataFrame, raw: Path, lplc2_bodies: np.ndarray) -> tuple[dict, dict]:
    """The inhibitory two-hop path T4/T5 -> LPi -> LPLC2, in flyvis columns.

    LPi are lobula-plate intrinsic neurons, all predicted glutamate or GABA
    (inhibitory in insects). They are wide-field, so what is kept is not a
    position for them but their input and output synapse counts:

        t4t5_to_lpi[j, k, c]   synapses LPi cell j gets from T4/T5 type k, column c
        lpi_t4t5_total[j]      ALL its T4/T5 input, including columns outside the
                               721-column hexagon, so a cell mostly fed from
                               outside the lattice is not read as fully driven
        lpi_to_lplc2[i, j]     synapses from LPi j onto LPLC2 neuron i
    """
    lpi = n[n["type"].str.startswith("LPi", na=False)]
    mot_all = n[n["type"].isin(MOTION)]
    mot = mot_all[mot_all["in_lattice"]]
    bodies = np.concatenate([lpi["bodyId"].to_numpy(), mot_all["bodyId"].to_numpy(), lplc2_bodies])
    edges = load_edges(raw, bodies, 1)
    e_in = edges[edges["body_pre"].isin(mot_all["bodyId"]) & edges["body_post"].isin(lpi["bodyId"])]
    e_out = edges[edges["body_pre"].isin(lpi["bodyId"]) & edges["body_post"].isin(lplc2_bodies)]

    j_of = {int(b): i for i, b in enumerate(lpi["bodyId"])}
    i_of = {int(b): i for i, b in enumerate(lplc2_bodies)}
    col = column_index()
    pre = mot.assign(col=[col[(int(u), int(v))] for u, v in zip(mot["u"], mot["v"])]).set_index("bodyId")
    type_index = {t: i for i, t in enumerate(MOTION)}
    w_in = np.zeros((len(lpi), len(MOTION), len(col)), dtype=np.float32)
    total = np.zeros(len(lpi), dtype=np.float32)
    for r in e_in.itertuples():
        total[j_of[int(r.body_post)]] += r.weight
        if r.body_pre in pre.index:
            w_in[j_of[int(r.body_post)], type_index[pre.at[r.body_pre, "type"]], pre.at[r.body_pre, "col"]] += r.weight
    w_out = np.zeros((len(lplc2_bodies), len(lpi)), dtype=np.float32)
    for r in e_out.itertuples():
        w_out[i_of[int(r.body_post)], j_of[int(r.body_pre)]] += r.weight
    info = {
        "n_lpi": int(len(lpi)),
        "lpi_to_lplc2_synapses": int(w_out.sum()),
        "t4t5_to_lpi_synapses_total": int(total.sum()),
        "t4t5_to_lpi_inside_lattice": int(w_in.sum()),
        "lpi_types_reaching_lplc2": lpi.assign(w=lpi["bodyId"].map(dict(zip(lpi["bodyId"], w_out.sum(axis=0)))))
        .groupby("type")["w"].sum().loc[lambda x: x > 0].astype(int).sort_values(ascending=False).to_dict(),
    }
    return {"lpi_t4t5": w_in, "lpi_t4t5_total": total, "lpi_to_lplc2": w_out}, info


def radial_check(inp: dict) -> dict:
    """Is the wiring radial in this lattice orientation?

    For each LPLC2 neuron, the receptive-field centre is the synapse-weighted
    centroid of all its T4/T5 input columns. Each direction channel's input
    centroid, relative to that, is where its motion is read. A looming detector
    has each channel on the side its motion points AWAY from; a wrong lattice
    handedness or a wrong a-d labelling inverts or rotates this. Returns, per
    type, the mean offset (degrees, eye plane) over neurons.
    """
    plane = column_plane_deg(flyvis_columns())
    w = inp["weights"]  # (n, 8, 721)
    tot = w.sum(axis=(1, 2), keepdims=True)
    centre = (w.sum(axis=1)[:, :, None] * plane[None]).sum(axis=1) / np.maximum(tot[:, :, 0], 1e-9)
    out = {}
    for i, t in enumerate(inp["types"]):
        wi = w[:, i]
        s = wi.sum(axis=1, keepdims=True)
        cen = (wi[:, :, None] * plane[None]).sum(axis=1) / np.maximum(s, 1e-9)
        rel = cen - centre
        out[str(t)] = {
            "mean_offset_deg": rel.mean(axis=0).round(2).tolist(),
            "weight_share": float(wi.sum() / max(w.sum(), 1e-9)),
        }
    return out


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("L", "R"), default="R")
    ap.add_argument("--min-weight", type=int, default=INFERENCE_MIN_WEIGHT)
    ap.add_argument("--out", type=Path, default=Path("data/ol"))
    args = ap.parse_args()
    require_remote_execution("malecns_ol.py", "reads the 1.05 GB connectome table and groups ~1M optic-lobe edges")

    raw = data_dir() / "raw"
    n = load_neurons(raw, args.side)
    banner(f"MaleCNS optic lobe, side {args.side}: {len(n):,} typed neurons in {n['type'].nunique()} types")
    edges = load_edges(raw, n["bodyId"].to_numpy(), args.min_weight)
    print(f"edges among them at weight >= {args.min_weight}: {len(edges):,}")

    n, holdout = assign_columns(n, edges)
    print(
        f"column hold-out on {holdout['n_inferable']:,} tagged neurons: exact {holdout['exact']:.3f}, "
        f"within 1 {holdout['within_1']:.3f}, within 2 {holdout['within_2']:.3f}"
    )
    print("  worst types:", dict(sorted(holdout["exact_by_type"].items(), key=lambda kv: kv[1])[:5]))
    print("  hex_source:", n["hex_source"].value_counts().to_dict())
    mot = n[n["type"].isin(MOTION)]
    print("  T4/T5 with a column:", int((mot["hex_source"] != "none").sum()), "of", len(mot))

    banner("orientation")
    fv_json = Path(__import__("flyvis").__file__).parent / "connectome/fib25-fib19_v2.2.json"
    fv = flyvis_filters(fv_json)
    shared = {t for (s, t, _, _) in fv} & set(n["type"])
    mc = malecns_filters(n, edges, shared)
    best, table = choose_orientation(fv, mc)
    print("best M", best, "cosine", table[0]["cosine"], "| 2nd", table[1], "| worst", table[-1])
    mapped = apply_map(best, mc)
    print(f"{'pair':<10} {'flyvis centroid':>18} {'malecns raw':>14} {'malecns mapped':>16}")
    ds_report = []
    for s, t in DS_PAIRS:
        row = (centroid(fv, s, t), centroid(mc, s, t), centroid(mapped, s, t))
        ds_report.append({"pair": f"{s}->{t}", "flyvis": row[0], "malecns_raw": row[1], "malecns_mapped": row[2]})
        print(f"{s + '->' + t:<10} {str(row[0]):>18} {str(row[1]):>14} {str(row[2]):>16}")

    n, lat = to_flyvis_columns(n, best)
    print("lattice:", lat, "| neurons inside the 721-column hexagon:", int(n["in_lattice"].sum()))
    for t in MOTION[:1] + ("Mi1",):
        sub = n[n["type"] == t]
        print(f"  {t}: {int(sub['in_lattice'].sum())} of {len(sub)} inside; columns covered "
              f"{sub[sub['in_lattice']][['u', 'v']].drop_duplicates().shape[0]} of 721")

    banner("LPLC2 <- T4/T5")
    inp, info = lplc2_inputs(n, raw)
    print(json.dumps(info, indent=2))
    lpi, lpi_info = lpi_pathway(n, raw, inp["bodies"])
    inp.update(lpi)
    print("LPi pathway:", json.dumps(lpi_info, indent=2))
    rad = radial_check({k: inp[k] for k in ("weights", "types")})
    print("channel input centroid relative to the LPLC2 receptive-field centre (eye-plane degrees):")
    for t, r in rad.items():
        print(f"  {t}: offset {r['mean_offset_deg']}  share {r['weight_share']:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    n.drop(columns=["inf_hex1", "inf_hex2"]).to_parquet(args.out / f"neurons_{args.side}.parquet", index=False)
    np.savez_compressed(args.out / f"lplc2_inputs_{args.side}.npz", **inp)
    (args.out / f"malecns_ol_{args.side}.json").write_text(
        json.dumps(
            {"holdout": holdout, "orientation": {"chosen": list(best), "table_top": table[:6], "worst": table[-1],
                                                 "ds_centroids": ds_report}, "lattice": lat,
             "lplc2": info, "lpi": lpi_info, "radial_check": rad},
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {args.out}/neurons_{args.side}.parquet, lplc2_inputs_{args.side}.npz, malecns_ol_{args.side}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
