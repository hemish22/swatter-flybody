"""The escape graph the LIF stage runs on: LPLC2 -> (interneurons) -> descending neurons.

`extract_subgraph.py` counted the circuit at population level; the LIF stage
needs per-neuron weights. This builds them, small enough to run a trial in
milliseconds:

    sources   every LPLC2 neuron, both sides (the optic lobe drives these)
    targets   the giant fiber (DNp01) and the parallel escape descending neurons
              ranked in Week 0: DNp04, DNp02, DNp11, DNp103
    relays    neurons that LPLC2 drives AND that drive a target, ranked by the
              weaker of the two links (a relay is only as good as its worst
              hop), top `--relays`. Zero gives the direct-synapse circuit only.

Edges are every chemical connection among those neurons at weight >= 1 (these
are counts of specific contacts, not a statistic). The sign of each presynaptic
neuron is its predicted neurotransmitter: glutamate, GABA and histamine are
inhibitory in insects, everything else excitatory (the same rule as
`extract_subgraph.py`; getting it backwards inverts the circuit).

Nothing is hardcoded by body ID: types are looked up in the annotations, so a
re-release that moves an ID does not silently change the circuit.

Reads the 1 GB connectome table, so it is DGX-side like fetch_malecns.py:
    SWATTER_REMOTE=1 .venv/bin/python offline/escape_graph.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, data_dir, require_remote_execution  # noqa: E402

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

SOURCE_TYPE = "LPLC2"
GIANT_FIBER = "DNp01"
# Ranked in Week 0 (docs/week0_status.md): DNp04 leads by detector share, DNp103
# is the size-selective one the literature list missed, DNp02/DNp11 are the
# literature candidates that do receive detector input.
PARALLEL_DNS = ("DNp04", "DNp103", "DNp02", "DNp11")
INHIBITORY_NT = ("gaba", "glutamate", "histamine")


def build(raw: Path, relays: int) -> tuple[dict, dict]:
    ann = pd.read_feather(raw / ANNOTATIONS, columns=["bodyId", "type", "somaSide", "superclass"])
    ann = ann[ann["superclass"].notna() & (ann["superclass"] != "")]
    typ = ann.set_index("bodyId")["type"]
    side = ann.set_index("bodyId")["somaSide"]

    src = ann.loc[ann["type"] == SOURCE_TYPE, "bodyId"].to_numpy()
    tgt = ann.loc[ann["type"].isin((GIANT_FIBER, *PARALLEL_DNS)), "bodyId"].to_numpy()

    table = pf.read_table(raw / WEIGHTS, memory_map=True)
    keep_all = pa.array(ann["bodyId"].to_numpy())
    table = table.filter(
        pc.and_(pc.is_in(table["body_pre"], value_set=keep_all), pc.is_in(table["body_post"], value_set=keep_all))
    )
    pre = table["body_pre"].to_numpy()
    post = table["body_post"].to_numpy()
    w = table["weight"].to_numpy()

    relay_ids = np.array([], dtype=np.int64)
    if relays:
        from_src = pd.Series(w[np.isin(pre, src) & ~np.isin(post, np.concatenate([src, tgt]))]).groupby(
            post[np.isin(pre, src) & ~np.isin(post, np.concatenate([src, tgt]))]
        ).sum()
        to_tgt = pd.Series(w[np.isin(post, tgt) & ~np.isin(pre, np.concatenate([src, tgt]))]).groupby(
            pre[np.isin(post, tgt) & ~np.isin(pre, np.concatenate([src, tgt]))]
        ).sum()
        both = pd.concat([from_src.rename("in"), to_tgt.rename("out")], axis=1).dropna()
        both["link"] = both.min(axis=1)
        relay_ids = both.sort_values("link", ascending=False).head(relays).index.to_numpy(dtype=np.int64)

    nodes = np.concatenate([src, relay_ids, tgt]).astype(np.int64)
    role = np.array(["source"] * len(src) + ["relay"] * len(relay_ids) + ["target"] * len(tgt))
    inside = np.isin(pre, nodes) & np.isin(post, nodes)
    pre, post, w = pre[inside], post[inside], w[inside]
    index = {int(b): i for i, b in enumerate(nodes)}

    nt = pd.read_feather(raw / NEUROTRANSMITTERS, columns=["body", "consensus_nt"]).set_index("body")["consensus_nt"]
    sign = np.array(
        [-1.0 if str(nt.get(int(b), "")).lower() in INHIBITORY_NT else 1.0 for b in nodes], dtype=np.float32
    )
    graph = {
        "bodies": nodes,
        "role": role,
        "type": typ.reindex(nodes).fillna("").to_numpy(dtype=str),
        "side": side.reindex(nodes).fillna("").to_numpy(dtype=str),
        "sign": sign,
        "pre": np.array([index[int(b)] for b in pre], dtype=np.int32),
        "post": np.array([index[int(b)] for b in post], dtype=np.int32),
        "weight": w.astype(np.float32),
    }
    info = {
        "neurons": len(nodes),
        "by_role": {r: int((role == r).sum()) for r in ("source", "relay", "target")},
        "edges": int(len(w)),
        "synapses": int(w.sum()),
        "relay_types": pd.Series(graph["type"][role == "relay"]).value_counts().head(12).to_dict(),
        "inhibitory_neurons": int((sign < 0).sum()),
        "direct_source_to_target_synapses": {
            t: int(w[(graph["role"][graph["pre"]] == "source") & (graph["type"][graph["post"]] == t)].sum())
            for t in (GIANT_FIBER, *PARALLEL_DNS)
        },
    }
    return graph, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relays", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("data/ol/escape_graph.npz"))
    args = ap.parse_args()
    require_remote_execution("escape_graph.py", "reads the 1.05 GB connectome table")
    graph, info = build(data_dir() / "raw", args.relays)
    banner("escape graph")
    print(json.dumps(info, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **graph)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
