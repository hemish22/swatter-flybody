"""Extract the looming-escape subgraph from MaleCNS v1.0 and count it.

This answers the Week 0 deliverable in the build plan:

    "MaleCNS access working; LPLC2, LC4, GF, candidate DN IDs; subgraph file
     with neuron and synapse counts"

and settles the question the plan deliberately refused to assume:

    "Parallel escape DNs are chosen by querying MaleCNS for the strongest DN
     targets of LPLC2 and LC4. Literature candidates such as DNp02, DNp04 and
     DNp11 must be confirmed in the data, not assumed."

So the script does not take a list of parallel DNs. It ranks every descending
neuron by how strongly LPLC2 and LC4 drive it, and prints the ranking. The
literature candidates are then scored against whatever the data says.

Pipeline:
  body-annotations   -> dense neuron index, type/instance/side/superclass
  body-neurotransmitters -> per-neuron sign (glutamate is inhibitory in insects)
  connectome-weights -> one streaming pass, edges with both ends annotated

Every array produced downstream is indexed by the dense neuron index, which is
the index space `weights` lives in. The annotation arrays are filtered and
reordered to match once, in `load_neurons`, so there is exactly one place where
that alignment happens.

Usage:
    python offline/extract_subgraph.py                 # report + write subgraph
    python offline/extract_subgraph.py --min-weight 5  # drop weak edges
    python offline/extract_subgraph.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir as default_data_dir, require_remote_execution  # noqa: E402

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Populations the escape circuit is built from. Types, not instances: MaleCNS
# splits every instance into many bodies (LPLC2 is 185 of them).
DETECTORS = ("LPLC2", "LC4")
GIANT_FIBER = "DNp01"

# Literature candidates for the parallel / long-mode escape pathway. These are
# only labels scored against the data-derived ranking, never an input to it.
LITERATURE_PARALLEL_DNS = ("DNp02", "DNp04", "DNp11")

# Insect neurotransmitters that are inhibitory. Getting this backwards inverts
# the whole network, so it is a named constant rather than an inline literal.
INHIBITORY_NT = ("gaba", "glutamate", "histamine")

DN_SUPERCLASS = "descending_neuron"
BATCH_ROWS = 8_000_000


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Annotations:
    """Per-neuron annotation arrays, all aligned to the dense neuron index."""

    body: np.ndarray  # int64 (n,)
    cell_type: np.ndarray  # object (n,)
    instance: np.ndarray  # object (n,)
    superclass: np.ndarray  # object (n,)
    side: np.ndarray  # object (n,)

    @property
    def n(self) -> int:
        return int(self.body.size)

    def of_type(self, cell_type: str) -> np.ndarray:
        return np.flatnonzero(self.cell_type == cell_type)

    def is_dn(self) -> np.ndarray:
        return self.superclass == DN_SUPERCLASS

    def index_of(self) -> dict[int, int]:
        return {int(b): i for i, b in enumerate(self.body)}


def _str_column(column: pa.ChunkedArray) -> np.ndarray:
    values = column.to_numpy(zero_copy_only=False)
    return np.array(["" if v is None else str(v) for v in values], dtype=object)


def load_neurons(raw: Path) -> Annotations:
    """Annotated neurons, filtered and reordered to a dense index.

    A body counts as annotated when it has a non-empty `superclass`. That filter
    is what yields the 166,700 figure: the feather holds 211,577 bodies, the rest
    being unannotated fragments.
    """
    table = feather.read_table(raw / ANNOTATIONS, memory_map=True)
    superclass = _str_column(table.column("superclass"))
    keep = np.flatnonzero(superclass != "")
    print(f"neurons: {table.num_rows:,} bodies -> {keep.size:,} annotated (superclass non-empty)")

    body = np.asarray(table.column("bodyId").take(keep).to_numpy(zero_copy_only=False), dtype=np.int64)

    # Sort, then keep the first row per bodyId, so a future re-release with
    # duplicate annotations cannot silently shift the index space.
    order = np.argsort(body, kind="stable")
    body = body[order]
    _, first = np.unique(body, return_index=True)  # ascending, one per bodyId
    if first.size != body.size:
        print(f"neurons: dropped {body.size - first.size:,} duplicate bodyId rows")
    rows = order[first]
    body = body[first]

    def take(name: str, dtype=object) -> np.ndarray:
        column = _str_column(table.column(name).take(keep))
        return column[rows] if dtype is object else column[rows].astype(dtype)

    return Annotations(
        body=body,
        cell_type=take("type"),
        instance=take("instance"),
        superclass=take("superclass"),
        side=take("somaSide"),
    )


def load_signs(raw: Path, ann: Annotations) -> np.ndarray:
    """+1 excitatory, -1 inhibitory, from the predicted neurotransmitter."""
    table = feather.read_table(raw / NEUROTRANSMITTERS, memory_map=True)
    bodies = np.asarray(table.column("body").to_numpy(zero_copy_only=False), dtype=np.int64)
    labels = table.column("predicted_nt").to_pylist()

    index_of = ann.index_of()
    sign = np.ones(ann.n, dtype=np.float32)
    matched = 0
    for body, label in zip(bodies, labels):
        i = index_of.get(int(body))
        if i is None or not label:
            continue
        matched += 1
        lowered = str(label).lower()
        sign[i] = -1.0 if any(nt in lowered for nt in INHIBITORY_NT) else 1.0

    print(
        f"signs: {matched:,} of {ann.n:,} neurons have a predicted NT; "
        f"excitatory {int((sign > 0).sum()):,}, inhibitory {int((sign < 0).sum()):,}"
    )
    return sign


def stream_edges(
    raw: Path,
    ann: Annotations,
    min_weight: int,
) -> tuple[sparse.csr_matrix, int, int]:
    """One pass over connectome-weights, keeping edges between annotated neurons.

    The table is sorted by weight descending, so `min_weight > 1` early-stops the
    scan instead of reading the whole 1.05 GB file.
    """
    table = feather.read_table(raw / WEIGHTS, memory_map=True)
    total_rows = table.num_rows
    print(f"edges: scanning {total_rows:,} rows (min_weight={min_weight})")

    lookup = np.full(int(ann.body.max()) + 2, -1, dtype=np.int64)
    lookup[ann.body] = np.arange(ann.n, dtype=np.int64)

    pre_all: list[np.ndarray] = []
    post_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    seen = 0
    scanned_rows = 0

    for batch in table.to_batches(max_chunksize=BATCH_ROWS):
        pre = np.asarray(batch.column("body_pre").to_numpy(zero_copy_only=False), dtype=np.int64)
        post = np.asarray(batch.column("body_post").to_numpy(zero_copy_only=False), dtype=np.int64)
        weight = np.asarray(batch.column("weight").to_numpy(zero_copy_only=False), dtype=np.int64)
        scanned_rows += len(pre)

        if min_weight > 1 and len(weight) and weight[-1] < min_weight:
            print(f"edges: early stop at row {scanned_rows:,} (table is sorted by weight desc)")
            break

        pre_idx = lookup[np.where(pre < lookup.size, pre, 0)]
        post_idx = lookup[np.where(post < lookup.size, post, 0)]
        ok = (pre_idx >= 0) & (post_idx >= 0) & (weight >= min_weight)

        if ok.any():
            pre_all.append(pre_idx[ok])
            post_all.append(post_idx[ok])
            w_all.append(weight[ok])
        seen += int(ok.sum())
        del pre, post, weight, pre_idx, post_idx

    if not pre_all:
        raise SystemExit("no edges survived the annotated-neuron filter")

    pre = np.concatenate(pre_all)
    post = np.concatenate(post_all)
    weight = np.concatenate(w_all)
    del pre_all, post_all, w_all

    synapses = int(weight.sum())
    print(
        f"edges: {seen:,} edges between annotated neurons, {synapses:,} synapses "
        f"(scanned {scanned_rows:,} of {total_rows:,} rows)"
    )

    # W is post-major: W[post, pre], so W @ v gives the current entering each
    # neuron. That is the convention the LIF stage consumes.
    matrix = sparse.csr_matrix((weight.astype(np.float32), (post, pre)), shape=(ann.n, ann.n))
    matrix.sum_duplicates()
    return matrix, seen, synapses


# --------------------------------------------------------------------------- #
# circuit analysis
# --------------------------------------------------------------------------- #


def population_summary(weights: sparse.csr_matrix, ann: Annotations, names: tuple[str, ...]) -> list[dict]:
    out: list[dict] = []
    for name in names:
        idx = ann.of_type(name)
        if idx.size == 0:
            out.append({"type": name, "neurons": 0})
            continue
        # W is post-major: W[post, pre]. So W[idx, :] selects rows, which are
        # postsynaptic targets -> these drive `name`. W[:, idx] selects columns,
        # which are presynaptic sources -> these are driven by `name`.
        driven_by = weights[idx, :]
        drives = weights[:, idx]
        sides = ann.side[idx]
        out.append(
            {
                "type": name,
                "neurons": int(idx.size),
                "instances": sorted({str(v) for v in ann.instance[idx]}),
                "soma_side": {s: int((sides == s).sum()) for s in sorted({str(v) for v in sides})},
                "body_ids": [int(b) for b in ann.body[idx]],
                "incoming_synapses": int(driven_by.sum()),
                "outgoing_synapses": int(drives.sum()),
                # Distinct partners, pooled over every body of this type. indptr
                # differences count rows, which for a 2-body population is just 2,
                # so the partner count has to come from the index arrays.
                "presynaptic_inputs": int(np.unique(driven_by.indices).size),
                "postsynaptic_targets": int(np.unique(drives.indices).size),
            }
        )
    return out


def dn_ranking(
    weights: sparse.csr_matrix,
    ann: Annotations,
    sources: tuple[str, ...],
    exclude_types: tuple[str, ...] = (),
    top: int = 20,
) -> list[dict]:
    """Rank descending neurons by incoming drive from the looming detectors.

    This is the empirical substitute for the plan's "confirm, do not assume".
    """
    source_idx = np.unique(np.concatenate([ann.of_type(name) for name in sources]))
    if source_idx.size == 0:
        return []

    drive = np.asarray(weights[:, source_idx].sum(axis=1)).ravel()
    candidates = ann.is_dn() & (drive > 0)
    for name in exclude_types:
        candidates &= ann.cell_type != name

    order = np.argsort(drive, kind="stable")[::-1]
    rows: list[dict] = []
    for i in order:
        if not candidates[i]:
            continue
        cell_type = str(ann.cell_type[i])
        rows.append(
            {
                "type": cell_type,
                "bodyId": int(ann.body[i]),
                "instance": str(ann.instance[i]),
                "incoming_synapses_from_detectors": int(drive[i]),
                "in_literature_set": cell_type in LITERATURE_PARALLEL_DNS,
            }
        )
        if len(rows) >= top:
            break
    return rows


def print_ranking(title: str, rows: list[dict], value_key: str = "incoming_synapses_from_detectors") -> None:
    print(f"\n-- {title} --")
    if not rows:
        print("   (none)")
        return
    for rank, row in enumerate(rows, 1):
        marker = "  <- literature candidate" if row["in_literature_set"] else ""
        print(f"   {rank:2d}. {row['type']:<10} body {row['bodyId']:<8} "
              f"{row[value_key]:>9,} syn{marker}")


def escape_circuit_table(weights: sparse.csr_matrix, ann: Annotations) -> list[dict]:
    """Synapses from each detector onto the GF and the parallel DN candidates.

    This is the plan's central premise, stated as a number: LPLC2 and LC4 synapse
    directly onto the giant fiber (Ache et al. 2019). If these two columns are
    large, the premise holds in MaleCNS v1.0; if they are near zero, the escape
    circuit cannot be built from the chemical synapses in this dataset and the
    Week 1 gate needs the fallback path.
    """
    targets = (GIANT_FIBER, *LITERATURE_PARALLEL_DNS)
    source_idx = {name: ann.of_type(name) for name in DETECTORS}
    rows: list[dict] = []
    for target in targets:
        idx = ann.of_type(target)
        if idx.size == 0:
            continue
        drives = weights[idx, :]
        total = int(drives.sum())
        entry: dict[str, object] = {
            "target": target,
            "bodies": int(idx.size),
            "total_incoming_synapses": total,
        }
        for name, src in source_idx.items():
            entry[f"from_{name}"] = int(drives[:, src].sum())
        if total:
            share = sum(int(entry[f"from_{n}"]) for n in source_idx)
            entry["detector_share_of_incoming"] = round(share / total, 4)
        rows.append(entry)
    return rows


def print_circuit(rows: list[dict]) -> None:
    print("\n-- escape circuit core: detector -> DN synapse counts --")
    print(f"   {'target':<8} {'bodies':>6} {'incoming':>10} " + " ".join(f"{n:>8}" for n in DETECTORS) + "  share")
    for row in rows:
        cells = " ".join(f"{int(row[f'from_{n}']):>8,}" for n in DETECTORS)
        share = row.get("detector_share_of_incoming")
        share_text = f"{share:>6.1%}" if share is not None else "     -"
        print(f"   {row['target']:<8} {row['bodies']:>6} {int(row['total_incoming_synapses']):>10,} "
              f"{cells}  {share_text}")


def detector_dn_block(
    weights: sparse.csr_matrix,
    ann: Annotations,
    sources: tuple[str, ...],
) -> tuple[list[str], sparse.csr_matrix]:
    """Compact (source population x descending neuron) matrix.

    Row i sums every synapse from population `sources[i]` onto one DN. Signed
    weights are applied by the caller; this returns raw synapse counts.
    """
    dn_idx = np.flatnonzero(ann.is_dn())
    dn_rows = weights[dn_idx, :]  # (n_dns x n_neurons)

    block_rows: list[np.ndarray] = []
    block_cols: list[np.ndarray] = []
    block_data: list[float] = []
    for row_i, name in enumerate(sources):
        source_idx = ann.of_type(name)
        if source_idx.size == 0:
            continue
        block = dn_rows[:, source_idx].T.tocsr()  # (n_sources x n_dns)
        coo = block.tocoo()
        block_rows.append(np.full(coo.nnz, row_i, dtype=np.int32))
        block_cols.append(coo.col.astype(np.int32))
        block_data.append(coo.data)

    shape = (len(sources), dn_idx.size)
    matrix = sparse.coo_matrix(
        (
            np.concatenate(block_data),
            (np.concatenate(block_rows), np.concatenate(block_cols)),
        ),
        shape=shape,
    ).tocsr()
    labels = [str(v) for v in ann.cell_type[dn_idx]]
    return labels, matrix


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, type=Path, help="data dir (default: $SWATTER_DATA or data)")
    parser.add_argument("--min-weight", type=int, default=1, help="drop edges with fewer synapses")
    parser.add_argument("--out", default="subgraph", type=Path, help="output dir under the data dir")
    parser.add_argument("--report-only", action="store_true", help="skip writing subgraph files")
    args = parser.parse_args()

    data = default_data_dir(args.data)
    if not (data / "raw").is_dir():
        print(f"no raw tables under {data}/raw; run fetch_malecns.py on the DGX first", file=sys.stderr)
        return 1

    require_remote_execution(
        "extract_subgraph.py",
        "one streaming pass over the 1.05 GB connectome-weights table "
        "(151.9M rows) to rebuild the 25.6M-edge annotated subgraph",
    )

    raw = data / "raw"
    out_dir = data / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = load_neurons(raw)
    sign = load_signs(raw, ann)
    weights, n_edges, n_synapses = stream_edges(raw, ann, args.min_weight)

    summary = population_summary(weights, ann, (*DETECTORS, GIANT_FIBER))
    print("\n-- populations --")
    for row in summary:
        compact = {k: v for k, v in row.items() if k != "body_ids"}
        compact["body_ids_first"] = row.get("body_ids", [])[:8]
        print("   " + json.dumps(compact))

    # The plan asks for the parallel escape DNs to be derived, not assumed. The
    # GF itself is excluded from the ranking so it cannot win by default.
    combined = dn_ranking(weights, ann, DETECTORS, exclude_types=(GIANT_FIBER,))
    print_ranking("descending neurons by incoming drive from LPLC2+LC4 (GF excluded)", combined)
    for name in DETECTORS:
        print_ranking(f"descending neurons by incoming drive from {name} alone (GF excluded)",
                      dn_ranking(weights, ann, (name,), exclude_types=(GIANT_FIBER,), top=10))
    print_ranking("strongest inputs onto the GF itself (any neuron type)",
                  gf_inputs(weights, ann), value_key="incoming_synapses")

    literature_hit = [row for row in combined if row["in_literature_set"]]
    print(f"\n-- literature candidates {list(LITERATURE_PARALLEL_DNS)} --")
    if literature_hit:
        for row in literature_hit:
            print(f"   {row['type']}: {row['incoming_synapses_from_detectors']:,} synapses "
                  f"from LPLC2+LC4")
    else:
        print("   none of them receive any LPLC2 or LC4 input in MaleCNS v1.0")
    print("   treat the ranking above, not this list, as the parallel-pathway choice")

    circuit = escape_circuit_table(weights, ann)
    print_circuit(circuit)

    if not combined:
        print("\nno detector input onto any descending neuron: check the superclass")
        print("annotation before assuming the extract is empty")
        return 1

    if args.report_only:
        print("\nreport-only: nothing written")
        return 0

    labels, block = detector_dn_block(weights, ann, (*DETECTORS, GIANT_FIBER))
    np.savez_compressed(
        out_dir / "detector_dn.npz",
        data=block.data.astype(np.float32),
        indices=block.indices.astype(np.int32),
        indptr=block.indptr.astype(np.int64),
        shape=np.array(block.shape, dtype=np.int64),
        populations=np.array([*DETECTORS, GIANT_FIBER]),
        dn_labels=np.array(labels),
    )
    print(f"\nwrote {out_dir / 'detector_dn.npz'}: {block.shape[0]} populations x "
          f"{block.shape[1]:,} DNs, {block.nnz:,} nonzero entries")

    manifest = {
        "source": "MaleCNS v1.0, GCS bucket flyem-male-cns",
        "min_weight": args.min_weight,
        "annotated_neurons": ann.n,
        "edges": n_edges,
        "synapses": n_synapses,
        "detectors": list(DETECTORS),
        "giant_fiber_type": GIANT_FIBER,
        "giant_fiber_bodies": [int(b) for b in ann.body[ann.of_type(GIANT_FIBER)]],
        "parallel_dn_candidates_data_derived": [row["type"] for row in combined],
        "literature_parallel_dn_candidates": list(LITERATURE_PARALLEL_DNS),
        "literature_candidates_with_detector_input": [row["type"] for row in literature_hit],
        "escape_circuit_core": circuit,
        "population_summary": summary,
        "dn_ranking_lplc2_lc4": combined,
    }
    np.save(out_dir / "sign.npy", sign)
    np.save(out_dir / "body_index.npy", ann.body)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir / 'manifest.json'}, sign.npy, body_index.npy")
    return 0


def gf_inputs(weights: sparse.csr_matrix, ann: Annotations) -> list[dict]:
    """Which annotated neurons drive the giant fiber, strongest first."""
    gf_idx = ann.of_type(GIANT_FIBER)
    if gf_idx.size == 0:
        return []
    drive = np.asarray(weights[gf_idx, :].sum(axis=0)).ravel()
    rows: list[dict] = []
    for i in np.argsort(drive, kind="stable")[::-1][:10]:
        if drive[i] <= 0:
            continue
        rows.append(
            {
                "type": str(ann.cell_type[i]),
                "bodyId": int(ann.body[i]),
                "instance": str(ann.instance[i]),
                "incoming_synapses": int(drive[i]),
                "in_literature_set": str(ann.cell_type[i]) in LITERATURE_PARALLEL_DNS,
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
