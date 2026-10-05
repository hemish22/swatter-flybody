"""Direction selectivity of the T4/T5 types for each flyvis ensemble member.

Used to choose the optic-lobe member on a criterion that does not look at the
loom gate: the members are ranked by how direction-selective their motion
detectors are under flyvis's own moving-edge protocol. Member 000 is simply the
best by validation loss, and its T4d is barely tuned (DSI 0.14).

    CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/ensemble_dsi.py 0 1 2 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import require_remote_execution  # noqa: E402
import optic_wrapper  # noqa: E402,F401  (sets FLYVIS_ROOT_DIR before flyvis is imported)

TYPES = [f"T4{g}" for g in "abcd"] + [f"T5{g}" for g in "abcd"]


def member_dsi(member: int) -> dict[str, float]:
    import flyvis
    from flyvis.analysis.moving_bar_responses import direction_selectivity_index

    ds = flyvis.NetworkView(flyvis.results_dir / f"flow/0000/{member:03d}").moving_edge_responses()
    dsi = direction_selectivity_index(ds)
    names = list(ds.cell_type.values)
    out = {}
    for t in TYPES:
        inten = 1 if t.startswith("T4") else 0  # T4 is the ON pathway, T5 the OFF
        out[t] = float(np.asarray(dsi.sel(intensity=inten))[..., names.index(t)].squeeze())
    return out


def main() -> int:
    require_remote_execution("ensemble_dsi.py", "simulates flyvis's moving-edge protocol once per member (~1 min each)")
    members = [int(a) for a in sys.argv[1:]] or list(range(10))
    res = {}
    for m in members:
        res[m] = member_dsi(m)
        r = res[m]
        print(f"member {m:03d}  min DSI {min(r.values()):.2f}  mean {np.mean(list(r.values())):.2f}  "
              + " ".join(f"{t} {v:.2f}" for t, v in r.items()), flush=True)
    Path("docs/ensemble_dsi.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
