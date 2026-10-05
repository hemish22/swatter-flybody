# SWATTER — working notes for agents

## The one hard rule

**Never run anything heavy on the user's laptop.** All big execution goes on the
DGX. This is a standing instruction from the user and it matches the compute plan
in `docs/build-plan.md`.

Concretely: do not download the MaleCNS tables, do not stream
`connectome-weights`, do not run the LIF sweeps or ablations locally. Those touch
1.05 GB and 25.6M edges, and take minutes.

The guard in `offline/common.py` enforces this — `fetch_malecns.py` and
`extract_subgraph.py` exit 2 locally. **Do not set `SWATTER_ALLOW_LOCAL_HEAVY=1`
to work around it** unless the user explicitly asks for a specific local run. It
was set once already by mistake and re-ran a 2-minute pass on the laptop.

Use `./scripts/dgx.sh run <job>` instead. Cheap things stay local: stimulus
generation, `make test`, reading the manifest, byte-compiling.

## State

Week 0 done. Week 1 is the go/no-go gate. Read `docs/week0_status.md` first —
it has the measured neuron and synapse counts, the confirmed parallel escape DN
identities, and what is deliberately not started.

Not started, and in dependency order: `optic_wrapper.py` (flyvis on the MaleCNS
lattice — critical path), `lif.py`, `loom_sweep.py`, `export.py`, then
`engine/`, `web/`, `server/`.

The DGX has not been reached yet — `h4hgpu` and `172.16.0.32` both time out on
port 22. `scripts/dgx.sh` is written but untested against a real machine.

## Conventions

- Python 3.11, `uv` venv at `.venv/`. Run scripts as `.venv/bin/python offline/x.py`.
- Keep heavy scripts free of local imports of each other; they share `offline/common.py`.
- Comments explain *why*, especially where a number is a measured fact from
  MaleCNS rather than a guess. Cite the measurement inline.
- When a plan claim needs checking against the data, check it and write down what
  the data said, even when the plan turns out to be right. Two of the plan's
  figures were off and both are recorded in `docs/week0_status.md`.
- Never hardcode a cell type or body ID that could be looked up. `extract_subgraph.py`
  derives the parallel escape DNs by ranking the data and scores the literature
  names against the result.
- Every bug found gets a regression test in `tests/` that is cheap to run. The
  units bug (radians against degrees) and the post/ pre-major transpose bug both
  produced plausible-looking output, which is exactly why they are pinned now.
