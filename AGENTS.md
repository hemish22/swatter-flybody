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

Week 0 done. Week 1 is the go/no-go gate. The optic lobe half of the gate has
been run: **Partial, not Go**. Read `docs/optic_lobe.md` first (it supersedes
the "what is left" list in `docs/week0_status.md`), then `docs/week0_status.md`
for the measured neuron and synapse counts and the escape DN identities.

Built: `offline/eye.py` (721-column lattice, 2-D loom renderer),
`offline/optic_wrapper.py` (flyvis behind `OpticLobe`), `offline/malecns_ol.py`
(MaleCNS columns, lattice orientation, LPLC2 <- T4/T5), `offline/optic_gate.py`.
Not started, in dependency order: `lif.py`, `loom_sweep.py`, `export.py`, then
`engine/`, `web/`, `server/`. The optic-lobe `malecns` backend is deliberately
not built (reasons in `docs/optic_lobe.md`).

**This host, `dgxa100`, is the DGX.** No ssh hop and no `scripts/dgx.sh`: run
jobs directly with `SWATTER_REMOTE=1 SWATTER_DATA=data .venv/bin/python
offline/<job>.py`. That variable is the repo's own DGX marker, not a bypass. The
GPUs are shared: check `nvidia-smi` and pick a free one with
`CUDA_VISIBLE_DEVICES`, and cap `torch.set_num_threads` in anything CPU-side.

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
- Trials start from a network settled on their own first frame, never on a bare
  sky: otherwise a disc already on screen at t=0 arrives as an onset flash and
  swamps the loom-versus-control comparison (`docs/optic_lobe.md`).
- Every bug found gets a regression test in `tests/` that is cheap to run. The
  units bug (radians against degrees) and the post/ pre-major transpose bug both
  produced plausible-looking output, which is exactly why they are pinned now.
