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

Week 0 done. **Week 1 gate decided: the plan's No-go rescope** (`docs/week1_gate.md`
first, then `docs/optic_lobe.md`). The optic lobe cannot separate translating
discs from looming ones under any setting of the three fitted LIF parameters, so
LPLC2 is driven by the stimulus's angular growth and the escape decision comes
from the connectome-wired LIF circuit. The UI and `docs/validation.md` must say
the looming detection is a model component. `docs/week0_status.md` still has the
measured counts and DN identities.

Built: `eye.py`, `optic_wrapper.py`, `malecns_ol.py`, `optic_gate.py` (optic lobe,
kept offline), `escape_graph.py`, `lif.py`, `loom_sweep.py`, `lif_scan.py`.
Held-out LIF fit done, provisional (`offline/lif_fit.py`, `docs/lif_fit.md`; sharded over
GPUs, ~4 min). Criterion 3 (mode) fails, criterion 4 (heading) is met: `docs/lif_fit.md`, `docs/heading.md`.
Validation v1 done: `docs/validation.md` (plots 1-4 and ablations 1, 3, 4; plot 5 and the GF-silenced mode need the game).
Export + Python reference engine (`offline/export.py`, `offline/engine_ref.py`, `web/brain/`, `docs/engine.md`) agree on 50 parity stimuli.
Rust/WASM engine built (`engine/`, `web/brain/engine.wasm`, tested via `offline/wasm_host.py` + wasmtime). Rust lives in ~/.cargo (`export PATH=$HOME/.cargo/bin:$PATH`); Node is in ~/.local/node (`export PATH=$HOME/.local/node/bin:$PATH`; `cd web && npm test` runs the TypeScript parity test).
Game world model + round sim in `web/game/world.ts` (tests: `cd web && npm test`; skill curves: `node game/difficulty.ts`); defaults deviate from the plan on purpose, see `docs/game.md`.
Not started: renderer/UI, overlay,
`engine/`, `web/`, `server/`. The optic-lobe `malecns` backend is deliberately not
built (`docs/optic_lobe.md`).

**This host, `dgxa100`, is the DGX.** No ssh hop and no `scripts/dgx.sh`: run
jobs directly with `SWATTER_REMOTE=1 SWATTER_DATA=data .venv/bin/python
offline/<job>.py`. That variable is the repo's own DGX marker, not a bypass. The
GPUs are shared: check `nvidia-smi` and pick a free one with
`CUDA_VISIBLE_DEVICES` (GPU jobs refuse to start without one), and cap
`torch.set_num_threads` in anything CPU-side.

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
