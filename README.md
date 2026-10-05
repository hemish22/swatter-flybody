# SWATTER

A browser game where the fruit fly's escape reflex is the opponent.

The fly sees your swatter as a looming object, a connectome-wired escape circuit
decides when and where it jumps, and its reflexes beat yours. The escape decision
comes out of a LIF simulation of MaleCNS neurons; the UI shows which neurons
fired. The body animation only plays back what the neurons already decided.

Full design, criteria and timeline: `docs/build-plan.md`.
Progress notes: [`docs/week0_status.md`](docs/week0_status.md).

## Status

Week 0 complete (data access + escape-circuit extraction). Week 1 is the
go/no-go gate: optic lobe on synthetic looms, criteria 1 and 2 plotted.

Done:
- MaleCNS v1.0 access working, anonymously, no token (see `docs/week0_status.md`)
- LPLC2, LC4, GF and the parallel escape DNs identified from the data
- Escape subgraph extracted and counted: 166,700 neurons, 25.6M edges
- Lab stimulus generator with all four control conditions, 128 trials

Week 1 gate decided (`docs/week1_gate.md`): the optic lobe is built and run
(`docs/optic_lobe.md`) but cannot separate translating discs from looming ones
under any setting of the fitted LIF parameters, so the plan's No-go rescope is
taken: LPLC2 is driven by the stimulus's angular growth and the escape decision
comes from the connectome-wired LIF circuit (`offline/lif.py`,
`offline/loom_sweep.py`). Under that drive the giant fiber spikes for every
expanding trial and no control, at 31 of 60 parameter settings.

Held-out fit done, provisional (`docs/lif_fit.md`); criterion 3 fails and criterion 4 is met
(`docs/heading.md`). Validation v1 done (`docs/validation.md`). Export and a Python reference engine done, parity on 50 stimuli (`docs/engine.md`). The Rust/WASM engine (`engine/`) is built and matches the reference on all 50 parity stimuli, 19x faster than real time under wasmtime and 59x in Node, where the TypeScript glue passes the same parity set (`web/`); not run in a browser yet. A deterministic world model and round simulator (`web/game/`, `docs/game.md`) run on the real engine; with the plan's rules no strike can land (the fly reacts in 11 to 25 ms), so the defaults are a 40 mm hover and hold-to-charge strikes, my decision and easy to reverse. Not started: and all of `engine/`, `web/`, `server/`.

## Heavy work runs on the DGX

Nothing expensive runs on your laptop. The 1.05 GB connectome table, the LIF
sweeps and the ablations belong on the DGX per the compute plan. This is
enforced, not just documented: `offline/common.py` makes `fetch_malecns.py` and
`extract_subgraph.py` exit with code 2 and explain themselves rather than start.

```sh
./scripts/dgx.sh setup                  # one-time: sync code, build venv
./scripts/dgx.sh run fetch_malecns      # ~1.1 GB download
./scripts/dgx.sh run extract_subgraph   # escape subgraph
./scripts/dgx.sh logs extract_subgraph  # follow it
./scripts/dgx.sh pull                   # results back into data/subgraph
```

Code syncs laptop → DGX, results come back with `pull`. `data/raw` never moves.
Jobs run in a detached tmux session, so a dropped VPN does not kill a sweep.

Local work stays for cheap things, which run in under a second:

```sh
make stimuli      # stimulus suite summary
make test         # 73 unit tests (4 more on the DGX: make test-dgx)
make manifest     # escape-circuit counts from the manifest
make check        # all of the above
```

## Layout

```
offline/
  common.py          data dir resolution + the local/DGX guard
  fetch_malecns.py   download the MaleCNS tables        [DGX, ~1.1 GB]
  extract_subgraph.py escape subgraph + circuit counts  [DGX, ~2 min]
  loom.py            lab stimuli and controls           [local, instant]
  eye.py             721-column lattice, 2-D loom renderer [local, instant]
  optic_wrapper.py   flyvis optic lobe behind OpticLobe    [GPU]
  malecns_ol.py      MaleCNS columns, orientation, LPLC2<-T4/T5 [DGX, 30 s]
  optic_gate.py      Week 1 gate, optic-lobe half          [GPU, ~1 min]
  escape_graph.py    495-neuron LPLC2 -> DN graph          [DGX, 30 s]
  lif.py             reference LIF simulator               [GPU]
  loom_sweep.py      criteria 1-2 end to end               [GPU, ~1 min]
  lif_scan.py        scan of the three fitted parameters   [GPU, ~2 min]
engine/              Rust -> WASM, WGSL shaders          [Week 3]
web/
  game/ overlay/ lab/ TypeScript, Vite, Canvas2D         [Week 4]
server/              leaderboard + re-sim                 [Week 5]
docs/
  week0_status.md    what Week 0 found
  optic_lobe.md      Week 1 optic lobe results
  week1_gate.md      Week 1 gate decision (No-go rescope) and evidence
  validation.md      plots and ablations (v1 done)        [Week 2]
scripts/dgx.sh       remote runner
tests/               cheap tests, no connectome data needed
```

## Setup

On the DGX, where everything expensive runs:

```sh
git clone https://github.com/hemish22/swatter-flybody.git
cd swatter-flybody
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/dgx.sh fetch                       # ~1.1 GB into data/raw
./scripts/dgx.sh run extract_subgraph        # escape subgraph + manifest
./scripts/dgx.sh pull                        # results back to the laptop
```

`./scripts/dgx.sh setup` does the clone-side venv and dependency work if the
DGX has `uv`; otherwise the three lines above are equivalent. The runner's
default remote directory is `~/swatter-flybody`, which is where the clone above
lands, so the two agree without configuration.

Python 3.11+. `data/raw/` and `data/subgraph/` are gitignored: the tables are
public, checksummed and one command away, and 1.1 GB does not belong in git.

## How it works

```
swatter kinematics -> eye renderer (2 x 721 hex columns) -> optic lobe
   -> LPLC2 + LC4 -> giant fiber + escape DNs -> takeoff mode + heading -> animation
```

Every stage feeds the next each simulation step; the game renders at 60 fps from
the latest state. The escape readout stops at descending-neuron spikes: the GF
spike and a parallel escape pathway race, and their relative timing sets the
takeoff mode (von Reyn et al. 2014).

Three parameters get fitted — one input gain, one global gain, one threshold —
and they are frozen after the Week 2 fit. They are never tuned to make the game
easier or harder. Difficulty is changed only in the stimulus mapping: hover
height, strike speed cap, swatter size, fly starting heading.

## What is real and what is modelled

Real, from MaleCNS v1.0: the looming detector populations, the giant fiber, the
parallel escape descending neurons, and the synapse counts and signs between
them.

Modelled: point-neuron LIF dynamics, no electrical synapses, heading read from
left/right asymmetry in DN activity rather than pre-takeoff leg posture, and a
flyvis-derived optic lobe. The GF's fastest inputs are electrical and are known
to be undercounted in the EM data, so the chemical path into it is incomplete.
These are stated in the UI as well as in `docs/validation.md`.
