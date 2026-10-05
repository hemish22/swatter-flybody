# Week 0 status — data access and escape-circuit extraction

Dates: Sep 28 – Oct 4, 2026. Done: Oct 3, 2026.

Week 0 deliverable from the plan: *"MaleCNS access working; LPLC2, LC4, GF,
candidate DN IDs; subgraph file with neuron and synapse counts; flyvis port
running."*

Three of the four are done. The flyvis port is not started, and it is the
critical-path item for the Week 1 gate — see "What is left".

## Access route

The plan listed "neuPrint client or MaleCNS bulk download" as the thing to
settle in Week 0. Both work; the bulk route is better for this project.

| Route | What it needs | Verdict |
| --- | --- | --- |
| GCS bulk, `flyem-male-cns` bucket | nothing, anonymous HTTPS | **used** |
| neuPrint `male-cns:v1.0` | account token, mandatory | fallback only |

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

Anonymous, and the bucket listing is browsable, so this is reproducible by
anyone who clones the repo — which matters for the "it's faked" defence. The
one file not in the bucket (the optic-lobe hex column assignments) comes from
`flyconnectome/2025malecns`, pinned to a commit hash in `fetch_malecns.py`.

Tables taken, and what each is for:

| Table | Size | Rows | Used for |
| --- | ---: | ---: | --- |
| `body-annotations` | 14 MB | 211,577 | types, instances, sides, superclass |
| `body-neurotransmitters` | 43 MB | 1,835,518 | synapse signs |
| `connectome-weights` | 1,051 MB | 151,856,684 | the connectome |
| `optic-columns.xlsx` | 112 KB | — | optic-lobe hex columns |

Deliberately not taken: `syn-partners` (6.8 GB) is `connectome-weights` exploded
to one row per synapse, and `body-stats` (778 MB) is at supervoxel granularity
so it will not join to `bodyId`. `connectome-weights` weights are integer
synapse counts and sum to 311,833,243, which matches the `syn-partners` row
count exactly.

### Two corrections to the plan's numbers

- The plan says "125M synapses". The dataset holds **311,833,243 synapses**.
  125M is closer to the number of synapses *between annotated neurons*, which
  measured **124,177,617**. Both numbers are real; they count different things.
  `docs/` should use the annotated-pair figure, because that is what the
  simulation actually contains.
- Edges between annotated neurons: **25,582,938**, which needs no threshold —
  filtering to annotated bodies on both ends is enough.

## Neuron and synapse counts

`make manifest`, from `data/subgraph/manifest.json`:

| Type | Instance | Neurons | Soma side | Incoming syn | Outgoing syn | Distinct pre | Distinct post |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| LPLC2 | `LPLC2_L` / `LPLC2_R` | 185 | L 94 / R 91 | 351,434 | 186,652 | 32,868 | 185 |
| LC4 | `LC4_L` / `LC4_R` | 126 | L 71 / R 55 | 305,573 | 144,613 | 20,664 | 126 |
| DNp01 (GF) | `DNp01(GF)_L` / `_R` | 2 | L 1 / R 1 | 36,733 | 2,543 | 1,260 | 2 |

Whole subgraph: 166,700 annotated neurons, 25,582,938 edges, 124,177,617 synapses.

Cell IDs, confirmed against the annotation file rather than assumed:

| Role | Type | bodyIds |
| --- | --- | --- |
| Looming size | `LPLC2` | 185 bodies |
| Looming velocity | `LC4` | 126 bodies |
| Giant fiber | `DNp01` | 10001, 10010 |
| Parallel escape | `DNp04` | 11137, 531898 |
| Parallel escape | `DNp02` | 10117, 10197 |
| Parallel escape | `DNp11` | 10106, 10259 |

`hemibrainType` for DNp01 reads `Giant Fiber`, so the GF identification is
annotated rather than hand-checked against morphology. That is worth recording
because the plan flagged "hand-check the GF, whose giant axon is unmistakable"
as a risk; the annotation makes the check unnecessary.

Note LPLC2 (94 L / 91 R) and LC4 (71 L / 55 R) are left/right asymmetric. That is
incomplete reconstruction, not a modelling artefact, and it will show up as a
slight left/right bias in criterion 4 unless the two sides are normalised.

## The parallel escape DNs: confirmed, not assumed

The plan said to pick these by querying MaleCNS, and not to assume the
literature candidates. Doing that:

| Target | Bodies | Total incoming | From LPLC2 | From LC4 | Detector share |
| --- | ---: | ---: | ---: | ---: | ---: |
| DNp01 (GF) | 2 | 36,733 | 4,862 | 6,362 | 30.6% |
| DNp04 | 2 | 21,034 | 3,398 | 11,597 | **71.3%** |
| DNp02 | 2 | 16,038 | 5 | 4,209 | 26.3% |
| DNp11 | 2 | 21,379 | 71 | 3,666 | 17.5% |

All three literature candidates receive detector input, so none of them is
dropped. DNp04 is by a wide margin the most detector-dominated descending neuron
in the dataset and leads the parallel set. Ranking descending neurons by
LPLC2+LC4 input with the GF excluded puts DNp04 first, then DNp103, DNp02,
DNg40, DNp11, DNp03, DNp06.

DNp103 is the one genuinely new name here: 5,065 synapses from LPLC2 and only 335
none from LC4, so it is size-selective where DNp04 is strongly velocity-selective
too. Worth keeping in the model as the second parallel channel rather than
discarding it for not being in the literature.

**The central premise holds.** LPLC2 and LC4 both synapse directly onto the GF,
and together account for 30.6% of its recorded chemical input. That is the
"Ache et al. 2019" claim, measured.

### The honest caveat on the GF

The GF receives 36,733 synapses but sends only 2,543. Its strongest recorded
inputs are not the visual detectors at all but DNp70 (1,416), PVLP010 (711) and
several SAD neurons. That is consistent with the plan's own note that the GF
drives the jump motor largely through electrical synapses, which EM
reconstruction records poorly. Two consequences:

- The chemical-synapse-only path into the GF understates the real one, so the
  Week 1 fit will be compensating for a known deficit. That has to be stated in
  `docs/validation.md`, not just in this file.
- The plan's decision not to propagate through the VNC in v1 is the right call
  and the data supports it more strongly than the plan argued.

## Stimulus generator

`offline/loom.py` implements the plan's formula and the four control conditions,
generating all of them from one code path so a control cannot accidentally
differ in some other way. 128 trials: 4 r/v × 8 azimuths × 4 stimulus types.

At r/v = 20 ms over a 400 ms trial:

| Stimulus | θ start | θ end | Peak dθ/dt | Should escape |
| --- | ---: | ---: | ---: | --- |
| expanding | 4.77° | 28.07° | +0.335 °/ms | yes |
| receding | 4.77° | 2.60° | −0.003 °/ms | no |
| translating | 14.25° | 14.25° | 0 | no |
| dimming | 14.25° | 14.25° | 0 | no |

The flat controls are sized to the expanding trial's late run-up rather than its
mid-trial size, so they are ~14° discs the eye really responds to and differ
only in lacking growth. An earlier version matched them mid-trial, which made
them 0.6° — passing by being too small to see, which tests nothing.

33 unit tests cover the generator and the manifest claims, and run in 0.03 s
locally. Three of them pin bugs that were found and fixed during Week 0:
`receding` was byte-identical to `expanding` because of an `abs()`, the
luminance raster came back fully dark because angular offsets were in radians
while θ was in degrees, and the DN ranking was indexing the 211k-row annotation
table against the 166k-row weight matrix.

## Compute: heavy work goes to the DGX

Per instruction, nothing expensive runs on the laptop. `offline/common.py`
enforces it — `fetch_malecns.py` and `extract_subgraph.py` exit with code 2 and
explain themselves rather than starting. `scripts/dgx.sh` syncs code, runs a job
in a detached tmux session, and pulls results back; `data/raw` is never copied
down. `make help` splits the targets.

**The DGX has not been reached yet.** Neither `h4hgpu` nor `172.16.0.32`
answers on port 22 from here, so `scripts/dgx.sh` is written but untested
against a real machine. The host needs confirming.

## What is left

1. **flyvis on the MaleCNS lattice.** Not started, and it is the critical path.
   Everything above is CPU-only bookkeeping; the Week 1 gate needs an optic lobe
   that carries a looming signal. Two things to settle: `Grigoriy-V/fly-brain`
   has no environment notes in the README, so it may need pinning, and flyvis's
   native integration step has to be measured. If it is coarser than ~5 ms the
   LIF stages interpolate its outputs at 0.5–1 ms.
2. **The Week 1 gate itself** — criterion 1 and 2 plotted, and a go/no-go call.
   The partial-go path (radial-motion template over T4/T5 outputs) is the likely
   outcome and this repo is already arranged for it: `extract_subgraph.py` writes
   a detector × DN block rather than a hardcoded circuit, so substituting a
   modelled LPLC2 input for a connectome one is a change to one input, not a
   rewrite.
3. **`offline/lif.py`** and **`loom_sweep.py`**. Written last because their shape
   depends on what the gate decides about the optic lobe.

## Reproducing this

```sh
make setup                        # DGX: sync code, build venv
make fetch                        # DGX: ~1.1 GB
make extract                      # DGX: subgraph + manifest
make pull                         # bring data/subgraph back
make check                        # local: lint, tests, stimuli, manifest
```

`data/raw/` and `data/subgraph/` are gitignored. The tables are public,
checksummed and one command away, and 1.1 GB does not belong in git.
