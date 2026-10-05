# Week 1 gate — decision and evidence

Oct 5, 2026. Decided on the user's delegation ("do as you like"); every number is
from a run on the DGX and the files named. Supersedes the "Partial" verdict in
`docs/optic_lobe.md`, which was the optic-lobe half; this adds the LIF half.

## Decision

**Take the plan's No-go rescope: drive LPLC2 from the stimulus's angular growth
and ship a GF-circuit game, not an optic-lobe one.** The escape decision still
comes out of the connectome-wired LIF circuit (LPLC2 → giant fiber and parallel
descending neurons, real synapse counts and signs). The optic lobe stays in the
repo as an offline, validated component and is not in the game loop.

Why, in order of weight:

1. **No setting of the three fitted parameters rescues the optic-lobe path.** The
   Week 2 fit is `input_gain`, `weight_scale`, `v_th`. Scanning a 5 × 5 × 3 grid
   of them on the optic drive (`offline/lif_scan.py`, `docs/lif_scan.json`) gives
   a best criterion-1 margin of **0.00** (expanding minus the strongest control)
   and **zero of 75** settings above it. Translating discs drive the giant fiber
   at least as hard as looming ones wherever the loom does. This is not a
   fitting problem, so it is decidable before Week 2.
2. **Four independent optic-lobe readouts agree** (`docs/optic_lobe.md`):
   hand template, linear LPLC2 wiring, four-direction coincidence, LPi-inhibited.
3. **The rescope works downstream.** With angular-growth drive, 31 of 60 settings
   of the same three parameters give margin **+1.00**: the giant fiber spikes in
   all expanding trials at every r/v and in none of the controls. A broad
   plateau, not a knife edge.
4. **It also shrinks the browser job.** The plan sized the browser at ~1.35 M
   optic-lobe edges per step; with this rescope it is a 495-neuron LIF graph.

What it costs, to be said plainly in the UI and `docs/validation.md`: the looming
detection is **a model component, not the fly's optic lobe**. LPLC2 is driven by
`relu(d theta / dt)` of the rendered disc, weighted by distance from its
receptive-field centre. That is selective by construction, so criterion 1
passing under it is **not evidence about the fly's visual system**, only that the
connectome circuit downstream behaves sensibly. It is the plan's own No-go
wording ("ship as a GF-circuit game, not an optic-lobe one").

## What was built

| Piece | File |
| --- | --- |
| Escape graph: 185 LPLC2 + 300 relays + 10 targets, 42,396 edges, 322,901 synapses | `offline/escape_graph.py` |
| Reference LIF (Shiu et al. 2024 form), batched, deterministic, 12 tests | `offline/lif.py` |
| Criteria 1 and 2 end to end, both drives | `offline/loom_sweep.py` |
| Parameter scan | `offline/lif_scan.py` |

Targets: the giant fiber (DNp01) and the Week 0 parallel pathway DNp04, DNp103,
DNp02, DNp11. Relays are the 300 neurons LPLC2 drives that drive a target,
ranked by the weaker hop; 99 of them are LC4, which is why LC4 is in the graph
though nothing drives it (below). Signs from predicted neurotransmitter
(glutamate, GABA, histamine inhibitory).

## Results

### Through the optic lobe (`logs/loom_sweep.log`, `docs/loom_sweep.json`)

Rule-set gain (the most-driven LPLC2 neuron reaches 1.5× threshold at expanding
r/v = 20, set from expanding trials only). Fraction of 8 azimuths with a GF spike:

| r/v | expanding | receding | translating | dimming |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 |
| 20 | 0.00 | 0.00 | 0.50 | 0.00 |
| 40 | 0.62 | 0.00 | 1.00 | 0.38 |
| 80 | 1.00 | 0.00 | 0.88 | 0.88 |

Fails criterion 1: the loom is silent at r/v 10 and 20 while translating fires,
and dimming fires at r/v 80. The scan above shows no setting fixes it.

### Through the angular drive (`logs/loom_sweep_angular.log`, `docs/lif_scan_angular.json`)

At the rule-set gain the circuit is under-driven (fires for r/v = 80 only,
controls at 0), because the rule guarantees one LPLC2 neuron crosses threshold
and the giant fiber needs many. Over the scan, **31 of 60 settings** reach 100% of
expanding trials at all four r/v and 0% of every control. Across those 31, the
angle at the first GF spike **rises with r/v in every one** (e.g. 11.9°, 16.1°,
21.6°, 29.6° for r/v 10, 20, 40, 80 at gain ×2, weight ×4, v_th −47 mV), by a factor
of 2.3 to 5.4 from r/v 10 to 80. That is the size-plus-velocity behaviour the
criterion is named after: at a fixed size a faster loom is steeper
(θ' = 2 sin²(θ/2) / (r/v), pinned in a test), so it crosses threshold earlier.
The plan's wording of criterion 2 ("a consistent angular size") disagrees with
its own title ("size + velocity sum"); the rising trend is what a size-and-velocity
sum predicts, and I believe it is the direction of the published giant-fiber data
but **have not checked it against the papers**, which `docs/validation.md` must do
before claiming it.

## Caveats, all of which bind

- **In-sample (addressed in `docs/lif_fit.md`).** The best-of-60 above is chosen on
  the same 128 trials it is scored on. The Week 2 fit selects on train and scores
  on held-out r/v and azimuths: margin +1.00 on test. It also shows criterion 1
  barely constrains the parameters (206 of 378 settings perfect on train).
- **Selectivity under the angular drive is constructed.** See the decision above.
- **The LIF constants match the Methods text** of Shiu et al. 2024 (checked
  2026-10-05) but have not been checked against its Brian2 code, the plan's "LIF
  cross-check" row. Until then call this "a LIF model in the style of Shiu et al.". One spike's peak PSP is
  ~0.043 mV per synapse, so crossing the 7 mV gap takes ~162 synapses on one spike;
  the giant fiber fires on many LPLC2 spikes together.
- **Monocular (gate and fit runs).** The right eye is stimulated; the 94 left LPLC2 neurons are in
  the graph and driven by nothing, and the giant fiber is bilateral.
- **LC4 is undriven.** Its inputs are lobula types flyvis does not model, so the
  99 LC4 relays only receive what LPLC2 sends them. The plan's "LC4 encodes
  velocity" is not represented; velocity enters only through θ'.
- **No electrical synapses,** and the GF's strongest recorded inputs are not the
  visual detectors (`docs/week0_status.md`), so the chemical path into it is
  incomplete. The fit compensates for a known deficit.
- **Criterion 3 fails** (`docs/lif_fit.md`): every escape is short-mode at every
  r/v and every setting of the three parameters, because DNp103 and the GF are
  co-driven and the GF is never later than the wing-raise window. **Criterion 4
  is met** (`docs/heading.md`): bilateral drive, escape-DN asymmetry points away
  from the stimulus in 48 of 48 lateral trials; front/back is undetermined.
- **Gain rule.** `input_gain` by rule is a provisional number for the gate, not
  the Week 2 fit.
- **One optic-lobe member (000),** the best of 18 measured by direction
  selectivity (`docs/ensemble_dsi.json`).

## Reproducing

```sh
SWATTER_REMOTE=1 .venv/bin/python offline/escape_graph.py                 # 30 s
CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/loom_sweep.py            # ~1 min
CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/loom_sweep.py --drive angular
CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/lif_scan.py [--angular]
make check                                                                # 73 local tests
```

The sweep refuses to start without a visible GPU unless `--cpu` is passed. An
unplanned CPU fallback took six minutes and most of a shared host's cores.
