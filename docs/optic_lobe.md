# Optic lobe — Week 1 status

Dates: Oct 5, 2026. Everything here was run on the DGX (`dgxa100`), flyvis
ensemble member `flow/0000/000`, GPU 7 for the sweeps.

Week 0's flyvis item was "port running". What exists now:

| Piece | File | State |
| --- | --- | --- |
| Eye: 721 real columns, loom rendered in 2-D | `offline/eye.py` | done, 18 local tests |
| Optic lobe behind one interface | `offline/optic_wrapper.py` | `flyvis` backend done; `malecns` backend not built (see below) |
| MaleCNS columns, lattice orientation, LPLC2 ← T4/T5 | `offline/malecns_ol.py` | done, both eyes |
| Gate experiment | `offline/optic_gate.py`, `offline/ensemble_dsi.py` | done; **verdict below is Partial, not Go** |

## The result that matters

The optic lobe carries a clean, direction-selective motion signal, and the
MaleCNS LPLC2 wiring is radial on top of it. But **neither of two readouts
separates an expanding disc from a translating one**, so the plan's "Go"
condition (silent for receding, dimming *and translating*) is not met. This is
the plan's **Partial** row, and the radial-motion-template fallback it names was
tried first and did not rescue it either (history below). Nothing has been tuned
to make a control pass or fail.

`docs/optic_gate.json`, `logs/optic_gate*.log`. Readout = most-driven of the 91
right-eye LPLC2 neurons, median over 8 azimuths, loom at 40° eccentricity,
400 ms, 5 ms step. Threshold is half the median expanding peak at r/v = 20,
fixed for every condition and r/v.

LPLC2 linear drive (synapse-weighted sum of T4/T5 rise):

| r/v (ms) | expanding | receding | translating | dimming | expanding ÷ best control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 27.1 | 1.6 | 37.8 | 16.6 | 0.72 |
| 20 | 85.9 | 5.5 | 112 | 50.3 | 0.77 |
| 40 | 217 | 17.5 | 240 | 117 | 0.90 |
| 80 | 363 | 42.1 | 385 | 166 | 0.94 |

Same wiring, four-direction coincidence (geometric mean of the a/b/c/d group
drives per neuron):

| r/v (ms) | expanding | receding | translating | dimming | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 5.18 | 0.23 | 7.79 | 2.39 | 0.67 |
| 20 | 17.5 | 0.78 | 23.3 | 8.29 | 0.75 |
| 40 | 45.2 | 2.84 | 50.4 | 22.2 | 0.90 |
| 80 | 80.2 | 8.08 | 73.8 | 29.6 | 1.09 |

What passes and what does not:

- **Receding: rejected**, 4–12% of expanding at every r/v, under both readouts.
- **Dimming: partly rejected**, 37–61% of expanding. Not silent.
- **Translating: not rejected.** Above expanding at every r/v under the linear
  readout, and for r/v ≤ 40 under the coincidence readout (0.92× at r/v = 80).
- **Criterion 2 precursor:** criterion 2 wants a roughly constant angular size
  at threshold across r/v. Linear readout: 17.5°, 18.0°, 25.8° for r/v 20, 40,
  80, and the r/v = 10 loom never crosses. Two of three agree; the slowest loom
  needs a disc half as large again, so this is not met either (the matching
  crossing times, 350, 228 and 130 ms on the trial clock, are just
  r/v / tan(θ/2) before collision and carry no extra information).

A caveat that cuts the other way: `loom.py`'s translating control moves a 14°
disc through 80° in 400 ms (200°/s), which is a near-optimal stimulus for
T4/T5. That is a choice of the control's speed, not a fact about the fly, and
whether 200°/s is the right "lateral translation" is the user's call, not
something to tune until the table looks right.

### Two follow-ups that did not move it

**LPi inhibition.** MaleCNS has the inhibitory two-hop path T4/T5 → LPi → LPLC2
(375 right LPi cells, all predicted glutamate or GABA; 9,061 synapses onto the
91 LPLC2 neurons against 63,742 excitatory T4/T5 synapses; 464,852 T4/T5
synapses onto LPi). Each LPi synapse subtracts the mean rise of its cell's T4/T5
input. Expanding ÷ best control, r/v 10/20/40/80:

| LPi gain | ratio |
| ---: | --- |
| 0 (linear) | 0.72 / 0.77 / 0.90 / 0.94 |
| 0.5 | 0.72 / 0.78 / 0.91 / 0.94 |
| **1 (verdict)** | 0.71 / 0.80 / 0.91 / 0.95 |
| 2 | 0.70 / 0.80 / 0.93 / 0.96 |
| 4 | 0.68 / 0.80 / 0.97 / 0.97 |

Gain is not fitted (the glutamate-versus-acetylcholine efficacy per synapse is
not in the connectome); the sweep is sensitivity, and the verdict is read at 1.
LPi cost translating and expanding about the same, because the same T4/T5
motion feeds both the excitation and the inhibition. The inhibitory synapses
are only 14% of the excitatory count.

**Translating speed.** `loom.py`'s control is 200°/s. Peak drive of the
translating disc against its speed (linear readout), with expanding for scale:

| Translating (°/s) | 25 | 50 | 100 | 200 | expanding r/v = 20 | expanding r/v = 40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| peak drive | 66 | 91.5 | 113 | 112 | 85.9 | 217 |

A 14° disc that moves only 10° in 400 ms already drives LPLC2 at 77% of an
r/v = 20 loom, because that loom's edges also travel ~11° in the same time. So
200°/s is not what is hurting: loom and a slow drift carry comparable motion
energy and the wiring cannot tell them apart. The control stays at 200°/s, the
plan's lateral-translation test; the sweep is there so the choice can be judged
rather than tuned.

### What the Partial verdict does and does not say

It says a **point-neuron LPLC2 fed by this optic lobe will not separate
translation from looming**: any monotone function of the linear drive (which is
all a LIF threshold is) cannot reject a stimulus that drives as hard as the loom
it is meant to be compared with. It does not say the fly's LPLC2 cannot; the
real cell has lobula-plate inhibition (LPi) and supralinear dendritic
integration that neither readout models. The honest options at the gate are in
`docs/build-plan.md`: label a radial-motion component as a model component (the
Partial row), or add LPi-style inhibition from MaleCNS and test that first.

## Pieces and what each one measured

### Eye and renderer (`eye.py`)

721 columns in flyvis's `(u, v)` order, nearest neighbours 5.8° apart (flyvis's
inter-ommatidial angle), outermost columns 87° off axis. Each column samples the
scene through a Gaussian of FWHM 5.8°, as a 37-point supersample, so a 4.8°
disc is dim rather than missing. The plane is flyvis's own `hex_to_pixel`
embedding, rescaled; a test pins it, because the T4/T5 preferred directions are
angles in that plane.

`loom.EyeGeometry` (the old 1-D azimuth strip) is untouched and still used by
`loom.py`'s own summary; `eye.render_loom` is the 2-D path the optic lobe uses.
The lab loom's `azimuth_deg` is the position angle around the optical axis at
40° eccentricity. `Loom.elevation_deg` is ignored (its default of 90° is a pole
where azimuth means nothing).

### Optic lobe wrapper (`optic_wrapper.py`)

`OpticLobe(backend="flyvis", member, dt_s)`: `run(movie)` for sweeps,
`reset/step/read` for streaming (the two agree to 5e-7), per-type activity as
`(batch, time, 721)` with NaN where a type is thinned. 45,669 cells; a 128-trial
sweep takes under a minute on one A100.

**Integration step.** flyvis is explicit Euler with a caller-chosen step; it
warns above 20 ms (the training step) and its fastest time constant is 19 ms,
so 5 ms is stable. T4/T5 responses to one r/v = 20 ms loom, relative RMS against
the 5 ms run: **10 ms 4.2%, 20 ms 12.3%.** The LIF stages therefore interpolate
a 5 ms optic lobe, which closes the plan's open question. The pretrained weights
were fitted at 20 ms, so 5 ms is a modelling choice with a measured cost, not a
more correct answer.

**Settling matters more than anything else in the gate.** `Network.steady_state`
keeps every intermediate state and does not fit on a 40 GB A100 at batch 32;
`_settle` keeps the last. More important, trials are settled on their *own first
frame*. Settling on the bare sky made every disc already on screen at t=0 arrive
as an onset flash: dimming and translating scored above expanding, and receding scored several
times higher than it does now (hand template, v2 → v3). A regression test pins it.

### Calibration: is the renderer's lattice the right way round?

A bar swept through 8 directions gives each T4/T5 type a measured preferred
direction. It matches flyvis's own `moving_edge_responses` on the same network:

| Type | Nominal | flyvis protocol | this renderer | DSI (flyvis) |
| --- | ---: | ---: | ---: | ---: |
| T4a | 180 | 175.7 | 177.7 | 0.51 |
| T4b | 0 | 351.4 | 352.4 | 0.52 |
| T4c | 90 | 58.8 | 58.6 | 0.59 |
| T4d | 270 | 270.4 | 65.4 | **0.14** |
| T5a | 180 | 185.8 | 174.6 | 0.78 |
| T5b | 0 | 351.1 | 348.9 | 0.78 |
| T5c | 90 | 103.2 | 103.9 | 0.56 |
| T5d | 270 | 251.8 | 253.9 | 0.57 |

T4c is ~30° off nominal and T4d is barely direction-selective in member 000
under flyvis's *own* protocol, so those are properties of the network, not of
the renderer. Any template built from the nominal angles points at the wrong
thing for T4c and T4d.

### MaleCNS columns and orientation (`malecns_ol.py`)

| | Right | Left |
| --- | ---: | ---: |
| typed OL neurons | 49,677 | 49,475 |
| column-tagged by MaleCNS | 13,267 | 10,453 |
| column inferred from tagged partners | 29,525 | 29,798 |
| **hold-out** exact / within 1 / within 2 | 0.958 / 0.986 / 1.000 | 0.998 / 0.999 / 1.000 |
| T4/T5 with a column | 6,793 / 6,793 | 6,784 / 6,787 |

T4/T5 carry no tag, so their column is the synapse-weighted median of their
tagged partners', and the hold-out above is that same procedure applied to
tagged neurons. The weak spot is Tm4 on the right (36% exact), a MaleCNS tag
problem the fly-brain port also reports (its ISS-0001), not an artefact here.

Orientation: M = (−1, 0, 1, −1) on both sides, origin at the centroid of the Mi1
sheet (hex 19, 20). **The cosine fit alone is a thin margin** (0.895 against
0.889 for the runner-up on the right, 0.905 against 0.898 on the left, worst
candidate 0.82–0.85). What fixes it is the direction-selective inputs: Mi9→T4a–d
and Tm9→T5a–d offset centroids all point the same way as flyvis's after M, which
also confirms the a–d labels agree between the two connectomes. Both eyes
choosing the same M independently is further support. The origin is a choice
(the optical axis is not in the data) and is recorded in the output JSON.

### LPLC2 ← T4/T5, and the check that the wiring is radial

91 right / 94 left LPLC2 neurons; 63,742 / 49,199 synapses from T4/T5, of which
90.9% / 90.8% land inside the 721-column hexagon. Channel input centroids
relative to each neuron's receptive-field centre, right eye, eye-plane degrees:

| Channel | Offset (x, y) | Measured preferred direction |
| --- | --- | --- |
| T4a / T5a | (−15, 0) | 178° / 175° (−x) |
| T4b / T5b | (+13, −1…−2) | 352° / 349° (+x) |
| T4c / T5c | (0, +11…13) | 59° / 104° (+y) |
| T4d / T5d | (0, −14) | n/a (T4d untuned) / 254° (−y) |

Each channel is read from the side of the receptive field that its preferred
direction points to, which is outward motion on every side: the wiring of a looming detector, found in the
connectome without being put there, in the lattice orientation chosen above. The
left eye gives the same picture.

## History of the gate (`logs/optic_gate_v*.log`, local only)

| Run | Readout | Settling | Expanding ÷ best control (r/v 10/20/40/80) |
| --- | --- | --- | --- |
| v1 | hand template, geometric mean of rectified T4/T5 | sky | 1.21 / 1.16 / 0.86 / 0.71 |
| v2 | + motion opponency (a−b, c−d) | sky | 0.95 / 0.82 / 0.67 / 0.86 |
| v3 | same | **first frame** | 0.75 / 0.73 / 0.68 / 0.76 |
| v4 | signed divergence, 4-quadrant AND | first frame | 0.62 / 0.85 / 0.76 / 0.74 |
| v5 | **MaleCNS LPLC2 wiring, linear** | first frame | 0.72 / 0.77 / 0.90 / 0.94 |
| v6 | MaleCNS LPLC2 wiring, 4-group AND | first frame | 0.67 / 0.75 / 0.90 / 1.09 |

v1→v3 fixed receding and dimming; v3→v6 never fixed translating. The hand
template's wake (the ON trailing edge behind a translating dark disc drives T4
strongly, and the response outlasts the disc by ~40°) was visible in a plotted
motion field and is probably the main thing a real LPLC2 does not read.

## Not done, on purpose

- **The `malecns` backend** (flyvis *dynamics* on MaleCNS wiring). The
  reference port measured direction selectivity 0.152 against flyvis's 0.391,
  an OFF pathway that is spatially broken (its ISS-0015), and missing CT1/Am
  wiring, and needs a neuPrint token for its column ROIs. What the game needs
  from MaleCNS in the optic lobe, the LPLC2 input wiring, is done above without
  it. Run the dynamics on flyvis's own wiring until a reason to switch exists.
- **LC4.** Its inputs are lobula types flyvis does not model.
- **LPi / lobula-plate inhibition** and the LIF stage itself.
- **Other ensemble members.** Member 000 is the right one: ranked by T4/T5
  direction selectivity under flyvis's own moving-edge protocol (a criterion
  that never looked at the loom gate), it is far ahead. Of 18 members measured
  (`docs/ensemble_dsi.json`, `offline/ensemble_dsi.py`), 000 has a minimum DSI of
  0.14 and a mean of 0.56; no other member has a minimum above 0.04 or a mean
  above 0.41, and most have several types near zero. So the weak T4d is the best
  this ensemble offers, and the translating result is not a bad-member artefact.
  Ensemble spread of the gate itself is still not measured.

## Decisions taken at the gate

Delegated by the user on 2026-10-05, made for these reasons:

1. **Verdict: Partial.** Receding and dimming are handled; translating is not,
   under four independent readouts (hand template, linear, four-direction AND,
   LPi-inhibited), the best optic-lobe member, and any LPi gain from 0 to 4.
2. **The translating control stays at 200°/s** and stays pass/fail. The speed
   sweep shows slowing it does not rescue the result, so there is nothing to
   gain by touching it.
3. **No `malecns` dynamics backend.** The needed MaleCNS content (LPLC2 and LPi
   wiring) is in; the reference port's own dynamics are weaker than flyvis's.
4. **Next stage uses the Partial path the plan names**, with the limitation
   written down rather than papered over: the LIF stage takes LPLC2 input from
   this T4/T5 drive, and `docs/validation.md` must state that translation is not
   separated by the wiring and what the game does about it. The honest game-level
   consequence is that a swatter drifting sideways at hover height can spook the
   fly; that is part of the stimulus the player controls (the plan's hover phase
   already says approach alone can spook it), and the post-round card can show it.

## Reproducing

On the DGX (this host, so no `dgx.sh`):

```sh
make weights                                     # flyvis pretrained ensemble, CPU
SWATTER_REMOTE=1 .venv/bin/python offline/malecns_ol.py --side R   # 30 s
SWATTER_REMOTE=1 .venv/bin/python offline/malecns_ol.py --side L
CUDA_VISIBLE_DEVICES=<free gpu> SWATTER_REMOTE=1 .venv/bin/python offline/optic_gate.py --dt-check
make check                                       # local: 55 tests
make test-dgx                                    # real network, CPU, ~1 min
```
