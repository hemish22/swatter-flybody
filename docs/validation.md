# Validation

Not written yet. Target: Week 2 (Oct 12–18), before the Week 3 engine work.

Everything here has to be reproducible from `offline/loom_sweep.py` on the DGX,
and every plot has to name the paper figure it is being compared against. The
point of this document is that a sceptic can check the fly without trusting the
game.

## Plots

| # | Plot | Criterion | Reference |
| --- | --- | --- | --- |
| 1 | GF voltage over time for r/v = 10, 20, 40, 80 ms, plus angular size at GF spike vs r/v | 2 | Ache et al. 2019 |
| 2 | GF spike probability: expanding vs receding vs dimming vs translating | 1 | Ache et al. 2019 |
| 3 | Short-mode fraction vs r/v | 3 | von Reyn et al. 2014, 2017 |
| 4 | Polar histogram of takeoff heading relative to stimulus azimuth | 4 | von Reyn et al. 2014 |
| 5 | Escape probability vs human strike r/v, from game logs | game target | — |

Plot 1 carries the most weight and has the most freedom to be cherry-picked, so
it needs both halves on one figure: the voltage traces *and* the spike angle
against r/v. A trace without the angle says nothing, because the same trace
looks different depending on how the loom was parameterised.

Plot 5 is the headline and the only one that depends on humans. It should show
individual sessions as well as the aggregate, or one good run will carry the
curve.

## Ablations

Each one mirrors a published genetic-silencing result, so the simulated fly is
tested the way real flies were tested rather than by our own criteria.

| Condition | Published result | Expected here | Status |
| --- | --- | --- | --- |
| LPLC2 silenced | Fewer GF-mediated escapes (Ache et al. 2019) | Short-mode fraction drops | not run |
| GF silenced | Flies caught more by damselflies (Chai et al. 2025) | Higher human hit rate, fast strikes most | not run |
| Shuffled LPLC2/LC4 → GF weights, degree-preserving | none | Looming selectivity collapses | not run |
| Distance-threshold bot replacing the brain | baseline only | Escapes depend on distance alone, not strike speed | not run |

The last row is the control for the whole project: it is what a hand-coded
version of this game would do, and SWATTER's fly has to differ from it. If the
escape circuit and the threshold bot produce the same hit rate, the neural
simulation is not doing the work and that has to be said plainly.

The first two ship as game modes, so their gap in human hit rate becomes a
result from real play rather than a number in a table.

## Limitations to state

These are already known and belong here as well as in the UI:

- Point-neuron LIF. Real neurons are not point neurons.
- No electrical synapses. This one matters most: the GF's fastest inputs are
  electrical and EM records them poorly, which is why its strongest recorded
  inputs are DNp70, PVLP010 and SAD neurons rather than the visual detectors.
  The chemical path into the GF is incomplete, so the Week 1 fit compensates for
  a known deficit.
- Heading from DN left/right asymmetry, not pre-takeoff leg posture. The plan
  labels this a simplification and so does the UI.
- LPLC2 and LC4 are left/right asymmetric in MaleCNS (94/91 and 71/55 bodies),
  from incomplete reconstruction. Criterion 4 needs the two sides normalised or
  it will partly measure reconstruction bias.
- Whatever fallback the Week 1 gate chooses, if it chooses one.
