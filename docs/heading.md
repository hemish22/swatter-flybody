# Criterion 4: takeoff heading from left/right DN asymmetry

Reproduce: `offline/heading.py` (one GPU, about a minute); raw trials in
`docs/heading.json`, log in `logs/heading.log`. Parameters are the three chosen by
`offline/lif_fit.py`; nothing else is fitted.

## What was run

Both eyes driven (the first bilateral run in this repo: the gate and fit runs
stimulate the right eye only). Expanding looms at r/v 10, 20, 40, 80 ms, in the
horizontal plane at 16 body azimuths (the plan's 8 and 8 in between), 0 = ahead,
+90 = the fly's right. The readout is the escape DNs (giant fiber, DNp04, DNp103,
DNp02, DNp11, both sides): spike counts in the 20 ms from the first of them to
spike, A = (R - L) / (R + L). A > 0 means the right side is stronger, so the
heading is biased left, away from the stimulus.

Assumptions (listed in the module docstring): eye axes point laterally; the left eye
is the mirror of the right but keeps its own measured LPLC2 receptive fields; which
plane direction is anterior is unpinned (it swaps front and back, not left and
right).

## Result

- **Lateral stimuli** (|sin az| >= sin 22.5 deg, 48 trials where a DN fired): the
  stronger side is the stimulus side, hence heading away, in **48 of 48**; the first
  DN to spike is ipsilateral in 48 of 48. Mean |A| = 0.90.
- **Ahead and behind** (az 0 and 180): no side is preferred; mean |A| = 0.16, and in 2
  of 8 trials no DN fired at all. Left/right asymmetry cannot say whether the fly
  turns forward or backward, so this rule gives no front/back component, and the
  game would have to pick a rule for it, labelled as one.
- **Criterion 4 as the plan states it ("heading biased away from the stimulus") is
  met** for every lateral azimuth at every r/v tested.

## What this does and does not establish

- The ipsilateral dominance comes from the connectome (right LPLC2 to right DN
  weights), not from the construction: the drive is mirror-symmetric, so a
  bilaterally wired circuit would give the same on both sides, but the DN counts
  would not be one-sided if LPLC2 projected contralaterally.
- It is a left/right sign with a magnitude, not a heading angle. The size of A is
  nearly binary (counts of 1 to 4 DN spikes per side in the window), so any mapping
  from A to a turn angle would be invented; v1 should use only the side, as the
  plan labels it.
- The right side saturates at +1.00 while the left side is graded (-0.33 to -1.00),
  so the circuit is not perfectly mirror symmetric (91 right and 94 left LPLC2
  neurons in the graph). Not examined further.
- The geometry assumptions above are unverified against real eye axes. The drive is
  the angular-growth one, selective by construction (`docs/week1_gate.md`).
