# LIF parameter fit with a held-out set (Week 2, provisional)

Reproduce: `offline/lif_fit.py` (sharded over GPUs, ~4 min on 16 shards; see its
docstring). Raw numbers: `docs/lif_fit.json`. Log: `logs/lif_fit.log`.

## Protocol

- **Drive:** the angular-growth drive of `docs/week1_gate.md`. Selective by
  construction, so a pass shows the connectome circuit and parameters behave
  sensibly, not that the fly's visual system discriminates.
- **Train:** r/v 10, 20, 40, 80 ms x azimuths 0, 45, ..., 315 x 4 stimulus kinds
  (128 trials). `input_gain` is anchored by the existing rule on train expanding
  r/v = 20 and applied unchanged to test.
- **Test (held out):** r/v 14, 28, 57 ms (inside the trained range) and 7, 120 ms
  (outside it), at azimuths 22.5, 67.5, ... No r/v or azimuth is shared with train
  (a test pins this).
- **Grid:** gain x{1 .. 16} (9) x weight x{1 .. 8} (7) x v_th {-48 .. -43} mV (6) =
  378 settings.
- **Rule, fixed before any test number was seen:** take the setting whose whole
  3x3x3 grid neighbourhood has the best worst-case train margin (ties: neighbourhood
  mean, then smaller GF-angle spread). Grid-edge settings are not eligible.

## Result

Chosen: gain x11.3 (`input_gain` 82.86), weight x1.5, v_th -45 mV.

| split | margin | expanding GF fraction | controls | GF angle at first spike (deg) by r/v |
| --- | --- | --- | --- | --- |
| train | +1.00 | 1.0 at every r/v | 0, 0, 0 | 7.8 / 10.2 / 13.2 / 20.6 at 10 / 20 / 40 / 80 |
| test, inside range | +1.00 | 1.0 at 14, 28, 57 | 0, 0, 0 | 8.2 / 12.5 / 15.9 at 14 / 28 / 57 |
| test, outside range | +1.00 | 1.0 at 7, 120 | 0, 0, 0 | 6.6 at 7, 29.4 at 120 |

The single best setting on train (ties are broken arbitrarily by order) is worse
on test: margin +0.88 inside the range, +0.50 outside (it never fires at r/v 7).
That is the reason for the neighbourhood rule.

197 of the 206 settings that were perfect on train were still perfect on held-out
interpolation.

## What this does and does not establish

- **Criterion 1 barely constrains the parameters.** 206 of 378 settings are
  perfect on train. The held-out check shows the choice generalises across r/v and
  azimuth; it does not show the parameters are identified. Anything that picks a
  setting inside that region passes criterion 1.
- **Angle at the GF spike is not constant across r/v, for any setting.** Among the
  206 perfect settings the largest-to-smallest median angle ratio is 2.26 at best
  (median 2.9), and the angle rises with r/v in all 206. That follows from a drive
  proportional to dθ/dt being integrated. Criterion 2 wording ("consistent angular
  size") holds only loosely. Whether the published GF threshold angle also rises
  with r/v has not been checked against the papers.
- **Time to collision at the spike is 147 / 225 / 345 / 440 ms at r/v 10 / 20 /
  40 / 80**: slower looms are answered earlier before collision, as a
  roughly constant-angle trigger would. Measured from stimulus onset the order is
  reversed (333 / 255 / 135 / 40 ms), because every trial has the same collision
  time. The plan's "earlier in time for faster looms" is therefore true only in
  one of the two readings; this repo reports time to collision.
- **Provisional, not frozen.** The Brian2 code behind the LIF constants was checked on
  2026-10-16 (section below): the constants match, but the neuron's behaviour around its own
  spike differs, and that moves spike timing. Criteria 3 and 4 have been run since. The
  parameters stay provisional.

## Brian2 cross-check (2026-10-16)

Script `offline/brian2_check.py`, result `brian2_check.json`, log `logs/brian2_check.log`. Run on the DGX in
tmux (Brian2 2.10.1, numpy target, about 80 s for everything). Reference: `model.py` of Shiu et al. 2024
(github.com/philshiu/Drosophila_brain_model), read on the same day. Same 495-neuron graph, weights, fitted
parameters and per-step drive, through Brian2, against the outcomes `web/brain/parity.json` records for the engine
(50 trials: 42 expanding, 4 controls, 4 edge cases).

**The constants match** (v0 = v_rst = -52 mV, v_th = -45 mV, t_mbr 20 ms, tau 5 ms, t_rfc 2.2 ms, t_dly 1.8 ms,
w_syn 0.275 mV). **Three things about a neuron's own spike do not**, and the engine follows the other side of each:

| | Shiu's Brian2 model | `lif.py` and the engine |
| --- | --- | --- |
| Refractory period | `v` and `g` both frozen (`unless refractory`) | `v` held at reset, `g` keeps decaying and integrating |
| On a spike | `g` is set to 0 | `g` carries on |
| Integration | exact (`method='linear'`), 0.1 ms | forward Euler, 0.2 ms |

Results (GF = giant fiber; "brian minus engine" is the first-spike time difference):

| Brian2 variant | GF fires in the same trials | GF first spike, median and spread | trials > 5 ms apart | mode agrees | side agrees | spikes per trial vs engine |
| --- | --- | --- | --- | --- | --- | --- |
| Shiu's equations, 0.1 ms | 46 of 46, controls silent | +1.0 ms, 5 to 95%: -35 to +4.5 ms | 14 of 46 (12 earlier) | 48 of 50 | 50 of 50 | 0.46 |
| Shiu's equations, 0.2 ms | 46 of 46, controls silent | +0.9 ms, 5 to 95%: -35 to +32 ms | 16 of 46 (13 earlier) | 48 of 50 | 50 of 50 | 0.48 |
| the engine's scheme written in Brian2 (control) | 46 of 46, controls silent | +0.4 ms, 5 to 95%: 0 to +1.1 ms | 2 of 46 | 50 of 50 | 50 of 50 | 1.07 |

The control row says how far the harness itself is from the engine: a steady +0.4 ms (Brian applies its state update
and synaptic events in a slightly different order than the engine's step), and two borderline trials that move by 11
and 22 ms. So differences under about half a millisecond below are the harness, not the neuron.

What this does and does not show:

- **Criterion 1 survives the change of equations.** With the fitted parameters untouched, the giant fiber fires in
  exactly the same 46 trials, the four controls stay silent, and the heading side agrees in all 50.
- **Timing does not survive.** Under Shiu's equations a typical GF spike is 1 to 4 ms later, and in 14 to 16 of
  46 trials it moves by more than 5 ms, mostly earlier (by about 10 to 38 ms, and up to 100 ms or more in a few).
  Total spikes fall to under half. The probable reason is that a neuron whose `g` is reset after each spike fires
  less often, so the circuit's inhibition is weaker and the GF crosses threshold sooner, but that was not isolated.
  Everything quoted in this repo about the angle at the spike, time to collision and reaction time is a property of
  the engine's neuron scheme, not of Shiu's model.
- **Criterion 3 does not change.** Short-mode trials by r/v among the 46 that fire, engine then Shiu's equations
  (0.1 ms): 8 then 7 of 8 at r/v 20, 5 then 5 of 6 at 28, 6 then 5 of 6 at 40, and 6 of 6 either way at 10, 14, 57 and
  80. Two trials flip from short to long, at r/v 20 and 40, and the rest of the table does not move. No trend over r/v
  in either column.
- **The parameters are not refit.** Matching Shiu's equations would mean changing `lif.py`, the Rust engine and
  `engine_ref.py`, re-exporting `web/brain/`, and rerunning the 378-setting fit, and the game would feel different.
  That is not needed for the game to work; it is needed only to call the numbers "Shiu-model" numbers. Left as is,
  and stated.

## Criterion 3 (takeoff mode): not met

Rule: `offline/takeoff.py`, the plan's state machine with W = 6.87 ms, nothing
fitted. Short fraction among expanding trials that escape, per r/v, at the chosen
setting:

| split | short fraction by r/v | parallel-DN first spike minus GF first spike (ms) |
| --- | --- | --- |
| train, r/v 10 / 20 / 40 / 80 | 1.0 / 1.0 / 1.0 / 1.0 | -1.2 / -1.9 / -2.3 / -0.7 |
| test, r/v 14 / 28 / 57 | 1.0 / 1.0 / 1.0 | -1.4 / -1.0 / -2.0 |
| test, r/v 7 / 120 | 1.0 / 1.0 | -0.8 / -0.9 |

Every expanding trial is a short-mode takeoff, and so is every trial in 199 of
the 206 settings that pass criterion 1 (lag range over them: -2.4 to 0.0 ms). The
other 7 vary only because a few trials at r/v 10 do not escape, which changes the
denominator; none shows a trend over r/v, and none fits the plan's direction
better than noise. So no setting of the three parameters satisfies "short-mode
fraction rises as r/v falls".

Why: DNp103, the first parallel DN to fire in every trial, and the giant fiber are
driven by the same LPLC2 population, and DNp103 leads the GF by about 1 to 2 ms at every r/v.
That lag is set by the weight ratio, not by loom speed, and it is far below W. The
other three parallel DNs (DNp04, DNp02, DNp11) fire 50 to 120 ms after the GF, so
choosing them as the "parallel pathway" would make every escape short as well.

What would be needed for the plan's mode behaviour is a mechanism that makes the
GF late for slow looms relative to the wing-raise pathway: LC4's velocity input
(undriven here), the GF's electrical inputs (absent from EM), or a nonlinearity
in the real GF. None is in this graph. The game must not claim mode selection from
the connectome; options are listed in `docs/week1_gate.md`.

### Addendum: criterion 3 in the bilateral body-frame geometry

The table above is the lab suite (right eye, 40 deg off axis, azimuths 0 to 315 in the eye plane). The
same question under `heading.py`'s bilateral geometry (16 body azimuths, both eyes, r/v 10 to 80 plus 14,
28, 57), at the chosen setting, for lateral stimuli (14 per r/v that fire): short fraction 0.93 / 0.86 /
0.86 / 0.79 / 0.93 / 0.93 / 0.93 at r/v 10 / 14 / 20 / 28 / 40 / 57 / 80. There are long-mode trials here
(11 of 98), but they sit at azimuths 22.5, 315 and 337.5, near the front of the field where the GF is late or
silent, and they do not vary with r/v in either direction. So the conclusion stands: **no short-fraction
trend with r/v**; what sets long mode in this model is where the stimulus lands, which the plan does not
claim. (Scratch analysis, not in a committed script.)
