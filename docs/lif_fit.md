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
- **Provisional, not frozen.** The Brian2 code behind the LIF constants is still
  unchecked (the constants match the paper's Methods text), and criteria 3 and 4
  have not been run. The three parameters are frozen only after those.
