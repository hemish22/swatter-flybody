# Game world model (web/game) and what the frozen brain does to it

Status: the world model, a deterministic round simulator and a skill analysis exist and are tested
under Node against the real engine. There is no renderer, cursor handling, UI or overlay yet.

## The finding that shaped the rules

Under the plan's rules the fly cannot be hit. The plan sets a 150 mm hover and takes strike speed from
the cursor's speed over the last 100 ms. With the frozen brain:

- The fly leaves 11 to 25 ms after the swatter starts to grow in its eyes (giant-fiber latency, plus the
  plan's 6.87 ms for a short takeoff). A strike from 150 mm at 5 m/s lasts 30 ms, so the fly is already gone
  at every speed in the plan's range. From a still hover, 24 of 24 strikes at every speed were escaped.
- A flick is a loom. To measure 3 m/s over 100 ms the cursor must travel 300 mm, and the swatter's approach
  over that stretch is what the fly sees, about 100 ms before the strike begins. Of the flicks tried
  (24 seeds, three approach directions, five flick lengths, hover heights 20, 40 and 150 mm) none produced a
  hit; the fly left during the approach or before landing. Pinned in
  `world.test.ts` ("the plan's own rules ... leave no strike able to land").

So the rules below are my decision, taken under "do whatever is best" and not the plan's. They are the
smallest change that gives a skill game, and they are easy to reverse (`CONFIG` in `world.ts`; the plan's
rules are still there as `strikeSpeedSource: "cursor"`).

| | Plan | Default here | Why |
| --- | --- | --- | --- |
| Hover height | 150 mm, "tunable" | **40 mm** | A strike has to cover the whole descent before the fly's 11 to 25 ms is up. 150 mm cannot, at 5 m/s. Hover height is one of the plan's four legal difficulty knobs. |
| Strike speed | cursor speed over the last 100 ms | **hold the button to charge, release to strike**, 0.6 to 5 m/s over a 1 s hold | The flick is detected before the strike. Charging keeps the plan's 0.6 to 5 m/s range, so r/v still spans 10 to 83 ms, the lab range. |

Both are stimulus-side choices; nothing touches the brain, as the plan requires.

## What the world does

Pure and deterministic: a round is a function of `(seed, cursor trace)`, with no clock and no
`Math.random`, so the leaderboard re-sim runs the same code. One tick is 5 ms.

- Fly: random position and heading from a seeded mulberry32. Eye 1 mm above the table.
- Swatter: a disc of half-width 50 mm. Hover phase: it tracks the cursor at the hover height. Strike phase:
  vertical descent at the charged speed, from the point where the button was released. The landing time is
  rounded to the engine's 0.2 ms step, so a strike's r/v is reported exactly as simulated.
- Frames: each tick the world computes the swatter's angular size 2 atan(r/D), its analytic growth rate (no
  finite-difference lag) and its position on each eye's plane, and hands them to the engine. The geometry is
  the same as `offline/heading.py`'s, pinned to it by `fixtures.json` (made by `offline/game_fixtures.py`).
- Takeoff: the plan's mode rule on the engine's spikes (`decideTakeoff`). Short: the giant fiber spikes
  first or within 6.87 ms of the parallel DNs, and the fly leaves 6.87 ms after the spike. Long: the raise
  completes without a giant-fiber spike, and the fly leaves at the end of the raise. Both 6.87 ms figures are
  the plan's single number for a short takeoff, not measurements of leg-contact loss (unverified), and they
  make the game slightly easier than a faster takeoff would.
- Outcome at landing: **hit** if the fly is inside the 50 mm footprint and has not left; **escaped** if it is
  inside and had left; **miss** if it was outside; **spooked** if it left during the hover; **timeout** after
  60 s. The card carries strike r/v, the swatter's angular size when the giant fiber spiked, takeoff mode,
  margin (landing time minus the time it left) and heading side.

## Skill curves (`web/game/difficulty.ts`, 12 seeds, raw numbers in `docs/game_difficulty.json`)

Hit rate against strike speed, cursor still above the fly:

| strike speed | 0.6 | 1 | 1.5 | 2 | 3 | 4 | 5 m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r/v | 83 | 50 | 33 | 25 | 17 | 13 | 10 ms |
| hits of 12 | 0 | 0 | 0 | 0 | 4 | 12 | 12 |
| median margin (ms) | +51 | +26 | +13 | +7 | +0.5 | -2.7 | -4.5 |

The switch from escape to hit sits near 3 m/s (r/v about 17 ms), where the descent equals the fly's reaction
time.

Creeping up unnoticed (cursor moving toward the fly from 300 mm, then a full charge). The fastest creep speed
at which all 12 rounds still ended in a hit:

| approach from | ahead (0) | right-front (45) | right (90) | behind (180) | other sides |
| --- | --- | --- | --- | --- | --- |
| m/s | 0.2 | 0.03 | 0.03 | 0.05 | 0.02 |

Dead ahead is the fly's blind spot in this model, ten times more forgiving than the sides. That comes from
the lateral eye-axis assumption (front and back sit at the edge of both eyes' fields), so it is a prediction of
the model as built, not something measured in flies.

## Limits to keep in mind

- **The response is a switch.** The circuit is deterministic with no noise, so hit rate against speed (and
  against creep speed) steps from 0 to 1 over a narrow range instead of sloping. The "fly escapes most
  swats but not all" target is met only for skilled play, and a gentler curve would need noise or a
  stochastic escape, which the brain does not have and the game must not invent (plan: no knob touches the
  brain).
- Creep speeds of 0.02 to 0.05 m/s are slow: crossing 300 mm takes 6 to 15 s. How that feels depends on the
  arena's pixel scale, which is not decided.
- Hold-to-charge and 40 mm hover are untested with a human. The plan's cursor rule may still be preferred if
  the stimulus mapping is changed some other way (a much smaller swatter, say); that is a product decision.
- Eye axes, eye height and the takeoff delay are assumptions (see `world.ts`).
- Heading is a side only (`docs/heading.md`); the game can show which side the fly turns to, not an angle.
