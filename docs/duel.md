# Duel mode: the fly's brain against a second copy of itself

`?mode=duel` (the Duel tab). Nobody plays. Brain A is the fly, exactly as in Classic. Brain B is a second, independent
instance of the same 495-neuron circuit (`FlyBrain.create()` instantiates its own WebAssembly module), and it flies the
swatter. Code: `web/game/duel.ts`; tests: `duel.test.ts`; fit: `duel_tune.ts` → `docs/duel_tune.json`.

## How B is wired

B's output is turned into the cursor stream (`Input`) that `Round` already takes, so the rules, the stage, the sound
and the fly's side are the ordinary game. A test replays B's cursor stream through `simulateRound` and gets the same
round.

| | Real (the circuit) | Ours (the mapping) |
| --- | --- | --- |
| What B sees | the looming detectors, relays and escape neurons, with their wiring | two forward-facing eyes, each looking 25° to its own side; the fly as a disc of radius 10 mm seen from the swatter's height; the detectors are driven by the disc's growth rate, as in the fly's own model |
| Sensitivity | | `gainB`: B's growth rate is multiplied by it before it drives the engine (the drive is linear in the rate, so this is the engine's input gain) |
| Moving | | a fixed creep of `creepMmPerMs`; heading turns toward the eye whose looming detectors carry more voltage |
| Striking | B's escape neurons spiking, by the same takeoff rule as the fly (`decideTakeoff`) | the swatter's "jump" is the strike: the button is held from the start and released 6.87 ms after the decisive spike. If B has not struck by 2.5× the time the approach should take, or is within 6 mm of the fly, it strikes anyway ("ran out of road") |

Nothing here is biology. It is a way to let a second copy of the circuit make the one decision a swatter has to make.

## What the first run found

- The fly's detector is almost blind to a *static* swatter and to a very slow one: a creep of 0.02 mm/ms never spooks it
  (16 of 16 hits when B just walks in); at 0.03 it spooks 6 of 16; at 0.1 it spooks 11 of 16, from over 100 mm away.
- B's strike distance rises steeply with `gainB`. With a 0.03 mm/ms creep it never fires below about 3; at 5 it fires from a median 66 mm out; at
  12, from about 110 mm, and always strikes too early. The swatter's footprint is a 50 mm half-width and the
  strike falls straight down, so a strike from much farther out misses. The useful range of `gainB` is narrow.
- The fly's eyes look sideways, so a swatter approaching from ahead or behind is seen later than one coming from the
  side. That, not noise, is what makes some duels go one way and some the other. It is the assumed eye geometry
  (`docs/game.md`) showing through, not a measured blind spot.

## Fitted values (`DUEL`)

`gainB` 3, creep 0.035 mm/ms, start 160 mm out and up to 40° off the fly, turn rate 0.004 rad/ms at full eye difference.
The grid search (`duel_tune.ts`: `gainB` 2–4.5 by 0.25, creep 0.02–0.05 by 0.005, 32 seeds each) aimed for a mixed
result, not a fixed win rate. On the 32 fit seeds: 41% hits (34% on B's own spikes), 44% the fly got away, 16% B struck too
early. On 64 fresh seeds: 39% hits (25% on B's spikes), 50% the fly got away, 11% too early. About a third of hits are B
reaching the fly without its circuit firing first, so the tuning is not as brain-driven as it could be.

The whole search takes about two minutes on one laptop core (a duel is about 3 s of game time, ~40 ms to simulate), so it
did not need the DGX. `gainB` is sensitive: at 0.03 mm/ms the hit rate by B's spikes goes from 56% at 3.25 to 3% at 4.

## Page behaviour

The speed is automatic by default: fast while the swatter is far away, slower as it closes in, and 0.1× once either
brain's escape neurons have fired, so both decisions can be watched. The speed menu can fix it at 0.1×, 1× or 4×.
Duels follow on by themselves; a seed replays the same way. The voltage strips show the last 600 ms.

## Not done

- No human has watched or judged these duels, and the sound has not been heard.
- The phone layout is stacked (stage, fly's brain, swatter's brain) and has not been checked on a device.
- B's steering uses the eye-voltage difference; it was not compared with other signals.
