// Scripted swats for tests: a valid trace for a seed that ends the way the test wants.
import type { FlyBrain } from "../web/engine/brain.ts";
import { encodeTrace, quantise } from "../web/game/trace.ts";
import { Round, type Input } from "../web/game/world.ts";
import type { SwatSubmission } from "./verify.ts";

/** Hover 40 ticks 20 mm beside the fly, hold `holdTicks`, release, then wait for the landing; trimmed to the ticks the round used. */
export function scriptedSwat(brain: FlyBrain, seed: number, holdTicks: number): SwatSubmission {
  const fly = new Round(brain, seed).fly;
  const at = { x: fly.x + 20, y: fly.y };
  const plan: Input[] = [];
  for (let i = 0; i < 40; i++) plan.push({ ...at, down: false });
  for (let i = 0; i < holdTicks; i++) plan.push({ ...at, down: true });
  for (let i = 0; i < 400; i++) plan.push({ ...at, down: false });
  const round = new Round(brain, seed), used: Input[] = [];
  for (const p of plan) {
    const q = quantise(p);
    used.push(q);
    round.tick(q);
    if (round.phase === "done") break;
  }
  const c = round.card!;
  return { trace: encodeTrace(used), claimed: { outcome: c.outcome, mode: c.mode, headingSide: c.headingSide } };
}
