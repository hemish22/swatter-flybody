// Leaderboard verification: replay every swat of a Classic run from its seed and cursor trace, with the same
// world.ts and the same WASM engine the browser plays, and accept the score only if the result matches the claim.
// Compared fields are the plan's: outcome, takeoff mode and heading side. Never raw floats.
import type { FlyBrain } from "../web/engine/brain.ts";
import { decodeTrace, runSeeds } from "../web/game/trace.ts";
import { simulateRound, type Card } from "../web/game/world.ts";

export const CLASSIC_SWATS = 20;
/** 12,000 ticks is the 60 s hover timeout; a whole run may not exceed 80,000 ticks (400 s of game time), which bounds server work to a few seconds. */
export const MAX_TICKS_PER_SWAT = 12_001;
export const MAX_TICKS_PER_RUN = 80_000;

export interface ClaimedCard { outcome: string; mode: string; headingSide: string | null }
export interface SwatSubmission { trace: unknown; claimed: ClaimedCard }
export type Verdict =
  | { ok: true; hits: number; cards: Card[] }
  | { ok: false; reason: string; swat?: number };

export function verifyRun(brain: FlyBrain, baseSeed: number, swats: readonly SwatSubmission[]): Verdict {
  if (!Array.isArray(swats) || swats.length !== CLASSIC_SWATS) return { ok: false, reason: `a Classic run has ${CLASSIC_SWATS} swats` };
  const seeds = runSeeds(baseSeed, CLASSIC_SWATS);
  const cards: Card[] = [];
  let ticks = 0, hits = 0;
  for (let k = 0; k < swats.length; k++) {
    const s = swats[k];
    const trace = decodeTrace(s?.trace);
    if (!trace || trace.length === 0) return { ok: false, reason: "malformed trace", swat: k };
    if (trace.length > MAX_TICKS_PER_SWAT) return { ok: false, reason: "trace too long", swat: k };
    ticks += trace.length;
    if (ticks > MAX_TICKS_PER_RUN) return { ok: false, reason: "run too long", swat: k };
    const round = simulateRound(brain, seeds[k], trace);
    if (round.phase !== "done" || !round.card) return { ok: false, reason: "trace does not finish the round", swat: k };
    if (trace.length !== round.tickIndex + 1) return { ok: false, reason: "trace has ticks after the round ended", swat: k };
    const c = round.card, claim = s.claimed;
    if (!claim || claim.outcome !== c.outcome || claim.mode !== c.mode || claim.headingSide !== c.headingSide)
      return { ok: false, reason: `claimed ${JSON.stringify(claim)} but the replay gives ${c.outcome}/${c.mode}/${c.headingSide}`, swat: k };
    if (c.outcome === "hit") hits++;
    cards.push(c);
  }
  return { ok: true, hits, cards };
}
