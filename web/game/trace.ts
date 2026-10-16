// Cursor traces on the wire, and the seed chain a run uses. Shared by the browser and the leaderboard server so that
// both quantise, decode and derive seeds identically; the server's verdict only means something if they do.
import { CONFIG, type Input } from "./world.ts";

/** Cursor positions are kept to 0.1 mm. The browser feeds the round the quantised value, so what the server replays is what was played. */
export const quantise = (i: Input): Input => ({ x: Math.round(i.x * 10) / 10, y: Math.round(i.y * 10) / 10, down: i.down });

/** Flat integers [x10, y10, down, ...]: JSON-friendly and lossless for quantised inputs. */
export function encodeTrace(trace: readonly Input[]): number[] {
  const out: number[] = [];
  for (const i of trace) out.push(Math.round(i.x * 10), Math.round(i.y * 10), i.down ? 1 : 0);
  return out;
}

/** Null if the array is not a well-formed trace (wrong length, non-integers, outside the arena, bad button flag). */
export function decodeTrace(flat: unknown): Input[] | null {
  if (!Array.isArray(flat) || flat.length % 3 !== 0) return null;
  const out: Input[] = [];
  const hx = (CONFIG.arenaMm.w / 2) * 10, hy = (CONFIG.arenaMm.h / 2) * 10;
  for (let k = 0; k < flat.length; k += 3) {
    const [x, y, d] = [flat[k], flat[k + 1], flat[k + 2]];
    if (!Number.isInteger(x) || !Number.isInteger(y) || (d !== 0 && d !== 1)) return null;
    if (Math.abs(x) > hx || Math.abs(y) > hy) return null;
    out.push({ x: x / 10, y: y / 10, down: d === 1 });
  }
  return out;
}

/** The seed of the next swat in a run. 32-bit LCG; the product stays under 2^53, so it is exact in a double. */
export const nextSeed = (seed: number): number => (seed * 1664525 + 1013904223) >>> 0;

export const runSeeds = (base: number, n: number): number[] => {
  const s = [base >>> 0];
  while (s.length < n) s.push(nextSeed(s[s.length - 1]));
  return s;
};

/** The rules id for a brain: the game's CONFIG plus the exported brain's provenance (graph hash, commit). Browser and server both compute it. */
export const rulesFor = (m: { provenance?: Record<string, unknown> }): string => rulesId(CONFIG, JSON.stringify(m.provenance ?? null));

/** FNV-1a over the rules the round is played under (CONFIG), so a client on other rules is refused rather than mis-scored. */
export function rulesId(cfg: object = CONFIG, extra = ""): string {
  const text = JSON.stringify(cfg) + extra;
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, "0");
}
