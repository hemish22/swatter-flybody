// Fit Duel's two free parameters (B's sensitivity and the swatter's creep speed) on a grid, then check the winner on
// fresh seeds. Target: a duel that is neither a foregone hit nor a foregone escape, and where the swatter's brain, not
// its running out of road, is what usually triggers the strike.
// Run: cd web && node game/duel_tune.ts [--seeds 32] [--check 64] [--out ../docs/duel_tune.json]
import { readFileSync, writeFileSync } from "node:fs";
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { DUEL, runDuel, type DuelParams } from "./duel.ts";

const arg = (name: string, d: string) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : d; };
const SEEDS = +arg("seeds", "32"), CHECK = +arg("check", "64"), OUT = arg("out", new URL("../../docs/duel_tune.json", import.meta.url).pathname);

const dir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, dir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const ab = () => bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer;
const A = await FlyBrain.create(manifest, ab(), read("engine.wasm"));
const B = await FlyBrain.create(manifest, ab(), read("engine.wasm"));

interface Tally { n: number; hit: number; hitBySpikes: number; flyWins: number; miss: number; gameSeconds: number }
function run(params: DuelParams, first: number, n: number): Tally {
  const t: Tally = { n, hit: 0, hitBySpikes: 0, flyWins: 0, miss: 0, gameSeconds: 0 };
  for (let s = first; s < first + n; s++) {
    const d = runDuel(A, B, s, params), o = d.round.card!.outcome;
    if (o === "hit") { t.hit++; if (d.strikeReason === "spikes") t.hitBySpikes++; }
    else if (o === "miss") t.miss++;
    else t.flyWins++; // escaped, spooked or timeout: the fly got away
    t.gameSeconds += d.timeMs / 1000 / n;
  }
  return t;
}
const frac = (t: Tally) => ({ hit: t.hit / t.n, bySpikes: t.hitBySpikes / t.n, fly: t.flyWins / t.n, miss: t.miss / t.n });
// target mix: the swatter wins about 4 in 10 on its own spikes, the fly 4 in 10, the rest are early strikes that miss
const score = (t: Tally) => { const f = frac(t); return (f.bySpikes - 0.4) ** 2 + (f.fly - 0.4) ** 2 + (f.miss - 0.2) ** 2 + (f.hit - f.bySpikes) ** 2; };

const gains: number[] = [], creeps: number[] = [];
for (let g = 2; g <= 4.51; g += 0.25) gains.push(+g.toFixed(2));
for (let v = 0.02; v <= 0.0501; v += 0.005) creeps.push(+v.toFixed(3));
const rows: Array<{ gainB: number; creepMmPerMs: number; tally: Tally; score: number }> = [];
console.log(`grid ${gains.length} x ${creeps.length}, ${SEEDS} seeds each (seeds 1..${SEEDS})\ngainB creep | hit(by spikes) fly miss | game s`);
for (const v of creeps) for (const g of gains) {
  const params = { ...DUEL, gainB: g, creepMmPerMs: v, startMm: 160 };
  const t = run(params, 1, SEEDS), f = frac(t);
  rows.push({ gainB: g, creepMmPerMs: v, tally: t, score: score(t) });
  console.log(`${String(g).padEnd(5)} ${String(v).padEnd(5)} | ${f.hit.toFixed(2)} (${f.bySpikes.toFixed(2)}) ${f.fly.toFixed(2)} ${f.miss.toFixed(2)} | ${t.gameSeconds.toFixed(1)}`);
}
rows.sort((a, b) => a.score - b.score);
const best = rows[0];
const check = run({ ...DUEL, gainB: best.gainB, creepMmPerMs: best.creepMmPerMs, startMm: 160 }, 1001, CHECK);
console.log("\nbest on the fit seeds:", best.gainB, best.creepMmPerMs, JSON.stringify(frac(best.tally)));
console.log(`fresh seeds 1001..${1000 + CHECK}:`, JSON.stringify(frac(check)), `game s ${check.gameSeconds.toFixed(1)}`);
writeFileSync(OUT, JSON.stringify({ fitSeeds: SEEDS, checkSeeds: CHECK, best: { gainB: best.gainB, creepMmPerMs: best.creepMmPerMs, startMm: 160 }, fit: frac(best.tally), check: frac(check), top: rows.slice(0, 8).map((r) => ({ gainB: r.gainB, creepMmPerMs: r.creepMmPerMs, ...frac(r.tally), score: +r.score.toFixed(3) })) }, null, 1));
