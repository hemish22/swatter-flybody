// Feasibility probe for Duel mode: does the second brain fire usefully, and what do the duels look like?
// Sweeps B's sensitivity and creep speed and prints outcomes plus where B decided to strike.
// Run: cd web && node game/duel_probe.ts
import { readFileSync } from "node:fs";
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { DUEL, runDuel, type DuelParams } from "./duel.ts";

const dir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, dir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const ab = () => bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer;
const A = await FlyBrain.create(manifest, ab(), read("engine.wasm"));
const B = await FlyBrain.create(manifest, ab(), read("engine.wasm"));
const SEEDS = Array.from({ length: 16 }, (_, i) => i + 1);
const q = (a: number[], p: number) => (a.length ? [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(p * a.length))] : NaN);

function sweep(label: string, over: Partial<DuelParams>): void {
  const params = { ...DUEL, ...over };
  const out: Record<string, number> = { hit: 0, escaped: 0, spooked: 0, miss: 0, timeout: 0 };
  const why: Record<string, number> = { spikes: 0, arrived: 0, patience: 0, none: 0 };
  const dist: number[] = [], dur: number[] = [];
  const t0 = performance.now();
  for (const s of SEEDS) {
    const d = runDuel(A, B, s, params);
    out[d.round.card!.outcome]++;
    why[d.strikeReason ?? "none"]++;
    if (d.strikeDistMm !== null) dist.push(d.strikeDistMm);
    dur.push(d.timeMs / 1000);
  }
  const ms = performance.now() - t0;
  console.log(`${label.padEnd(34)} hit ${out.hit} esc ${out.escaped} spook ${out.spooked} miss ${out.miss} tmo ${out.timeout} | B strikes by spikes ${why.spikes} arrived ${why.arrived} patience ${why.patience} fly-first ${why.none} | strike dist mm med ${q(dist, .5).toFixed(0)} [${q(dist, .1).toFixed(0)}-${q(dist, .9).toFixed(0)}] | game s med ${q(dur, .5).toFixed(1)} | ${(ms / SEEDS.length).toFixed(0)} ms/duel`);
}

console.log("defaults:", JSON.stringify(DUEL));
for (const v of [0.02, 0.03, 0.04]) {
  for (const g of [1, 2, 3, 5, 8, 12]) sweep(`creep ${v} gainB ${g}`, { creepMmPerMs: v, gainB: g, startMm: 160 });
  console.log();
}
