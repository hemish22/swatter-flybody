// Ablation: silence the giant fibers and see what the swatter bot hits.
// Chai et al. 2025 report that flies with a silenced giant fiber are caught more by damselflies. The circuit-level
// prediction here is the same hover-and-strike bot as game/difficulty.ts, run once with the circuit as built and once
// with the giant fibers' spikes ignored (the parallel DNs alone decide takeoff, always long-mode).
// This is a bot on the model, not a human and not a fly. Run: cd web && node game/ablation_gf.ts
import { readFileSync, writeFileSync } from "node:fs";
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { CONFIG, Round, simulateRound, type Config, type Input } from "./world.ts";

const dir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, dir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const brain = await FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), read("engine.wasm"));
const SEEDS = Array.from({ length: 24 }, (_, i) => i + 1);

function hoverStrike(fly: { x: number; y: number }, cfg: Config): Input[] {
  const at = { x: fly.x + 20, y: fly.y }, tr: Input[] = [];
  for (let i = 0; i < 40; i++) tr.push({ ...at, down: false });
  for (let i = 0; i < Math.round(cfg.chargeMs / cfg.tickMs) + 2; i++) tr.push({ ...at, down: true });
  for (let i = 0; i < 400; i++) tr.push({ ...at, down: false });
  return tr;
}

const rows: Array<Record<string, unknown>> = [];
console.log("strike m/s  r/v ms | intact: hit / escaped / spooked | GF silenced: hit / escaped / spooked | median early-by ms (intact -> silenced)");
for (const v of [0.6, 1, 1.5, 2, 3, 4, 5]) {
  const out: Record<string, { hit: number; escaped: number; spooked: number; other: number; margin: number | null }> = {};
  for (const [label, silence] of [["intact", false], ["silenced", true]] as const) {
    const cfg: Config = { ...CONFIG, strikeSpeedMin: v, strikeSpeedMax: v, silenceGf: silence };
    const c = { hit: 0, escaped: 0, spooked: 0, other: 0 }, margins: number[] = [];
    for (const s of SEEDS) {
      const r = simulateRound(brain, s, hoverStrike(new Round(brain, s, cfg).fly, cfg), cfg).card!;
      if (r.outcome === "hit" || r.outcome === "escaped" || r.outcome === "spooked") c[r.outcome]++; else c.other++;
      if (r.marginMs !== null) margins.push(r.marginMs);
    }
    margins.sort((a, b) => a - b);
    out[label] = { ...c, margin: margins.length ? margins[margins.length >> 1] : null };
  }
  const a = out.intact, b = out.silenced, n = SEEDS.length;
  console.log(`${String(v).padEnd(10)}  ${(50 / v).toFixed(0).padStart(4)}   | ${a.hit}/${n} / ${a.escaped} / ${a.spooked}`.padEnd(48) + `| ${b.hit}/${n} / ${b.escaped} / ${b.spooked}`.padEnd(30) + `| ${a.margin?.toFixed(1) ?? "-"} -> ${b.margin?.toFixed(1) ?? "-"}`);
  rows.push({ strikeMs: v, rvMs: 50 / v, intact: a, silenced: b });
}
writeFileSync(new URL("../../docs/ablation_gf.json", import.meta.url), JSON.stringify({ seeds: SEEDS.length, hoverMm: CONFIG.hoverHeightMm, rows }, null, 1));
