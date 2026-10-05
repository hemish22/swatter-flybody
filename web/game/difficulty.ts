// The skill curves behind docs/game.md, from the real engine: hit rate vs strike speed, and how slowly the
// swatter must creep (and from which side) to reach the fly unnoticed.
// Run: cd web && PATH=$HOME/.local/node/bin:$PATH node game/difficulty.ts   (writes ../docs/game_difficulty.json)
import { readFileSync, writeFileSync } from "node:fs";
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { CONFIG, Round, simulateRound, type Config, type Input } from "./world.ts";

const dir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, dir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const brain = await FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), read("engine.wasm"));
const SEEDS = Array.from({ length: 12 }, (_, i) => i + 1);

function hoverStrike(fly: { x: number; y: number }, cfg: Config): Input[] {
  const at = { x: fly.x + 20, y: fly.y }, tr: Input[] = [];
  for (let i = 0; i < 40; i++) tr.push({ ...at, down: false });
  for (let i = 0; i < Math.round(cfg.chargeMs / cfg.tickMs) + 2; i++) tr.push({ ...at, down: true });
  for (let i = 0; i < 400; i++) tr.push({ ...at, down: false });
  return tr;
}

// 1. hit rate against strike speed, cursor still: the swatter hovers over the fly and strikes at a fixed speed
const byStrikeSpeed: Record<string, { rvMs: number; hit: number; escaped: number; spooked: number; marginMedianMs: number | null }> = {};
for (const v of [0.6, 1, 1.5, 2, 3, 4, 5]) {
  const cfg = { ...CONFIG, strikeSpeedMin: v, strikeSpeedMax: v };
  const c = { hit: 0, escaped: 0, spooked: 0 }, margins: number[] = [];
  for (const s of SEEDS) {
    const r = simulateRound(brain, s, hoverStrike(new Round(brain, s, cfg).fly, cfg), cfg).card!;
    if (r.outcome === "hit" || r.outcome === "escaped" || r.outcome === "spooked") c[r.outcome]++;
    if (r.marginMs !== null) margins.push(r.marginMs);
  }
  margins.sort((a, b) => a - b);
  byStrikeSpeed[v] = { rvMs: 50 / v, ...c, marginMedianMs: margins.length ? margins[margins.length >> 1] : null };
  console.log(`strike ${v} m/s (r/v ${(50 / v).toFixed(0)} ms): hit ${c.hit}/${SEEDS.length}, escaped ${c.escaped}, spooked ${c.spooked}; median margin ${byStrikeSpeed[v].marginMedianMs?.toFixed(1) ?? "-"} ms`);
}

// 2. creeping: from 300 mm away along each bearing (0 = dead ahead of the fly, 90 = its right), then a full charge
const bearings = [0, 45, 90, 135, 180, 225, 270, 315];
const speeds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3];
const creep: Record<string, Record<string, number>> = {};
for (const b of bearings) {
  creep[b] = {};
  for (const u of speeds) {
    let hits = 0;
    for (const s of SEEDS) {
      const fly = new Round(brain, s).fly, a = fly.heading - (b * Math.PI) / 180, L = 300, n = Math.round(L / (u * CONFIG.tickMs));
      const tr: Input[] = [];
      for (let i = 0; i <= n; i++) { const f = 1 - i / n; tr.push({ x: fly.x + Math.cos(a) * L * f, y: fly.y + Math.sin(a) * L * f, down: false }); }
      const at = tr[tr.length - 1];
      for (let i = 0; i < CONFIG.chargeMs / CONFIG.tickMs + 2; i++) tr.push({ ...at, down: true });
      for (let i = 0; i < 400; i++) tr.push({ ...at, down: false });
      if (simulateRound(brain, s, tr).card!.outcome === "hit") hits++;
    }
    creep[b][u] = hits / SEEDS.length;
  }
  const ok = speeds.filter((u) => creep[b][u] === 1);
  console.log(`bearing ${String(b).padStart(3)}: ` + speeds.map((u) => `${u}:${creep[b][u].toFixed(2)}`).join(" ") + `   fastest safe creep ${ok.length ? Math.max(...ok) : "<0.01"} m/s`);
}
writeFileSync(new URL("../../docs/game_difficulty.json", import.meta.url), JSON.stringify({ config: CONFIG, seeds: SEEDS.length, byStrikeSpeed, creepHitFraction: creep }, null, 1));
