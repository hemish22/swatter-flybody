// Run: cd web && npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { DUEL, DuelRun, runDuel } from "./duel.ts";
import { CONFIG, simulateRound } from "./world.ts";

const brainDir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, brainDir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const newBrain = () => FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), read("engine.wasm"));
const [A, B] = await Promise.all([newBrain(), newBrain()]);

test("a duel is a function of its seed and parameters", () => {
  for (const seed of [3, 7, 19]) {
    const a = runDuel(A, B, seed), b = runDuel(A, B, seed);
    assert.deepEqual(a.round.card, b.round.card);
    assert.equal(a.strikeAtMs, b.strikeAtMs);
    assert.deepEqual(a.trace, b.trace);
  }
});

test("the fly's side of a duel is the ordinary game: replaying B's cursor stream through Round gives the same round", () => {
  for (const seed of [2, 5, 11, 23]) {
    const d = runDuel(A, B, seed);
    const r = simulateRound(A, seed, d.trace);
    assert.deepEqual(r.card, d.round.card);
    assert.deepEqual(r.firstSpikes, d.round.firstSpikes);
  }
});

test("the swatter stays on the table and starts a sensible distance from the fly", () => {
  for (let seed = 1; seed <= 24; seed++) {
    const d = new DuelRun(A, B, seed);
    assert.ok(d.distanceMm > 40, `seed ${seed}: starts ${d.distanceMm} mm away`);
    for (let i = 0; i < 100000 && !d.done; i++) d.tick();
    for (const p of d.trace) assert.ok(Math.abs(p.x) <= CONFIG.arenaMm.w / 2 && Math.abs(p.y) <= CONFIG.arenaMm.h / 2);
  }
});

test("with B's looming detectors switched off it never strikes on its own spikes; it only runs out of road", () => {
  for (const seed of [1, 4, 9]) {
    const d = runDuel(A, B, seed, { ...DUEL, gainB: 0, creepMmPerMs: 0.02 });
    assert.equal(d.tGf, null);
    assert.equal(d.tPar, null);
    assert.notEqual(d.strikeReason, "spikes");
  }
});

test("a very sensitive B strikes on its own spikes, from farther away than a less sensitive one", () => {
  const far = (gainB: number) => {
    const ds: number[] = [];
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const d = runDuel(A, B, seed, { ...DUEL, gainB, creepMmPerMs: 0.03, startMm: 160 });
      if (d.strikeReason === "spikes") ds.push(d.strikeDistMm!);
    }
    return ds;
  };
  const lo = far(6), hi = far(12);
  assert.equal(hi.length, 6);
  assert.ok(Math.min(...hi) > Math.max(...lo), `${lo} vs ${hi}`);
});

test("B steers toward the fly: a swatter that starts 40 degrees off still arrives within the footprint", () => {
  for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
    const d = runDuel(A, B, seed, { ...DUEL, gainB: 1, creepMmPerMs: 0.02, startMm: 160, startBearingDeg: 40 });
    assert.ok(d.strikeDistMm !== null && d.strikeDistMm <= CONFIG.swatterHalfWidthMm, `seed ${seed}: ${d.strikeDistMm}`);
  }
});
