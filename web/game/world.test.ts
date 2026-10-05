// Run: cd web && PATH=$HOME/.local/node/bin:$PATH npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { CONFIG, Round, decideTakeoff, eyeView, growthRate, rng, simulateRound, type Input } from "./world.ts";

const brainDir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, brainDir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const newBrain = () => FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), read("engine.wasm"));
const fixtures = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url)).toString());
const R = CONFIG.swatterHalfWidthMm;

// --- geometry, pinned to heading.plane_position

test("horizontal-plane directions land where the Python reference puts them, on both eyes", () => {
  for (const row of fixtures.horizontal_plane_positions as Array<{ azimuth_deg: number; R: number[]; L: number[] }>) {
    const a = (row.azimuth_deg * Math.PI) / 180;
    // fly at the origin with heading 0 (+x is forward): body azimuth a, +90 is the fly's right = -y world
    const v = eyeView({ x: 0, y: 0, heading: 0 }, { x: 200 * Math.cos(a), y: -200 * Math.sin(a), z: CONFIG.eyeHeightMm });
    assert.ok(Math.abs(v.right[0] - row.R[0]) < 1e-6 && Math.abs(v.right[1] - row.R[1]) < 1e-6, `R ${row.azimuth_deg}: ${v.right} vs ${row.R}`);
    assert.ok(Math.abs(v.left[0] - row.L[0]) < 1e-6 && Math.abs(v.left[1] - row.L[1]) < 1e-6, `L ${row.azimuth_deg}: ${v.left} vs ${row.L}`);
  }
});

test("heading rotates the body, not the world: turning the fly by 90 deg shifts the azimuth by 90", () => {
  const a = eyeView({ x: 0, y: 0, heading: 0 }, { x: 0, y: -100, z: 1 }); // to the right of a fly facing +x
  const b = eyeView({ x: 0, y: 0, heading: Math.PI / 2 }, { x: 100, y: 0, z: 1 }); // same, fly facing +y
  assert.deepEqual(a.right.map((x) => +x.toFixed(9)), b.right.map((x) => +x.toFixed(9)));
});

test("angular size matches the lab formula 2 atan(r/D), and the analytic rate matches a finite difference", () => {
  const fly = { x: 0, y: 0 };
  for (const rv of [10, 20, 40, 80]) {
    const v = R / rv; // mm/ms
    for (const D of [400, 200, 80]) {
      const at = (d: number) => ({ x: 0, y: 0, z: CONFIG.eyeHeightMm + d });
      const theta = (d: number) => eyeView({ ...fly, heading: 0 }, at(d)).thetaDeg;
      assert.ok(Math.abs(theta(D) - (2 * Math.atan(R / D) * 180) / Math.PI) < 1e-9);
      const h = 0.01; // ms
      const fd = (theta(D - v * h) - theta(D + v * h)) / (2 * h) * -1 * -1; // approaching: D shrinks
      const analytic = growthRate(fly, at(D), { x: 0, y: 0, z: -v });
      assert.ok(Math.abs(analytic - (theta(D - v * h) - theta(D + v * h)) / (2 * h)) < 1e-4 * Math.abs(analytic), `rv ${rv} D ${D}: ${analytic} vs ${fd}`);
    }
  }
});

test("a receding swatter has zero growth to the engine (relu), and a sideways one adds none at constant distance", () => {
  const fly = { x: 0, y: 0 };
  assert.ok(growthRate(fly, { x: 0, y: 0, z: 100 }, { x: 0, y: 0, z: 1 }) < 0);
  assert.ok(growthRate(fly, { x: 100, y: 0, z: CONFIG.eyeHeightMm }, { x: 0, y: 1, z: 0 }) === 0); // -0 counts
});

// --- the takeoff rule

test("takeoff rule: giant fiber first, within the raise window, and parallel-only", () => {
  assert.deepEqual(decideTakeoff(10, null, 10), { mode: "short", leaveMs: 10 + CONFIG.jumpDelayMs });
  assert.deepEqual(decideTakeoff(12, 10, 12), { mode: "short", leaveMs: 12 + CONFIG.jumpDelayMs });
  assert.equal(decideTakeoff(null, 10, 12), null, "parallel DNs fired but the raise has not completed: undecided");
  assert.deepEqual(decideTakeoff(null, 10, 10 + CONFIG.raiseWindowMs), { mode: "long", leaveMs: 10 + CONFIG.raiseWindowMs });
  assert.deepEqual(decideTakeoff(30, 10, 40), { mode: "long", leaveMs: 10 + CONFIG.raiseWindowMs });
  assert.equal(decideTakeoff(null, null, 100), null);
});

// --- rounds

const CURSOR_MODE = { ...CONFIG, strikeSpeedSource: "cursor" as const, hoverHeightMm: 150 }; // the plan's rules

/** Cursor still, 0.2 s above the fly (offset 20 mm), then hold the button for `holdMs` and release. */
function chargeTrace(fly: { x: number; y: number }, holdMs: number): Input[] {
  const at = { x: fly.x + 20, y: fly.y };
  const tr: Input[] = [];
  for (let i = 0; i < 40; i++) tr.push({ ...at, down: false });
  for (let i = 0; i < Math.max(1, Math.round(holdMs / CONFIG.tickMs)); i++) tr.push({ ...at, down: true }); // a tap is one tick
  for (let i = 0; i < 400; i++) tr.push({ ...at, down: false });
  return tr;
}

/** Cursor creeping toward the fly from 300 mm at `u` mm/ms along `bearing` from its heading, then a full charge. */
function creepTrace(fly: { x: number; y: number; heading: number }, bearing: number, u: number): Input[] {
  const a = fly.heading + bearing, L = 300, n = Math.round(L / (u * CONFIG.tickMs));
  const tr: Input[] = [];
  for (let i = 0; i <= n; i++) {
    const f = 1 - i / n;
    tr.push({ x: fly.x + Math.cos(a) * L * f, y: fly.y + Math.sin(a) * L * f, down: false });
  }
  const at = tr[tr.length - 1];
  for (let i = 0; i < CONFIG.chargeMs / CONFIG.tickMs + 2; i++) tr.push({ ...at, down: true });
  for (let i = 0; i < 400; i++) tr.push({ ...at, down: false });
  return tr;
}

test("the same seed and trace give an identical round", async () => {
  const brain = await newBrain();
  const fly = new Round(brain, 7).fly;
  const a = simulateRound(brain, 7, chargeTrace(fly, 1000));
  const b = simulateRound(brain, 7, chargeTrace(fly, 1000));
  assert.deepEqual(a.card, b.card);
  assert.deepEqual(a.thetaDeg, b.thetaDeg);
});

test("different seeds put the fly in different places", () => {
  const a = rng(1)(), b = rng(2)();
  assert.notEqual(a, b);
});

test("a strike that lands away from the fly is a miss", async () => {
  const brain = await newBrain();
  const fly = new Round(brain, 3).fly;
  const tr = chargeTrace({ x: fly.x + 200, y: fly.y }, 1000);
  assert.equal(simulateRound(brain, 3, tr).card?.outcome, "miss");
});

test("hold time sets r/v, from the plan's slowest (83 ms) to its fastest (10 ms)", async () => {
  const brain = await newBrain();
  const fly = new Round(brain, 11).fly;
  const min = simulateRound(brain, 11, chargeTrace(fly, 0)).card!; // a tap
  const max = simulateRound(brain, 11, chargeTrace(fly, 2000)).card!;
  const mid = simulateRound(brain, 11, chargeTrace(fly, 500)).card!;
  assert.ok(Math.abs(min.strikeRvMs! - 50 / CONFIG.strikeSpeedMin) < 0.05 * (50 / CONFIG.strikeSpeedMin), `min ${min.strikeRvMs}`);
  assert.ok(Math.abs(max.strikeRvMs! - 50 / CONFIG.strikeSpeedMax) < 0.1, `max ${max.strikeRvMs}`);
  assert.ok(mid.strikeRvMs! < min.strikeRvMs! && mid.strikeRvMs! > max.strikeRvMs!);
});

test("the slowest strike is escaped and the fastest lands, across seeds", async () => {
  const brain = await newBrain();
  const seeds = [1, 2, 3, 4, 5, 6, 7, 8];
  let slowEscapes = 0, fastHits = 0;
  for (const s of seeds) {
    const fly = new Round(brain, s).fly;
    if (simulateRound(brain, s, chargeTrace(fly, 0)).card!.outcome === "escaped") slowEscapes++;
    if (simulateRound(brain, s, chargeTrace(fly, 2000)).card!.outcome === "hit") fastHits++;
  }
  console.log(`slowest strikes escaped ${slowEscapes}/8, fastest strikes hit ${fastHits}/8`);
  assert.equal(slowEscapes, 8);
  assert.equal(fastHits, 8);
});

test("the plan's own rules (150 mm hover, strike speed from a flick) leave no strike able to land", async () => {
  // Pinned finding, docs/game.md: a flick is a loom the fly sees before the strike begins.
  const brain = await newBrain();
  let hits = 0;
  for (const s of [1, 2, 3, 4, 5, 6]) {
    const fly = new Round(brain, s, CURSOR_MODE).fly;
    const lead = 3 * CONFIG.strikeSpeedWindowMs, tr: Input[] = [];
    for (let i = 0; i < 40; i++) tr.push({ x: fly.x - lead, y: fly.y, down: false });
    for (let i = 1; i <= 20; i++) tr.push({ x: fly.x - lead + (lead * i) / 20, y: fly.y, down: i === 20 });
    for (let i = 0; i < 400; i++) tr.push({ x: fly.x, y: fly.y, down: true });
    if (simulateRound(brain, s, tr, CURSOR_MODE).card!.outcome === "hit") hits++;
  }
  assert.equal(hits, 0);
});

test("creeping works, and how slowly depends on the side you come from (model- and eye-axis-dependent)", async () => {
  const brain = await newBrain();
  const outcomes = (bearing: number, u: number) => [1, 2, 3, 4].map((s) => {
    const fly = new Round(brain, s).fly;
    return simulateRound(brain, s, creepTrace(fly, bearing, u)).card!.outcome;
  });
  const AHEAD = 0, RIGHT = -Math.PI / 2;
  assert.deepEqual(outcomes(AHEAD, 0.1), ["hit", "hit", "hit", "hit"], "from ahead at 0.1 m/s");
  assert.deepEqual(outcomes(RIGHT, 0.1), ["spooked", "spooked", "spooked", "spooked"], "from the side at 0.1 m/s");
  assert.deepEqual(outcomes(RIGHT, 0.02), ["hit", "hit", "hit", "hit"], "from the side at 0.02 m/s");
});

test("a round that never strikes ends in a timeout, not a hang", async () => {
  const brain = await newBrain();
  const tr: Input[] = Array.from({ length: 13000 }, () => ({ x: 1000, y: 1000, down: false }));
  assert.equal(simulateRound(brain, 5, tr).card?.outcome, "timeout");
});
