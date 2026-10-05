// Node parity test: the WASM engine through the TypeScript glue against web/brain/parity.json.
// Run: PATH=$HOME/.local/node/bin:$PATH node --test web/engine/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FlyBrain, type Frame, type Manifest, type Outcome } from "./brain.ts";

const dir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, dir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const parity = JSON.parse(read("parity.json").toString()) as {
  n: number; dt_ms: number;
  trials: Array<{ id: number; kind: string; rv_ms: number; azimuth_deg: number; frames: Frame[];
    expected: { gf_first_ms: number | null; parallel_first_ms: number | null; escape_first_ms: number | null;
      A: number; heading_side: string | null; mode: string } }>;
};
const brain = await FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), read("engine.wasm"));

const near = (a: number | null, b: number | null, tol: number) =>
  (a === null) === (b === null) && (a === null || Math.abs(a - (b as number)) <= tol + 1e-9);

test("50 parity stimuli: mode, heading side and first-spike times match the Python reference", () => {
  assert.equal(parity.n, 50);
  const bad: unknown[] = [];
  for (const t of parity.trials) {
    brain.run(t.frames);
    const g: Outcome = brain.outcome(), e = t.expected;
    const ok = g.mode === e.mode && g.headingSide === e.heading_side &&
      near(g.gfFirstMs, e.gf_first_ms, parity.dt_ms) && near(g.parallelFirstMs, e.parallel_first_ms, parity.dt_ms) &&
      near(g.escapeFirstMs, e.escape_first_ms, parity.dt_ms);
    if (!ok) bad.push([t.id, t.kind, t.rv_ms, t.azimuth_deg, g, e]);
  }
  assert.deepEqual(bad, []);
});

test("a second run on the same instance reproduces the first", () => {
  const f = parity.trials[3].frames;
  assert.deepEqual(brain.run(f), brain.run(f));
});

test("streaming frame by frame gives the same spikes as run()", () => {
  const f = parity.trials[8].frames;
  const whole = brain.run(f);
  brain.reset();
  for (const fr of f) brain.frame(fr);
  brain.finish();
  assert.deepEqual(brain.spikes(), whole);
});

test("speed: a 400 ms trial is much faster than real time", () => {
  const f = parity.trials[8].frames;
  brain.run(f);
  const t0 = performance.now();
  for (let i = 0; i < 20; i++) brain.run(f);
  const ms = (performance.now() - t0) / 20;
  console.log(`400 ms trial: ${ms.toFixed(1)} ms in Node (${(400 / ms).toFixed(0)}x real time)`);
  assert.ok(ms < 200, `${ms} ms per 400 ms trial`);
});
