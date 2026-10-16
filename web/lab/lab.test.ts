// Lab stimuli are the offline sweep's stimuli: frames equal the Python parity frames, outcomes equal the expected ones.
// Run: node --test "lab/*.test.ts"   (npm test)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FlyBrain, type Frame, type Manifest } from "../engine/brain.ts";
import { labFrames, runLab, type LabKind } from "./loom.ts";

const dir = new URL("../brain/", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, dir));
const manifest: Manifest = JSON.parse(read("manifest.json").toString());
const bin = read(manifest.binary);
const parity = JSON.parse(read("parity.json").toString()) as {
  trials: Array<{ id: number; kind: LabKind; rv_ms: number; azimuth_deg: number; frames: Frame[];
    expected: { gf_first_ms: number | null; mode: string; heading_side: string | null } }>;
};
const brain = await FlyBrain.create(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), read("engine.wasm"));

test("Lab frames equal the Python parity frames for all 50 stimuli", () => {
  for (const t of parity.trials) {
    const { frames } = labFrames({ kind: t.kind, rvMs: t.rv_ms, azimuthDeg: t.azimuth_deg });
    assert.equal(frames.length, t.frames.length, `stimulus ${t.id}`);
    frames.forEach((f, i) => f.forEach((v, k) => assert.ok(Math.abs(v - t.frames[i][k]) < 1e-6, `stimulus ${t.id} frame ${i}[${k}]: ${v} vs ${t.frames[i][k]}`)));
  }
});

test("Lab outcomes equal the Python reference's for all 50 stimuli", () => {
  for (const t of parity.trials) {
    const r = runLab(brain, { kind: t.kind, rvMs: t.rv_ms, azimuthDeg: t.azimuth_deg });
    assert.equal(r.mode, t.expected.mode, `stimulus ${t.id}`);
    assert.equal(r.headingSide, t.expected.heading_side, `stimulus ${t.id}`);
    assert.ok((r.gfFirstMs === null) === (t.expected.gf_first_ms === null), `stimulus ${t.id}`);
    if (r.gfFirstMs !== null) assert.ok(Math.abs(r.gfFirstMs - t.expected.gf_first_ms!) <= 0.2 + 1e-9, `stimulus ${t.id}`);
  }
});

test("criterion 1 in the browser's own run: GF spikes for every expanding Lab loom, never for the controls", () => {
  for (const rv of [10, 20, 40, 80]) for (const az of [45, 90, 135, 225, 270, 315]) {
    assert.notEqual(runLab(brain, { kind: "expanding", rvMs: rv, azimuthDeg: az }).gfFirstMs, null, `expanding r/v ${rv} az ${az}`);
    for (const kind of ["receding", "translating", "dimming"] as const)
      assert.equal(runLab(brain, { kind, rvMs: rv, azimuthDeg: az }).gfFirstMs, null, `${kind} r/v ${rv} az ${az}`);
  }
});

test("the angle at the GF spike rises with r/v (criterion 2's figure, from the engine)", () => {
  const med = (rv: number) => {
    const a = [45, 90, 135, 225, 270, 315].map((az) => runLab(brain, { kind: "expanding", rvMs: rv, azimuthDeg: az }).thetaAtGfDeg!).sort((x, y) => x - y);
    return a[a.length >> 1];
  };
  const m = [10, 20, 40, 80].map(med);
  console.log("median angle at GF spike for r/v 10/20/40/80:", m.map((v) => v.toFixed(1)).join(" / "));
  for (let i = 1; i < m.length; i++) assert.ok(m[i] > m[i - 1]);
});
