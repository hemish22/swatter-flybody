import { test } from "node:test";
import assert from "node:assert/strict";
import { diff, type SoundState } from "./sound.ts";

const s = (o: Partial<SoundState> = {}): SoundState => ({ roundId: 1, phase: "hover", spikeCount: 0, gfSpiked: false, left: false, outcome: null, ...o });

test("a new round, or the first look, is silent: it only sets the baseline", () => {
  assert.deepEqual(diff(null, s({ spikeCount: 400, gfSpiked: true, left: true })), []);
  assert.deepEqual(diff(s({ roundId: 1 }), s({ roundId: 2, spikeCount: 400, left: true })), []);
});

test("spikes, the giant fiber's first spike and the takeoff each sound once, when they happen", () => {
  assert.deepEqual(diff(s(), s({ spikeCount: 7 })), [{ kind: "tick", n: 7 }]);
  assert.deepEqual(diff(s({ spikeCount: 7 }), s({ spikeCount: 7 })), []);
  assert.deepEqual(diff(s({ spikeCount: 7 }), s({ spikeCount: 9, gfSpiked: true })), [{ kind: "tick", n: 2 }, { kind: "zap" }]);
  assert.deepEqual(diff(s({ gfSpiked: true }), s({ gfSpiked: true, left: true })), [{ kind: "takeoff" }]);
});

test("a strike and the result are separate events, and the result is named", () => {
  assert.deepEqual(diff(s(), s({ phase: "strike" })), [{ kind: "strike" }]);
  assert.deepEqual(diff(s({ phase: "strike" }), s({ phase: "done", outcome: "hit" })), [{ kind: "result", outcome: "hit" }]);
  // spooked: the round goes straight from hover to done, with no strike sound
  assert.deepEqual(diff(s(), s({ phase: "done", outcome: "spooked" })), [{ kind: "result", outcome: "spooked" }]);
  assert.deepEqual(diff(s({ phase: "done", outcome: "hit" }), s({ phase: "done", outcome: "hit" })), []);
});
