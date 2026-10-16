import { test } from "node:test";
import assert from "node:assert/strict";
import { deriveStory, growthStartBefore, resultLine, type Snapshot } from "./story.ts";

const base: Snapshot = {
  timeMs: 0, thetaDeg: 20, rateDegPerMs: 0, growthStartsMs: [], lplc2Active: 0, lplc2Driven: 91,
  firstLplc2Ms: null, firstGfMs: null, thetaAtGfDeg: null, thresholdMv: -45, takeoff: null, headingSide: null,
};

test("before anything happens every step is idle", () => {
  const s = deriveStory(base);
  assert.deepEqual(s.map((x) => x.state), ["idle", "idle", "idle", "idle"]);
});

test("steps light up in order as the brain reacts", () => {
  const growing = deriveStory({ ...base, rateDegPerMs: 0.4, growthStartsMs: [100] });
  assert.deepEqual(growing.map((x) => x.state), ["live", "live", "idle", "idle"]);
  const detectors = deriveStory({ ...base, rateDegPerMs: 0.4, growthStartsMs: [100], firstLplc2Ms: 108, lplc2Active: 12 });
  assert.deepEqual(detectors.map((x) => x.state), ["live", "done", "live", "idle"]);
  const done = deriveStory({
    ...base, growthStartsMs: [100], firstLplc2Ms: 108, firstGfMs: 114, thetaAtGfDeg: 31,
    takeoff: { mode: "short", leaveMs: 121 }, headingSide: "left",
  });
  assert.deepEqual(done.map((x) => x.state), ["done", "done", "done", "done"]);
  assert.match(done[2].detail, /14 ms in/);
  assert.match(done[3].detail, /short-mode takeoff, 21 ms after/);
  assert.match(done[3].detail, /turns left/);
});

test("a missing heading side is stated, not invented", () => {
  const s = deriveStory({ ...base, growthStartsMs: [0], firstGfMs: 5, takeoff: { mode: "short", leaveMs: 12 }, headingSide: null });
  assert.match(s[3].detail, /no side/);
});

test("an event is measured from the growth spell that preceded it", () => {
  assert.equal(growthStartBefore([100, 400, 900], 450), 400);
  assert.equal(growthStartBefore([100, 400], 50), 100);
  assert.equal(growthStartBefore([], 50), null);
});

test("the second step always carries the supplied-detector caveat", () => {
  assert.match(deriveStory(base)[1].caveat ?? "", /We supply/);
});

test("result lines name the outcome in plain words", () => {
  assert.match(resultLine("escaped", 14.2), /14 ms before the swat landed/);
  assert.match(resultLine("hit", 3), /^Hit/);
});
