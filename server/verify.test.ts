// Run: node --test "server/*.test.ts"
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadBrain } from "./server.ts";
import { createApp } from "./server.ts";
import { runSeeds, rulesFor, rulesId } from "../web/game/trace.ts";
import { CLASSIC_SWATS, verifyRun, type SwatSubmission } from "./verify.ts";
import { scriptedSwat } from "./testutil.ts";
import type { AddressInfo } from "node:net";

const { brain, manifest } = await loadBrain();
const BASE = 12345;
// alternate full-charge strikes (hits) and one-tick flicks (escaped or missed): a realistic mix
const run = (): SwatSubmission[] => runSeeds(BASE, CLASSIC_SWATS).map((s, k) => scriptedSwat(brain, s, k % 2 ? 1 : 200));

test("an honest run verifies and its hits are the replay's", () => {
  const swats = run();
  const v = verifyRun(brain, BASE, swats);
  assert.ok(v.ok, v.ok ? "" : v.reason);
  if (v.ok) {
    assert.equal(v.hits, swats.filter((s) => s.claimed.outcome === "hit").length);
    assert.ok(v.hits >= 5 && v.hits < CLASSIC_SWATS, `hits ${v.hits}`);
  }
});

test("a wrong claimed outcome is refused and names the swat", () => {
  const swats = run();
  swats[3] = { ...swats[3], claimed: { ...swats[3].claimed, outcome: swats[3].claimed.outcome === "hit" ? "escaped" : "hit" } };
  const v = verifyRun(brain, BASE, swats);
  assert.ok(!v.ok && v.swat === 3, JSON.stringify(v));
});

test("the wrong seed gives different rounds, so a replayed trace does not carry over", () => {
  assert.ok(!verifyRun(brain, BASE + 1, run()).ok);
});

test("a tampered trace is refused (cursor moved off the fly)", () => {
  const swats = run();
  const t = (swats[0].trace as number[]).slice();
  for (let k = 0; k < t.length; k += 3) t[k] += 3000; // 300 mm to the side
  assert.ok(!verifyRun(brain, BASE, [{ ...swats[0], trace: t }, ...swats.slice(1)]).ok);
});

test("structural abuse is refused: wrong count, padding, truncation, out-of-arena, junk", () => {
  const swats = run();
  assert.ok(!verifyRun(brain, BASE, swats.slice(0, 19)).ok);
  const padded = [...swats]; padded[2] = { ...swats[2], trace: [...(swats[2].trace as number[]), 0, 0, 0] };
  assert.match((verifyRun(brain, BASE, padded) as { reason: string }).reason, /after the round ended/);
  const cut = [...swats]; cut[2] = { ...swats[2], trace: (swats[2].trace as number[]).slice(0, 30) };
  assert.match((verifyRun(brain, BASE, cut) as { reason: string }).reason, /does not finish/);
  const out = [...swats]; out[2] = { ...swats[2], trace: [9999, 0, 0] };
  assert.match((verifyRun(brain, BASE, out) as { reason: string }).reason, /malformed/);
  const junk = [...swats]; junk[2] = { ...swats[2], trace: "nope" };
  assert.match((verifyRun(brain, BASE, junk) as { reason: string }).reason, /malformed/);
});

test("verification is deterministic and takes well under a second for an ordinary run", () => {
  const swats = run();
  const t0 = performance.now();
  const a = verifyRun(brain, BASE, swats), b = verifyRun(brain, BASE, swats);
  const ms = (performance.now() - t0) / 2;
  console.log(`verify 20 swats: ${ms.toFixed(0)} ms`);
  assert.deepEqual(a, b);
  assert.ok(ms < 2000);
});

test("HTTP: run, submit, leaderboard; a run cannot be submitted twice; other rules are refused", async () => {
  const server = await createApp(":memory:");
  await new Promise<void>((ok) => server.listen(0, ok));
  const base = `http://localhost:${(server.address() as AddressInfo).port}`;
  try {
    const r = await (await fetch(`${base}/api/run`)).json() as { runId: string; seed: number; rules: string };
    const swats = runSeeds(r.seed, CLASSIC_SWATS).map((s, k) => scriptedSwat(brain, s, k % 2 ? 1 : 200));
    const post = (b: object) => fetch(`${base}/api/submit`, { method: "POST", body: JSON.stringify(b) });
    const wrongRules = await post({ runId: r.runId, name: "x", rules: "deadbeef", swats });
    assert.equal(wrongRules.status, 409);
    assert.equal(r.rules, rulesFor(manifest)); // the browser computes the same id from the manifest it loaded
    const ok = await post({ runId: r.runId, name: "  Ada <b>L</b>  ", rules: rulesFor(manifest), swats });
    const body = await ok.json() as { accepted: boolean; hits: number; rank: number };
    assert.equal(ok.status, 200);
    assert.ok(body.accepted && body.rank === 1);
    assert.equal((await post({ runId: r.runId, name: "x", rules: r.rules, swats })).status, 409);
    const lb = await (await fetch(`${base}/api/leaderboard`)).json() as { rows: Array<{ name: string; hits: number }> };
    assert.equal(lb.rows.length, 1);
    assert.equal(lb.rows[0].name, "Ada bLb");
    assert.equal(lb.rows[0].hits, body.hits);
    const bad = await (await fetch(`${base}/api/run`)).json() as { runId: string; rules: string };
    assert.equal((await post({ runId: bad.runId, name: "cheat", rules: bad.rules, swats })).status, 422); // traces were for another seed
    assert.equal((await fetch(`${base}/..%2f..%2fetc/passwd`)).status, 404);
    assert.equal((await fetch(`${base}/brain/manifest.json`)).status, 200);
  } finally { server.close(); }
  assert.equal(rulesId({ a: 1 }), rulesId({ a: 1 }));
  assert.notEqual(rulesId({ a: 1 }), rulesId({ a: 2 }));
});
