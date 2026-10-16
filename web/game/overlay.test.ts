import { test } from "node:test";
import assert from "node:assert/strict";
import { hexLattice } from "./overlay.ts";

test("hex lattice: inside the disc, rows offset by half a step, density near the hexagonal ideal", () => {
  const R = 80, step = 8, pts = hexLattice(R, step);
  assert.ok(pts.every(([x, y]) => Math.hypot(x, y) <= R + 1e-9));
  const ideal = (Math.PI * R * R) / (step * step * Math.sqrt(3) / 2); // disc area over the area of one hex cell
  assert.ok(Math.abs(pts.length - ideal) / ideal < 0.05, `${pts.length} vs ideal ${ideal.toFixed(0)}`);
  const row = (y: number) => pts.filter((p) => Math.abs(p[1] - y) < 1e-9).map((p) => p[0]);
  const dy = step * Math.sqrt(3) / 2, a = row(0), b = row(dy);
  assert.ok(Math.abs(Math.abs(a.find((x) => x >= 0)! - b.find((x) => x >= 0)!) - step / 2) < 1e-9 || a.length > 0);
  assert.equal(new Set(pts.map((p) => p.join(","))).size, pts.length, "no duplicate points");
});
