// Lab stimuli in TypeScript: the same looms `offline/loom.py` makes, turned into engine frames the way
// `offline/export.py` does (growth rate by central difference, plane position from `offline/heading.py`).
// Pinned to the 50 parity vectors in lab.test.ts, so a Lab trial here is the offline sweep's trial.
import type { Frame } from "../engine/brain.ts";

export const LAB = { durationMs: 400, frameMs: 5, collisionOvershoot: 1.2, controlMatchFraction: 0.8 } as const;
export type LabKind = "expanding" | "receding" | "translating" | "dimming";
export const LAB_KINDS: readonly LabKind[] = ["expanding", "receding", "translating", "dimming"];

const DEG = 180 / Math.PI;

/** theta(t) in degrees at each frame, `loom.Loom.angular_size_deg`. */
export function angularSizeDeg(kind: LabKind, rvMs: number): number[] {
  const n = Math.round(LAB.durationMs / LAB.frameMs) + 1;
  const tc = LAB.durationMs * LAB.collisionOvershoot;
  const t = (i: number) => (i / (n - 1)) * LAB.durationMs;
  if (kind === "expanding") return Array.from({ length: n }, (_, i) => 2 * Math.atan(rvMs / Math.max(tc - t(i), 1e-6)) * DEG);
  if (kind === "receding") return Array.from({ length: n }, (_, i) => 2 * Math.atan(rvMs / (t(i) + tc)) * DEG);
  // translating and dimming hold the size an expanding trial reaches late in its run-up
  const hold = 2 * Math.atan(rvMs / Math.max(tc - LAB.durationMs * LAB.controlMatchFraction, 1e-6)) * DEG;
  return new Array(n).fill(hold);
}

/** Eye-plane position (deg) of a horizontal-plane stimulus at body azimuth `phiDeg`, `heading.plane_position`. */
export function planePosition(phiDeg: number, side: "R" | "L"): [number, number] {
  const phi = phiDeg / DEG;
  const lateral = Math.sin(phi) * (side === "R" ? 1 : -1);
  const ecc = Math.acos(Math.max(-1, Math.min(1, lateral))) * DEG;
  return [(Math.cos(phi) >= 0 ? 1 : -1) * ecc, 0];
}

/** numpy.gradient with uniform spacing: central inside, one-sided at the ends. */
function gradient(y: readonly number[], h: number): number[] {
  const n = y.length;
  return y.map((_, i) => (i === 0 ? (y[1] - y[0]) / h : i === n - 1 ? (y[n - 1] - y[n - 2]) / h : (y[i + 1] - y[i - 1]) / (2 * h)));
}

export interface LabStimulus { kind: LabKind; rvMs: number; azimuthDeg: number }

/** Engine frames and the angular size at each, for one Lab stimulus. */
export function labFrames(s: LabStimulus): { frames: Frame[]; thetaDeg: number[] } {
  const thetaDeg = angularSizeDeg(s.kind, s.rvMs);
  const rate = s.kind === "translating" ? thetaDeg.map(() => 0) : gradient(thetaDeg, LAB.frameMs).map((r) => Math.max(r, 0));
  const [xr, yr] = planePosition(s.azimuthDeg, "R"), [xl, yl] = planePosition(s.azimuthDeg, "L");
  return { frames: rate.map((r) => [r, xr, yr, xl, yl] as Frame), thetaDeg };
}

export interface LabResult {
  stimulus: LabStimulus;
  gfFirstMs: number | null;
  thetaAtGfDeg: number | null;
  mode: "short" | "long" | "none";
  headingSide: "left" | "right" | null;
  /** GF membrane potential per frame, right and left, for the live trace. */
  gfVoltage: { right: number[]; left: number[] };
  thetaDeg: number[];
}

/** Run one Lab stimulus through the engine, streaming it frame by frame as the game does. */
export function runLab(brain: import("../engine/brain.ts").FlyBrain, s: LabStimulus): LabResult {
  const { frames, thetaDeg } = labFrames(s);
  const gfs = brain.manifest.roles.gf, sides = brain.manifest.roles.target_side;
  const gfR = gfs[sides.indexOf("R")], gfL = gfs[sides.indexOf("L")];
  const gfVoltage = { right: [] as number[], left: [] as number[] };
  brain.reset();
  for (const f of frames) {
    brain.frame(f);
    gfVoltage.right.push(brain.voltage(gfR));
    gfVoltage.left.push(brain.voltage(gfL));
  }
  brain.finish();
  const o = brain.outcome();
  let thetaAtGfDeg: number | null = null;
  if (o.gfFirstMs !== null) {
    const i = o.gfFirstMs / LAB.frameMs, lo = Math.min(Math.floor(i), thetaDeg.length - 1), hi = Math.min(lo + 1, thetaDeg.length - 1);
    thetaAtGfDeg = thetaDeg[lo] + (thetaDeg[hi] - thetaDeg[lo]) * (i - lo);
  }
  return { stimulus: s, gfFirstMs: o.gfFirstMs, thetaAtGfDeg, mode: o.mode, headingSide: o.headingSide, gfVoltage, thetaDeg };
}
