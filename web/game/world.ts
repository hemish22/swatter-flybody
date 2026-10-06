// The game's world model: a swatter above a table, a fly on it, and the frames the fly's brain sees.
//
// Pure and deterministic: a round is a function of (seed, cursor trace), so the leaderboard server can
// re-simulate it with the same code in Node. No DOM, no clock, no Math.random.
//
// Units: mm and ms; speeds in mm/ms, which is numerically m/s. World x is right, y is up the screen, z is
// up from the table. Body frame: x forward, y to the fly's right, z up. Eye geometry follows
// offline/heading.py (right eye axis +y, left eye -y, plane x = forward, plane y = up), and
// web/game/fixtures.json pins the two to each other.
//
// Assumptions that are NOT measured: the lateral eye axes, the eye height, and the 6.87 ms takeoff
// delay (the plan's only number for a short takeoff; see docs/game.md).

import { FlyBrain, type Frame } from "../engine/brain.ts";

export interface Config {
  tickMs: number;
  swatterHalfWidthMm: number;
  hoverHeightMm: number;
  eyeHeightMm: number;
  /**
   * Where the strike speed comes from. "charge": hold the button, release to strike; speed grows with the hold
   * (the default; see docs/game.md for why). "cursor": the plan's rule, cursor speed over the last 100 ms at
   * button-down, which this brain makes unwinnable because the flick is itself a loom.
   */
  strikeSpeedSource: "charge" | "cursor";
  /** Hold time that reaches the maximum strike speed. */
  chargeMs: number;
  /** mm/ms (= m/s): the plan's 0.6-5 m/s */
  strikeSpeedMin: number;
  strikeSpeedMax: number;
  strikeSpeedWindowMs: number;
  arenaMm: { w: number; h: number };
  jumpDelayMs: number;
  raiseWindowMs: number;
  hoverTimeoutMs: number;
}

export const CONFIG: Config = {
  tickMs: 5,
  swatterHalfWidthMm: 50,
  hoverHeightMm: 40, // the plan says 150 and 'tunable'; at 150 no strike can land (docs/game.md)
  eyeHeightMm: 1,
  strikeSpeedSource: "charge",
  chargeMs: 1000,
  strikeSpeedMin: 0.6, // mm/ms (= m/s), the plan's 0.6-5 m/s
  strikeSpeedMax: 5.0,
  strikeSpeedWindowMs: 100,
  arenaMm: { w: 600, h: 400 },
  /** Short takeoff: leg contact is lost this long after the giant-fiber spike. The plan's bound for a short takeoff. */
  jumpDelayMs: 6.87,
  /** Wing raise that precedes a long takeoff, and the window in which a giant-fiber spike still makes it short. */
  raiseWindowMs: 6.87,
  hoverTimeoutMs: 60000,
};


/** Cursor position on the table in mm, and whether the button is down. One per 5 ms tick. */
export interface Input { x: number; y: number; down: boolean }

export type Outcome = "hit" | "escaped" | "spooked" | "miss" | "timeout";

export interface Card {
  outcome: Outcome;
  /** r / v of the strike, ms. Null if the round ended before a strike. */
  strikeRvMs: number | null;
  /** Angular size of the swatter at the moment the giant fiber first spiked. */
  thetaAtGfDeg: number | null;
  mode: "short" | "long" | "none";
  /** Impact time minus the time the fly lost leg contact: positive means it was gone before the swat landed. */
  marginMs: number | null;
  headingSide: "left" | "right" | null;
}

// ---------------------------------------------------------------------------------------------- rng

/** mulberry32: small, fast, and identical in every JS engine. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------------------------------ geometry

const DEG = 180 / Math.PI;

export interface EyeView {
  /** Disc centre distance from the eye, mm. */
  distance: number;
  /** Angular diameter, degrees. */
  thetaDeg: number;
  /** Plane coordinates of the disc on each eye, degrees. */
  right: [number, number];
  left: [number, number];
}

/** Direction of `swatter` from the fly's eye, in the body frame, and what each eye makes of it. */
export function eyeView(
  fly: { x: number; y: number; heading: number },
  swatter: { x: number; y: number; z: number },
  cfg: Config = CONFIG,
): EyeView {
  const dx = swatter.x - fly.x, dy = swatter.y - fly.y, dz = swatter.z - cfg.eyeHeightMm;
  const fwd = dx * Math.cos(fly.heading) + dy * Math.sin(fly.heading);
  const rgt = dx * Math.sin(fly.heading) - dy * Math.cos(fly.heading);
  const D = Math.hypot(fwd, rgt, dz);
  const u = [fwd / D, rgt / D, dz / D];
  const phi = Math.atan2(u[2], u[0]); // plane angle: forward is +x, up is +y
  const eccR = Math.acos(Math.max(-1, Math.min(1, u[1]))) * DEG;
  const eccL = Math.acos(Math.max(-1, Math.min(1, -u[1]))) * DEG;
  return {
    distance: D,
    thetaDeg: 2 * Math.atan(cfg.swatterHalfWidthMm / D) * DEG,
    right: [eccR * Math.cos(phi), eccR * Math.sin(phi)],
    left: [eccL * Math.cos(phi), eccL * Math.sin(phi)],
  };
}

/** d(theta)/dt in deg/ms for a swatter moving at `vel` (mm/ms) relative to a stationary fly. Positive = growing. */
export function growthRate(
  fly: { x: number; y: number },
  swatter: { x: number; y: number; z: number },
  vel: { x: number; y: number; z: number },
  cfg: Config = CONFIG,
): number {
  const dx = swatter.x - fly.x, dy = swatter.y - fly.y, dz = swatter.z - cfg.eyeHeightMm;
  const D = Math.hypot(dx, dy, dz);
  const dDdt = (dx * vel.x + dy * vel.y + dz * vel.z) / D;
  const r = cfg.swatterHalfWidthMm;
  return -((2 * r) / (D * D + r * r)) * dDdt * DEG; // theta = 2 atan(r/D)
}

// -------------------------------------------------------------------------------------------- takeoff

export interface Takeoff {
  mode: "short" | "long";
  /** When the fly loses leg contact, ms from round start. */
  leaveMs: number;
}

/**
 * The plan's mode rule applied to the spikes seen so far (`observedMs` of them).
 * Short: the giant fiber spikes first, or within the raise window of the parallel DNs; the fly leaves one
 * jump delay later. Long: the parallel DNs lead and the raise completes without a giant-fiber spike.
 * Returns null while undecided. (In this model every escape is short in practice: docs/lif_fit.md.)
 */
export function decideTakeoff(tGfMs: number | null, tParMs: number | null, observedMs: number, cfg: Config = CONFIG): Takeoff | null {
  const w = cfg.raiseWindowMs;
  if (tGfMs !== null && (tParMs === null || tGfMs <= tParMs + w)) return { mode: "short", leaveMs: tGfMs + cfg.jumpDelayMs };
  if (tParMs !== null && observedMs >= tParMs + w) return { mode: "long", leaveMs: tParMs + w };
  return null;
}

// ------------------------------------------------------------------------------------------ the round

export class Round {
  readonly fly: { x: number; y: number; heading: number };
  phase: "hover" | "strike" | "done" = "hover";
  tickIndex = 0;
  card: Card | null = null;
  swatter = { x: 0, y: 0, z: CONFIG.hoverHeightMm };
  /** The swatter's angular size at each tick, for the card and the replay. */
  readonly thetaDeg: number[] = [];
  /** Giant-fiber membrane potential (mV) at each tick, right and left, for the overlay. */
  readonly gfVoltage: { right: number[]; left: number[] } = { right: [], left: [] };
  /** Swatter growth rate (deg/ms) fed to the engine on the latest tick, and the engine behind it, for the population panel. */
  lastRate = 0;
  /** ms of each neuron's latest spike so far (-Infinity if none), for the population panel. */
  readonly lastSpikeMs: Float64Array;
  get engine(): FlyBrain { return this.brain; }

  private brain: FlyBrain;
  private cfg: Config;
  private history: Array<[number, number]> = [];
  private prevDown = false;
  private holdStartTick = -1;
  private strikeTick = -1;
  private strikeDurationMs = 0;
  private strikeStartMs = 0;
  private strikeSpeed = 0;
  private takeoff: Takeoff | null = null;
  private seenSpikes = 0;
  private tGf: number | null = null;
  private tPar: number | null = null;
  private roleSets: { gf: Set<number>; par: Set<number> };
  private gfIndex: { right: number; left: number };

  constructor(brain: FlyBrain, seed: number, cfg: Config = CONFIG) {
    this.brain = brain;
    this.cfg = cfg;
    const r = rng(seed);
    this.fly = {
      x: (r() - 0.5) * cfg.arenaMm.w * 0.6,
      y: (r() - 0.5) * cfg.arenaMm.h * 0.6,
      heading: r() * 2 * Math.PI,
    };
    this.lastSpikeMs = new Float64Array(brain.manifest.neurons).fill(-Infinity);
    this.roleSets = { gf: new Set(brain.manifest.roles.gf), par: new Set(brain.manifest.roles.parallel) };
    const side = (s: string) => brain.manifest.roles.gf[brain.manifest.roles.target_side.indexOf(s)];
    this.gfIndex = { right: side("R"), left: side("L") }; // target_side is aligned with gf first
    brain.reset();
  }

  get timeMs(): number { return this.tickIndex * this.cfg.tickMs; }

  /** 0 to 1 while the button is held in the hover phase, else 0. */
  get charge(): number {
    if (this.phase !== "hover" || this.holdStartTick < 0) return 0;
    return Math.min(1, ((this.tickIndex - this.holdStartTick) * this.cfg.tickMs) / this.cfg.chargeMs);
  }

  /** When the fly lost leg contact (ms from round start), once the spikes have decided it; else null. */
  get leftAtMs(): number | null { return this.takeoff ? this.takeoff.leaveMs : null; }

  /** First giant-fiber and parallel-DN spike seen so far, ms. */
  get firstSpikes(): { gf: number | null; parallel: number | null } { return { gf: this.tGf, parallel: this.tPar }; }

  /** Advance one 5 ms tick with this cursor state. */
  tick(inp: Input): void {
    if (this.phase === "done") return;
    const c = this.cfg, k = this.tickIndex, t = k * c.tickMs;
    this.history.push([inp.x, inp.y]);

    // --- swatter state and velocity for this tick
    let vel = { x: 0, y: 0, z: 0 };
    if (this.phase === "hover") {
      const prev = k > 0 ? this.history[k - 1] : [inp.x, inp.y];
      vel = { x: (inp.x - prev[0]) / c.tickMs, y: (inp.y - prev[1]) / c.tickMs, z: 0 };
      this.swatter = { x: inp.x, y: inp.y, z: c.hoverHeightMm };
      if (c.strikeSpeedSource === "cursor") {
        if (inp.down && !this.prevDown) this.startStrike(k, this.cursorSpeed(k));
      } else {
        if (inp.down && !this.prevDown) this.holdStartTick = k;
        if (!inp.down && this.prevDown && this.holdStartTick >= 0) {
          const held = (k - this.holdStartTick) * c.tickMs;
          this.startStrike(k, c.strikeSpeedMin + (c.strikeSpeedMax - c.strikeSpeedMin) * Math.min(1, held / c.chargeMs));
          this.holdStartTick = -1;
        }
      }
    }
    let impactMs: number | null = null;
    if (this.phase === "strike") {
      const since = t - this.strikeStartMs;
      if (since >= this.strikeDurationMs - 1e-9) impactMs = this.strikeStartMs + this.strikeDurationMs;
      this.swatter.z = impactMs !== null ? 0 : Math.max(0, c.hoverHeightMm - this.strikeSpeed * since);
      vel = { x: 0, y: 0, z: since === 0 ? 0 : -this.strikeSpeed };
    }
    this.prevDown = inp.down;

    // --- the fly sees this frame
    const view = eyeView(this.fly, this.swatter, c);
    this.thetaDeg.push(view.thetaDeg);
    const rate = Math.max(0, growthRate(this.fly, this.swatter, vel, c));
    this.lastRate = rate;
    const frame: Frame = [rate, view.right[0], view.right[1], view.left[0], view.left[1]];
    if (impactMs !== null) {
      // the swatter lands between ticks: the last frame covers only the time up to the landing
      this.brain.frameFor(frame, Math.max(1, Math.min(25, Math.round((impactMs - (t - c.tickMs)) / this.brain.dtMs))));
      this.brain.finish(); // the last sample, at the landing frame's own drive
      this.readSpikes(impactMs);
      return this.resolve(impactMs);
    }
    this.brain.frame(frame);
    this.gfVoltage.right.push(this.brain.voltage(this.gfIndex.right));
    this.gfVoltage.left.push(this.brain.voltage(this.gfIndex.left));
    this.readSpikes(t);

    if (this.takeoff && this.takeoff.leaveMs <= t && this.phase === "hover") return this.end("spooked", t);
    if (this.phase === "hover" && t >= c.hoverTimeoutMs) return this.end("timeout", t);
    this.tickIndex++;
  }

  /** Cursor speed over the last `strikeSpeedWindowMs`, mm/ms. */
  private cursorSpeed(k: number): number {
    const c = this.cfg;
    const n = Math.min(Math.round(c.strikeSpeedWindowMs / c.tickMs), k);
    const a = this.history[k], b = this.history[k - n];
    return n === 0 ? 0 : Math.hypot(a[0] - b[0], a[1] - b[1]) / (n * c.tickMs);
  }

  private startStrike(k: number, speed: number): void {
    const c = this.cfg;
    const v = Math.min(c.strikeSpeedMax, Math.max(c.strikeSpeedMin, speed));
    // the landing time is rounded to the engine's 0.2 ms step, so the strike speed is that of the rounded time
    this.strikeDurationMs = Math.max(this.brain.dtMs, Math.round(c.hoverHeightMm / v / this.brain.dtMs) * this.brain.dtMs);
    this.strikeSpeed = c.hoverHeightMm / this.strikeDurationMs;
    this.strikeStartMs = k * c.tickMs;
    this.strikeTick = k;
    this.phase = "strike";
  }

  private readSpikes(observedMs: number): void {
    const n = this.brain.spikeCount;
    if (n !== this.seenSpikes) {
      this.seenSpikes = n;
      for (const [s, j] of this.brain.spikes()) {
        const ms = s * this.brain.dtMs;
        if (ms > this.lastSpikeMs[j]) this.lastSpikeMs[j] = ms;
        if (this.roleSets.gf.has(j) && (this.tGf === null || ms < this.tGf)) this.tGf = ms;
        if (this.roleSets.par.has(j) && (this.tPar === null || ms < this.tPar)) this.tPar = ms;
      }
    }
    this.takeoff = decideTakeoff(this.tGf, this.tPar, observedMs, this.cfg);
  }

  private end(outcome: Outcome, t: number): void {
    this.phase = "done";
    const heading = this.brain.outcome();
    const thetaAt = (ms: number) => {
      const i = ms / this.cfg.tickMs, lo = Math.min(Math.floor(i), this.thetaDeg.length - 1), hi = Math.min(lo + 1, this.thetaDeg.length - 1);
      return this.thetaDeg[lo] + (this.thetaDeg[hi] - this.thetaDeg[lo]) * (i - lo);
    };
    this.card = {
      outcome,
      strikeRvMs: this.strikeDurationMs ? this.cfg.swatterHalfWidthMm / this.strikeSpeed : null,
      thetaAtGfDeg: this.tGf === null ? null : thetaAt(this.tGf),
      mode: this.takeoff ? this.takeoff.mode : "none",
      marginMs: this.takeoff && this.strikeDurationMs ? t - this.takeoff.leaveMs : null,
      headingSide: heading.headingSide,
    };
  }

  private resolve(t: number): void {
    const inFootprint = Math.hypot(this.swatter.x - this.fly.x, this.swatter.y - this.fly.y) <= this.cfg.swatterHalfWidthMm;
    const left = this.takeoff !== null && this.takeoff.leaveMs <= t;
    this.end(!inFootprint ? "miss" : left ? "escaped" : "hit", t);
  }
}

/** Replay a whole round from its seed and cursor trace: what the leaderboard server does. */
export function simulateRound(brain: FlyBrain, seed: number, trace: readonly Input[], cfg: Config = CONFIG): Round {
  const round = new Round(brain, seed, cfg);
  for (const inp of trace) {
    round.tick(inp);
    if (round.phase === "done") break;
  }
  return round;
}
