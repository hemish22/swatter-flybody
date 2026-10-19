// Duel mode: a second copy of the fly's brain drives the swatter against the first copy, which is the fly.
//
// Brain A is exactly the game's fly (Round, unchanged). Brain B is another FlyBrain; it sees the fly as a looming
// disc, steers the swatter by which of its two eyes fires more, and strikes when its own escape circuit decides to
// "jump". This file turns B's spikes into the cursor stream (`Input`) that Round already accepts, so the rules, the
// stage, the sound and the replay all work on a duel as they do on a human round.
//
// Only B's wiring is real. What B sees (two forward-facing eyes, the fly as a disc of `preyRadiusMm`), how its
// activity becomes motion (a fixed creep, a turn from the eye difference) and the sensitivity `gainB` are our
// mapping; docs/duel.md says so. Pure and deterministic like world.ts: a duel is a function of (seed, params).

import { FlyBrain, type Frame } from "../engine/brain.ts";
import { CONFIG, Round, decideTakeoff, growthRate, rng, type Config, type Input } from "./world.ts";

export interface DuelParams {
  /** Sensitivity of B's looming detectors: B's growth rate is multiplied by this before it drives the engine
   *  (the drive is linear in the rate, so this equals scaling the engine's input gain). */
  gainB: number;
  /** How fast the swatter creeps toward the fly, mm/ms (= m/s). */
  creepMmPerMs: number;
  /** Turn rate, radians per ms, at full eye difference. */
  steerGain: number;
  /** Radius of the disc B sees the fly as. The real fly is far smaller; this is part of the sensitivity mapping. */
  preyRadiusMm: number;
  /** Each eye looks this many degrees to its own side of straight ahead, so a fly dead ahead excites both equally. */
  eyeOffsetDeg: number;
  /** How far from the fly the swatter starts, mm (shortened if that would leave the table). */
  startMm: number;
  /** Largest angle between the swatter's heading and the fly at the start, degrees. */
  startBearingDeg: number;
}

/** Fitted by game/duel_tune.ts; see docs/duel.md. */
export const DUEL: DuelParams = {
  gainB: 3,
  creepMmPerMs: 0.035,
  steerGain: 0.004,
  preyRadiusMm: 10,
  eyeOffsetDeg: 25,
  startMm: 160,
  startBearingDeg: 40,
};

const DEG = 180 / Math.PI;
/** The swatter strikes anyway if it has not by this multiple of the time the approach should take, or once on top of the fly. */
const PATIENCE = 2.5;
const ARRIVED_MM = 6;

export interface SwatterState { x: number; y: number; heading: number }

export class DuelRun {
  readonly round: Round;
  readonly brainB: FlyBrain;
  readonly params: DuelParams;
  /** The swatter as B has moved it; Round holds the same position (and its height). */
  readonly sw: SwatterState;
  /** ms of each of B's neurons' latest spike (-Infinity if none), for drawing B's circuit. */
  readonly lastSpikeMs: Float64Array;
  /** B's giant-fiber voltage (mV, right then left) and the fly's angular size in B's view, per tick, for B's strip. */
  readonly gfVoltage: { right: number[]; left: number[] } = { right: [], left: [] };
  readonly thetaDeg: number[] = [];
  /** Horizontal distance from the swatter to the fly at the start of each tick, mm (for the probe and the tuner). */
  readonly distMm: number[] = [];
  /** The cursor stream B has produced; feeding it to simulateRound reproduces the fly's side of the duel. */
  readonly trace: Input[] = [];
  tGf: number | null = null;
  tPar: number | null = null;
  /** When B decided to strike and how far from the fly it was (horizontal, mm); null until it does. */
  strikeAtMs: number | null = null;
  strikeDistMm: number | null = null;
  strikeReason: "spikes" | "arrived" | "patience" | null = null;
  /** Latest steering signal in [-1, 1] (positive: fly is on the right) and the eye sums behind it, for display. */
  steer = 0;
  lastRate = 0;

  private cfg: Config;
  private cfgB: Config;
  private prevSpikes = 0;
  private roles: { gf: Set<number>; par: Set<number> };
  private gfIndex: { right: number; left: number };
  private eyes: { right: number[]; left: number[] };
  private releasing = false;
  private striking = false;

  constructor(brainA: FlyBrain, brainB: FlyBrain, seed: number, params: DuelParams = DUEL, cfg: Config = CONFIG) {
    this.cfg = cfg;
    this.params = params;
    this.brainB = brainB;
    this.round = new Round(brainA, seed, cfg);
    // B sees the fly as a disc of its own radius, from the height the swatter hovers at
    this.cfgB = { ...cfg, swatterHalfWidthMm: params.preyRadiusMm, eyeHeightMm: cfg.hoverHeightMm };
    const r = rng((seed ^ 0x9e3779b9) >>> 0);
    const fly = this.round.fly, phi = r() * 2 * Math.PI;
    const ux = Math.cos(phi), uy = Math.sin(phi);
    // shorten the start until it is on the table; the fly is inside the central 60%, so this keeps a sensible distance
    const tx = ux > 0 ? (cfg.arenaMm.w / 2 - 6 - fly.x) / ux : ux < 0 ? (-cfg.arenaMm.w / 2 + 6 - fly.x) / ux : Infinity;
    const ty = uy > 0 ? (cfg.arenaMm.h / 2 - 6 - fly.y) / uy : uy < 0 ? (-cfg.arenaMm.h / 2 + 6 - fly.y) / uy : Infinity;
    const d = Math.min(params.startMm, tx, ty);
    const x = fly.x + ux * d, y = fly.y + uy * d;
    const toFly = Math.atan2(fly.y - y, fly.x - x);
    this.sw = { x, y, heading: toFly + (r() * 2 - 1) * (params.startBearingDeg / DEG) };
    this.lastSpikeMs = new Float64Array(brainB.manifest.neurons).fill(-Infinity);
    const m = brainB.manifest;
    this.roles = { gf: new Set(m.roles.gf), par: new Set(m.roles.parallel) };
    const side = (s: string) => m.roles.gf[m.roles.target_side.indexOf(s)];
    this.gfIndex = { right: side("R"), left: side("L") };
    this.eyes = { right: [], left: [] };
    for (let i = 0; i < brainB.drive.neuron.length; i++) (brainB.drive.eye[i] === 0 ? this.eyes.right : this.eyes.left).push(brainB.drive.neuron[i]);
    brainB.reset();
  }

  get timeMs(): number { return this.round.timeMs; }
  get done(): boolean { return this.round.phase === "done"; }
  /** Horizontal distance from the swatter to the fly, mm. */
  get distanceMm(): number { return Math.hypot(this.round.fly.x - this.sw.x, this.round.fly.y - this.sw.y); }

  /** One 5 ms tick: B looks, B's circuit runs, B moves or strikes, then the fly's circuit reacts. */
  tick(): void {
    if (this.done) return;
    const c = this.cfg, p = this.params, dt = c.tickMs, t = this.round.timeMs;
    const fly = this.round.fly;
    this.distMm.push(this.distanceMm);

    if (!this.striking) {
      // what B sees: the fly as a disc, below and ahead of the swatter
      const dx = fly.x - this.sw.x, dy = fly.y - this.sw.y, hz = c.hoverHeightMm;
      const fwd = dx * Math.cos(this.sw.heading) + dy * Math.sin(this.sw.heading);
      const rgt = dx * Math.sin(this.sw.heading) - dy * Math.cos(this.sw.heading);
      const bearing = Math.atan2(rgt, fwd) * DEG;
      const elev = -Math.atan2(hz, Math.hypot(dx, dy)) * DEG;
      const v = p.creepMmPerMs, vx = v * Math.cos(this.sw.heading), vy = v * Math.sin(this.sw.heading);
      // the fly closes on B at -v; growthRate wants the object's velocity relative to the eye
      const rate = Math.max(0, growthRate(this.sw, { x: fly.x, y: fly.y, z: 0 }, { x: -vx, y: -vy, z: 0 }, this.cfgB)) * p.gainB;
      const D = Math.hypot(dx, dy, hz);
      this.thetaDeg.push(2 * Math.atan(p.preyRadiusMm / D) * DEG);
      this.lastRate = rate;
      const frame: Frame = [rate, bearing - p.eyeOffsetDeg, elev, bearing + p.eyeOffsetDeg, elev];
      this.brainB.frame(frame);
      this.gfVoltage.right.push(this.brainB.voltage(this.gfIndex.right));
      this.gfVoltage.left.push(this.brainB.voltage(this.gfIndex.left));
      this.readSpikes();

      // B decides to strike: its own takeoff rule, or it has run out of road
      const take = decideTakeoff(this.tGf, this.tPar, t, c);
      const dist = Math.hypot(dx, dy);
      if (take && take.leaveMs <= t) this.beginStrike(t, dist, "spikes");
      else if (dist <= ARRIVED_MM) this.beginStrike(t, dist, "arrived");
      else if (t >= (p.startMm / p.creepMmPerMs) * PATIENCE) this.beginStrike(t, dist, "patience");

      if (!this.striking) {
        // steer toward the eye that fires more, then creep
        const sum = (ids: number[]) => ids.reduce((a, j) => a + Math.max(0, this.brainB.voltage(j) - this.brainB.manifest.lif["v0"]), 0);
        const sR = sum(this.eyes.right), sL = sum(this.eyes.left);
        this.steer = sR + sL > 1e-6 ? (sR - sL) / (sR + sL) : 0;
        this.sw.heading -= p.steerGain * this.steer * dt; // positive steer: fly on the right, turn clockwise
        this.sw.x = Math.max(-c.arenaMm.w / 2, Math.min(c.arenaMm.w / 2, this.sw.x + vx * dt));
        this.sw.y = Math.max(-c.arenaMm.h / 2, Math.min(c.arenaMm.h / 2, this.sw.y + vy * dt));
      }
    }
    const inp: Input = { x: this.sw.x, y: this.sw.y, down: !this.releasing };
    this.trace.push(inp);
    this.round.tick(inp);
  }

  private beginStrike(t: number, dist: number, reason: "spikes" | "arrived" | "patience"): void {
    this.striking = true;
    this.releasing = true;
    this.strikeAtMs = t;
    this.strikeDistMm = dist;
    this.strikeReason = reason;
  }

  private readSpikes(): void {
    const n = this.brainB.spikeCount;
    if (n === this.prevSpikes) return;
    this.prevSpikes = n;
    for (const [s, j] of this.brainB.spikes()) {
      const ms = s * this.brainB.dtMs;
      if (ms > this.lastSpikeMs[j]) this.lastSpikeMs[j] = ms;
      if (this.roles.gf.has(j) && (this.tGf === null || ms < this.tGf)) this.tGf = ms;
      if (this.roles.par.has(j) && (this.tPar === null || ms < this.tPar)) this.tPar = ms;
    }
  }
}

/** Run a whole duel and return it; for the tests, the probe and the tuner. */
export function runDuel(brainA: FlyBrain, brainB: FlyBrain, seed: number, params: DuelParams = DUEL, cfg: Config = CONFIG): DuelRun {
  const d = new DuelRun(brainA, brainB, seed, params, cfg);
  for (let i = 0; i < 100000 && !d.done; i++) d.tick();
  return d;
}
