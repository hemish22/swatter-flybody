// The reaction chain shown beside the arena: what the fly's brain did, in order, in plain words.
// Pure functions of numbers the engine already produced, so it can be tested without a DOM or a WebGL context.
// Nothing here feeds back into a round; it only describes one.

export type StepState = "idle" | "live" | "done";

export interface Step {
  title: string;
  detail: string;
  state: StepState;
  /** Time of the event in ms from round start, or null if it has not happened. */
  atMs: number | null;
  /** A reminder about what is modelled rather than measured, shown under the step. */
  caveat?: string;
}

export interface Snapshot {
  timeMs: number;
  thetaDeg: number;
  /** Swatter growth rate on the latest tick, deg/ms. */
  rateDegPerMs: number;
  /** Start times of every spell of growth so far, ms. */
  growthStartsMs: readonly number[];
  lplc2Active: number;
  lplc2Driven: number;
  firstLplc2Ms: number | null;
  firstGfMs: number | null;
  /** Swatter angular size when the giant fiber first spiked. */
  thetaAtGfDeg: number | null;
  thresholdMv: number;
  takeoff: { mode: "short" | "long"; leaveMs: number } | null;
  headingSide: "left" | "right" | null;
}

/** A growth rate below this is the cursor drifting, not an approach. deg/ms. */
export const GROWTH_EPS = 0.02;

/** The latest growth spell that began at or before `t`: the approach an event at `t` was a reaction to. */
export function growthStartBefore(starts: readonly number[], t: number): number | null {
  let best: number | null = null;
  for (const s of starts) if (s <= t) best = s;
  return best ?? (starts.length ? starts[0] : null);
}

const ms = (v: number) => `${v.toFixed(v < 10 ? 1 : 0)} ms`;

export function deriveStory(s: Snapshot): Step[] {
  const growing = s.rateDegPerMs >= GROWTH_EPS;
  const anchor = (t: number | null) => (t === null ? null : growthStartBefore(s.growthStartsMs, t));
  const since = (t: number | null) => {
    const a = anchor(t);
    return t === null || a === null ? null : Math.max(0, t - a);
  };

  const grew = s.growthStartsMs.length > 0;
  const step1: Step = {
    title: "It sees the swatter grow",
    state: growing ? "live" : grew ? "done" : "idle",
    atMs: s.growthStartsMs.length ? s.growthStartsMs[s.growthStartsMs.length - 1] : null,
    detail: !grew
      ? "Waiting for the swatter to come closer."
      : `The swatter spans ${s.thetaDeg.toFixed(0)}° of its view` + (growing ? `, growing ${s.rateDegPerMs.toFixed(2)}° per ms.` : "."),
  };

  const dLplc2 = since(s.firstLplc2Ms);
  const step2: Step = {
    title: "Looming detectors fire",
    state: s.firstLplc2Ms !== null ? "done" : growing ? "live" : "idle",
    atMs: s.firstLplc2Ms,
    detail: s.firstLplc2Ms === null
      ? "No LPLC2 neuron has fired yet."
      : `${s.lplc2Active} of ${s.lplc2Driven} LPLC2 neurons firing now; the first fired ${dLplc2 === null || dLplc2 < 0.5 ? "immediately" : ms(dLplc2) + " in"}.`,
    caveat: "We supply this detector's input (how fast the swatter grows). The wiring after it is the real connectome.",
  };

  const dGf = since(s.firstGfMs);
  const step3: Step = {
    title: "Giant fiber crosses threshold",
    state: s.firstGfMs !== null ? "done" : s.firstLplc2Ms !== null ? "live" : "idle",
    atMs: s.firstGfMs,
    detail: s.firstGfMs === null
      ? `Still below ${s.thresholdMv.toFixed(0)} mV.`
      : `Fired ${dGf === null ? "" : ms(dGf) + " in, "}when the swatter was ${s.thetaAtGfDeg === null ? "?" : s.thetaAtGfDeg.toFixed(0) + "°"} wide.`,
  };

  const dLeave = s.takeoff ? since(s.takeoff.leaveMs) : null;
  const step4: Step = {
    title: "Takeoff",
    state: s.takeoff ? "done" : s.firstGfMs !== null ? "live" : "idle",
    atMs: s.takeoff ? s.takeoff.leaveMs : null,
    detail: !s.takeoff
      ? "The fly is still on the table."
      : `${s.takeoff.mode}-mode takeoff` + (dLeave === null ? "" : `, ${ms(dLeave)} after the swatter began to grow`)
        + (s.headingSide ? `. It turns ${s.headingSide}.` : ". The model gives no side for this direction.")
  };
  return [step1, step2, step3, step4];
}

/** Plain-language result line for the banner. */
export function resultLine(outcome: string, marginMs: number | null): string {
  switch (outcome) {
    case "hit": return "Hit. The swatter landed before the fly left.";
    case "escaped": return marginMs === null ? "Escaped." : `Escaped. It left ${ms(marginMs)} before the swat landed.`;
    case "spooked": return "Spooked. It took off before you struck.";
    case "miss": return "Missed. The swatter landed away from the fly.";
    default: return "Timed out.";
  }
}
