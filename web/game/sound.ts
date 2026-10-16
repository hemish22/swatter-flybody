// Sound, synthesised in the browser with Web Audio so there are no assets to ship.
//
// Same rule as the pictures: sound only plays back what the engine already produced. A tick is a spike that
// happened, the zap is the giant fiber's first spike, the buzz is the takeoff the spikes decided. Nothing here is
// read back by the rules. The pure part (`diff`) decides which sounds a change of state calls for, so it is testable
// without an audio device; `Sfx` turns those events into tones.

import type { Outcome } from "./world.ts";

export interface SoundState {
  /** Identifies the round being shown, so a new round starts from a clean baseline (no burst of sound). */
  roundId: number;
  phase: "hover" | "strike" | "done";
  /** Total spikes the engine has produced this round. */
  spikeCount: number;
  gfSpiked: boolean;
  /** The fly has left the table at the time being shown. */
  left: boolean;
  outcome: Outcome | null;
}

export type SoundEvent =
  | { kind: "tick"; n: number }
  | { kind: "zap" }
  | { kind: "strike" }
  | { kind: "result"; outcome: Outcome }
  | { kind: "takeoff" };

export function diff(prev: SoundState | null, cur: SoundState): SoundEvent[] {
  if (!prev || prev.roundId !== cur.roundId) return []; // first sight of a round only sets the baseline
  const out: SoundEvent[] = [];
  if (cur.spikeCount > prev.spikeCount) out.push({ kind: "tick", n: cur.spikeCount - prev.spikeCount });
  if (!prev.gfSpiked && cur.gfSpiked) out.push({ kind: "zap" });
  if (prev.phase === "hover" && cur.phase === "strike") out.push({ kind: "strike" });
  if (!prev.left && cur.left) out.push({ kind: "takeoff" });
  if (prev.phase !== "done" && cur.phase === "done" && cur.outcome) out.push({ kind: "result", outcome: cur.outcome });
  return out;
}

export class Sfx {
  enabled = true;
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private noise: AudioBuffer | null = null;
  private charge: { osc: OscillatorNode; gain: GainNode } | null = null;
  private lastTick = 0;

  /** Must be called from a user gesture the first time; later calls are free. */
  unlock(): void {
    if (this.ctx) { if (this.ctx.state === "suspended") void this.ctx.resume(); return; }
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.32;
    this.master.connect(this.ctx.destination);
    const len = Math.floor(this.ctx.sampleRate * 0.6);
    this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    let a = 12345; // a fixed little generator: the same noise every session
    for (let i = 0; i < len; i++) { a = (Math.imul(a, 1664525) + 1013904223) >>> 0; d[i] = (a / 2 ** 31 - 1); }
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    if (!on) this.setCharge(0);
  }

  private ready(): boolean { return this.enabled && !!this.ctx && !!this.master && this.ctx.state === "running"; }

  private tone(freq: number, dur: number, type: OscillatorType, gain: number, to = freq, when = 0): void {
    const c = this.ctx!, t = c.currentTime + when;
    const o = c.createOscillator(), g = c.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    if (to !== freq) o.frequency.exponentialRampToValueAtTime(Math.max(20, to), t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + Math.min(0.01, dur / 3));
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(this.master!);
    o.start(t); o.stop(t + dur + 0.02);
  }

  private burst(dur: number, from: number, to: number, q: number, gain: number, type: BiquadFilterType = "bandpass", when = 0): void {
    const c = this.ctx!, t = c.currentTime + when;
    const src = c.createBufferSource(); src.buffer = this.noise;
    const f = c.createBiquadFilter(); f.type = type; f.Q.value = q;
    f.frequency.setValueAtTime(from, t); f.frequency.exponentialRampToValueAtTime(Math.max(40, to), t + dur);
    const g = c.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + Math.min(0.02, dur / 3));
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f).connect(g).connect(this.master!);
    src.start(t, 0, Math.min(dur + 0.05, 0.55));
  }

  /** The held-button tone: climbs with the charge, silent at 0. Call every frame. */
  setCharge(level: number): void {
    if (!this.ready()) { if (this.charge && !this.enabled) this.stopCharge(); return; }
    const c = this.ctx!;
    if (level <= 0) { this.stopCharge(); return; }
    if (!this.charge) {
      const osc = c.createOscillator(), gain = c.createGain();
      osc.type = "triangle"; gain.gain.value = 0;
      osc.connect(gain).connect(this.master!); osc.start();
      this.charge = { osc, gain };
    }
    const t = c.currentTime;
    this.charge.osc.frequency.setTargetAtTime(150 + 520 * level * level, t, 0.03);
    this.charge.gain.gain.setTargetAtTime(0.05 + 0.1 * level, t, 0.03);
  }

  private stopCharge(): void {
    if (!this.charge || !this.ctx) { this.charge = null; return; }
    const { osc, gain } = this.charge, t = this.ctx.currentTime;
    gain.gain.setTargetAtTime(0, t, 0.02);
    osc.stop(t + 0.15);
    this.charge = null;
  }

  play(events: SoundEvent[]): void {
    if (!this.ready()) return;
    const now = this.ctx!.currentTime;
    for (const e of events) {
      switch (e.kind) {
        case "tick":
          // a small tick for spikes; throttled so a burst of hundreds is a rattle, not a wall
          if (now - this.lastTick > 0.035) {
            this.lastTick = now;
            this.tone(1500 + 900 * Math.min(1, e.n / 30), 0.03, "square", 0.025 + 0.05 * Math.min(1, e.n / 30));
          }
          break;
        case "zap": this.tone(1800, 0.16, "sawtooth", 0.1, 380); break;
        case "strike": this.burst(0.14, 300, 2600, 1.2, 0.5); break;
        case "takeoff":
          this.burst(0.12, 2400, 900, 2, 0.22);
          for (let i = 0; i < 6; i++) this.tone(190 + i * 4, 0.14, "sawtooth", 0.035 * (1 - i / 7), 175, i * 0.12);
          break;
        case "result":
          if (e.outcome === "hit") { this.tone(130, 0.18, "sine", 0.55, 48); this.burst(0.2, 900, 120, 0.8, 0.45, "lowpass"); }
          else if (e.outcome === "miss") this.tone(95, 0.12, "sine", 0.35, 55);
          else if (e.outcome === "escaped") this.tone(95, 0.12, "sine", 0.3, 55);
          break;
      }
    }
  }
}
