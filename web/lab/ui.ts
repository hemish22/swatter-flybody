// Lab mode UI: fire standard looms at a tethered fly and watch criteria 1 and 2 build up from the browser's own engine.
// The stimuli and the engine run are `loom.ts` (pinned to the offline sweep); this file only draws and keeps tallies.
import type { FlyBrain } from "../engine/brain.ts";
import { LAB, LAB_KINDS, runLab, type LabKind, type LabResult } from "./loom.ts";

const AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315];
const AZ_LABEL: Record<number, string> = { 0: "ahead", 45: "right-front", 90: "right", 135: "right-back", 180: "behind", 225: "left-back", 270: "left", 315: "left-front" };
const RVS = [10, 20, 40, 80];
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export interface LabEls { arena: HTMLCanvasElement; panel: HTMLElement }

export class Lab {
  private brain: FlyBrain;
  private els: LabEls;
  readonly results: LabResult[] = [];
  private playing: { r: LabResult; startMs: number } | null = null;
  private queue: Array<{ kind: LabKind; rvMs: number; azimuthDeg: number }> = [];

  constructor(brain: FlyBrain, els: LabEls) {
    this.brain = brain;
    this.els = els;
    this.build();
    this.drawPlot();
  }

  private build(): void {
    const kinds = LAB_KINDS.map((k) => `<option>${k}</option>`).join("");
    const azs = AZIMUTHS.map((a) => `<option value="${a}"${a === 90 ? " selected" : ""}>${a}° ${AZ_LABEL[a]}</option>`).join("");
    this.els.panel.innerHTML = `<h2>Lab</h2>
      <p class="note" style="margin-top:0">A tethered fly. You choose a standard stimulus (the lab's disc of half-size r approaching at speed v, set by r/v) and where it comes from. The browser runs the same engine and the same stimuli as the offline sweep.</p>
      <table>
        <tr><td><label for="lab-kind">Stimulus</label></td><td><select id="lab-kind">${kinds}</select></td></tr>
        <tr><td><label for="lab-rv">r/v</label></td><td><input id="lab-rv" type="range" min="10" max="80" step="1" value="20"> <output id="lab-rv-out">20 ms</output></td></tr>
        <tr><td><label for="lab-az">From</label></td><td><select id="lab-az">${azs}</select></td></tr>
      </table>
      <p><button id="lab-fire">Fire</button> <button id="lab-sweep">Run the standard sweep</button> <button id="lab-clear">Clear</button></p>
      <div id="lab-last" class="note">No stimulus fired yet.</div>
      <canvas id="lab-plot" width="300" height="210" aria-label="Angle at the first giant-fiber spike against r/v"></canvas>
      <div id="lab-tally"></div>
      <div class="note">Sweep: ${RVS.join(", ")} ms × 8 directions × 4 stimulus kinds = ${RVS.length * 8 * LAB_KINDS.length} trials. Criterion 1: the giant fiber spikes for the expanding disc and stays silent for the three controls. Ahead and behind are this model's weak directions (the eyes are assumed to face sideways), and at r/v 10 the disc has not grown enough by the end of the 400 ms for the giant fiber to spike there, so expect 2 of 8 directions to be silent at that one speed. Criterion 2: the angle it spikes at (here the plot) is not constant. Fired looms run at the fixed, offline-chosen parameters; nothing is tuned here.</div>`;
    const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
    const rv = $<HTMLInputElement>("lab-rv");
    rv.addEventListener("input", () => { $("lab-rv-out").textContent = `${rv.value} ms`; });
    $("lab-fire").addEventListener("click", () => this.fire({
      kind: $<HTMLSelectElement>("lab-kind").value as LabKind, rvMs: Number(rv.value), azimuthDeg: Number($<HTMLSelectElement>("lab-az").value) }));
    $("lab-sweep").addEventListener("click", () => {
      this.queue = [];
      for (const kind of LAB_KINDS) for (const rvMs of RVS) for (const azimuthDeg of AZIMUTHS) this.queue.push({ kind, rvMs, azimuthDeg });
    });
    $("lab-clear").addEventListener("click", () => { this.results.length = 0; this.queue = []; this.playing = null; this.drawPlot(); this.tally(); $("lab-last").textContent = "No stimulus fired yet."; });
    this.tally();
  }

  /** For screenshots: run the standard sweep to completion, then replay one expanding loom. */
  demo(): void {
    for (const kind of LAB_KINDS) for (const rvMs of RVS) for (const azimuthDeg of AZIMUTHS) this.fire({ kind, rvMs, azimuthDeg });
    this.fire({ kind: "expanding", rvMs: 20, azimuthDeg: 90 });
    this.playing = { r: this.results[this.results.length - 1], startMs: performance.now() - 120 };
  }

  private fire(s: { kind: LabKind; rvMs: number; azimuthDeg: number }): LabResult {
    const r = runLab(this.brain, s);
    this.results.push(r);
    this.playing = { r, startMs: performance.now() };
    const last = document.getElementById("lab-last")!;
    last.textContent = `${s.kind}, r/v ${s.rvMs} ms from ${s.azimuthDeg}°: ` + (r.gfFirstMs === null
      ? "the giant fiber did not spike."
      : `giant fiber spiked at ${r.gfFirstMs.toFixed(1)} ms, swatter ${r.thetaAtGfDeg!.toFixed(1)}° wide, ${r.mode}-mode takeoff` + (r.headingSide ? `, turned ${r.headingSide}.` : "."));
    this.drawPlot();
    this.tally();
    return r;
  }

  private tally(): void {
    const el = document.getElementById("lab-tally")!;
    const rows = LAB_KINDS.map((k) => {
      const t = this.results.filter((r) => r.stimulus.kind === k);
      return `<tr><td>${k}</td><td>${t.filter((r) => r.gfFirstMs !== null).length} of ${t.length} GF spiked</td></tr>`;
    });
    el.innerHTML = `<table>${rows.join("")}</table>`;
  }

  /** Angle at the first GF spike against r/v, every expanding trial that spiked, with the median at each r/v. */
  private drawPlot(): void {
    const c = document.getElementById("lab-plot") as HTMLCanvasElement, g = c.getContext("2d")!, w = c.width, h = c.height, pad = { l: 34, r: 8, t: 8, b: 26 };
    g.clearRect(0, 0, w, h);
    const x = (rv: number) => pad.l + ((rv - 5) / 80) * (w - pad.l - pad.r);
    const y = (a: number) => h - pad.b - (a / 30) * (h - pad.t - pad.b);
    g.font = "10px system-ui"; g.fillStyle = css("--muted"); g.strokeStyle = css("--muted"); g.lineWidth = 1;
    g.textAlign = "right";
    for (const a of [0, 10, 20, 30]) { g.fillText(`${a}°`, pad.l - 4, y(a) + 3); g.globalAlpha = 0.2; g.beginPath(); g.moveTo(pad.l, y(a)); g.lineTo(w - pad.r, y(a)); g.stroke(); g.globalAlpha = 1; }
    g.textAlign = "center";
    for (const rv of RVS) g.fillText(String(rv), x(rv), h - pad.b + 12);
    g.fillText("r/v (ms)", (pad.l + w - pad.r) / 2, h - 3);
    const pts = this.results.filter((r) => r.stimulus.kind === "expanding" && r.thetaAtGfDeg !== null);
    g.fillStyle = css("--accent"); g.globalAlpha = 0.55;
    for (const r of pts) { g.beginPath(); g.arc(x(r.stimulus.rvMs), y(Math.min(r.thetaAtGfDeg!, 30)), 3, 0, 7); g.fill(); }
    g.globalAlpha = 1;
    const med: Array<[number, number]> = [];
    for (const rv of RVS) {
      const a = pts.filter((r) => r.stimulus.rvMs === rv).map((r) => r.thetaAtGfDeg!).sort((p, q) => p - q);
      if (a.length) med.push([rv, a[a.length >> 1]]);
    }
    g.strokeStyle = css("--warn"); g.lineWidth = 2; g.beginPath();
    med.forEach(([rv, a], i) => (i ? g.lineTo(x(rv), y(a)) : g.moveTo(x(rv), y(a))));
    g.stroke();
    g.fillStyle = css("--muted"); g.textAlign = "left";
    g.fillText(pts.length ? "angle at first GF spike (line: median)" : "fire an expanding disc, or run the sweep", pad.l + 6, pad.t + 10);
  }

  /** Call every animation frame while Lab is the active mode. */
  frame(now: number): void {
    if (this.queue.length) { this.fire(this.queue.shift()!); this.playing = null; }
    this.drawArena(now);
  }

  private drawArena(now: number): void {
    const c = this.els.arena, g = c.getContext("2d")!, w = c.width, h = c.height;
    g.fillStyle = css("--panel") === "#fff" ? "#d9d6cd" : "#3a3d44";
    g.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.42;
    g.strokeStyle = "rgba(128,128,128,.25)"; g.lineWidth = 1;
    for (const f of [0.33, 0.66, 1]) { g.beginPath(); g.arc(cx, cy, R * f, 0, 7); g.stroke(); }
    g.fillStyle = css("--muted"); g.font = "12px system-ui"; g.textAlign = "center";
    g.fillText("ahead", cx, cy - R - 8); g.fillText("right", cx + R + 24, cy + 4); g.fillText("left", cx - R - 24, cy + 4); g.fillText("behind", cx, cy + R + 16);
    const p = this.playing;
    if (p) {
      const t = Math.min(now - p.startMs, LAB.durationMs);
      const i = Math.min(Math.floor(t / LAB.frameMs), p.r.thetaDeg.length - 1);
      const theta = p.r.thetaDeg[i];
      const az = (p.r.stimulus.azimuthDeg * Math.PI) / 180, half = (Math.min(theta, 179) * Math.PI) / 360;
      // screen angle: ahead is up, right is +x
      const a0 = -Math.PI / 2 + az - half, a1 = -Math.PI / 2 + az + half;
      const gone = p.r.gfFirstMs !== null && t >= p.r.gfFirstMs;
      g.fillStyle = p.r.stimulus.kind === "dimming" ? `rgba(20,22,26,${Math.max(0.05, 0.78 * (1 - t / LAB.durationMs))})` : "rgba(20,22,26,.78)";
      g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, a0, a1); g.closePath(); g.fill();
      if (gone) { g.strokeStyle = css("--warn"); g.lineWidth = 3; g.beginPath(); g.arc(cx, cy, 30, 0, 7); g.stroke(); }
      g.fillStyle = css("--ink"); g.textAlign = "left"; g.font = "14px system-ui";
      g.fillText(`t = ${t.toFixed(0)} ms   swatter ${theta.toFixed(1)}° wide${gone ? "   giant fiber spiked" : ""}`, 14, h - 14);
      if (t >= LAB.durationMs && !gone) { g.fillStyle = css("--muted"); g.fillText("no giant-fiber spike", 14, h - 34); }
    }
    // the fly, forward is up
    g.save(); g.translate(cx, cy); g.fillStyle = "#15161a";
    g.beginPath(); g.ellipse(0, 0, 9, 18, 0, 0, 7); g.fill();
    g.fillStyle = "#5aa9ff"; g.beginPath(); g.arc(0, -17, 6, 0, 7); g.fill(); g.restore();
  }
}
