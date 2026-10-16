// The escape circuit, drawn from the engine's own graph: 495 neurons and their strongest connections, with the
// voltage and spikes the engine produced this round. Layout is a diagram (three columns, rows ordered by wiring),
// NOT anatomy: the positions mean nothing, the connections and the activity are real.

import type { FlyBrain, Manifest } from "../engine/brain.ts";
import type { Round } from "./world.ts";
import { GROWTH_EPS } from "./story.ts";

export type Kind = "lplc2" | "relay" | "gf" | "parallel";

export interface Circuit {
  n: number;
  kind: Kind[];
  /** 0..1 position inside the drawing. */
  x: Float32Array;
  y: Float32Array;
  edges: Array<{ a: number; b: number; w: number }>;
  driven: number;
  counts: Record<Kind, number>;
  label: (j: number) => string;
}

const ELEMENT_BYTES: Record<string, number> = { "<i4": 4, "<u2": 2, "<f4": 4, "|u1": 1 };

function read(manifest: Manifest, bin: ArrayBuffer, name: string): Int32Array | Uint16Array | Float32Array {
  const s = manifest.arrays[name], len = s.shape.reduce((a, b) => a * b, 1);
  const bytes = bin.slice(s.offset, s.offset + len * ELEMENT_BYTES[s.dtype]);
  return s.dtype === "<i4" ? new Int32Array(bytes) : s.dtype === "<u2" ? new Uint16Array(bytes) : new Float32Array(bytes);
}

export function buildCircuit(manifest: Manifest, bin: ArrayBuffer, drive: FlyBrain["drive"]): Circuit {
  const n = manifest.neurons;
  const types = manifest.neuron_type ?? [];
  const indptr = read(manifest, bin, "csr_indptr") as Int32Array;
  const post = read(manifest, bin, "csr_post") as Int32Array;
  const weight = read(manifest, bin, "csr_weight_mv") as Float32Array;
  const gf = new Set(manifest.roles.gf), par = new Set(manifest.roles.parallel);
  const kind: Kind[] = Array.from({ length: n }, (_, j) => (gf.has(j) ? "gf" : par.has(j) ? "parallel" : types[j] === "LPLC2" ? "lplc2" : "relay"));
  const drivenSet = new Set(drive.neuron);

  // which side each relay pushes toward: summed weight into right-side minus left-side escape neurons
  const sideOf = new Map<number, number>();
  [...manifest.roles.gf, ...manifest.roles.parallel].forEach((j, i) => sideOf.set(j, manifest.roles.target_side[i] === "R" ? 1 : -1));
  const lean = new Float32Array(n);
  const edges: Circuit["edges"] = [];
  for (let a = 0; a < n; a++) {
    for (let e = indptr[a]; e < indptr[a + 1]; e++) {
      const b = post[e], w = weight[e];
      const sb = sideOf.get(b);
      if (sb !== undefined) lean[a] += sb * w;
      const ka = kind[a], kb = kind[b];
      if ((ka === "lplc2" && kb === "relay") || (ka === "relay" && (kb === "gf" || kb === "parallel")) || (ka === "lplc2" && (kb === "gf" || kb === "parallel"))) edges.push({ a, b, w });
    }
  }
  edges.sort((p, q) => Math.abs(q.w) - Math.abs(p.w));
  edges.length = Math.min(edges.length, 620);

  const x = new Float32Array(n), y = new Float32Array(n);
  // deterministic scatter in a column so dots do not form a ruler
  const jitter = (j: number) => ((Math.imul(j + 1, 2654435761) >>> 0) % 1000) / 1000;
  const column = (ids: number[], cx: number, spread: number, key: (j: number) => number) => {
    ids.sort((p, q) => key(p) - key(q));
    ids.forEach((j, i) => { x[j] = cx + (jitter(j) - 0.5) * spread; y[j] = 0.09 + (0.82 * (i + 0.5)) / ids.length; });
  };
  const ids = (k: Kind) => Array.from({ length: n }, (_, j) => j).filter((j) => kind[j] === k);
  column(ids("lplc2"), 0.11, 0.1, (j) => (drivenSet.has(j) ? 0 : 1e6) + lean[j] * 0 + j);
  column(ids("relay"), 0.45, 0.2, (j) => -lean[j]); // right-leaning relays at the top, left-leaning at the bottom
  const dns = [...manifest.roles.gf, ...manifest.roles.parallel];
  dns.forEach((j, i) => { x[j] = 0.76; y[j] = 0.1 + (0.8 * i) / (dns.length - 1); });
  const label = (j: number) => {
    const i = dns.indexOf(j);
    return i < 0 ? "" : `${types[j] ?? ""} ${manifest.roles.target_side[i]}`;
  };
  const counts: Record<Kind, number> = { lplc2: 0, relay: 0, gf: 0, parallel: 0 };
  for (const k of kind) counts[k]++;
  return { n, kind, x, y, edges, driven: drivenSet.size, counts, label };
}

// ----------------------------------------------------------------------------------------------- activity

export interface ActivitySummary {
  total: number;
  active: Record<Kind, number>;
  first: Record<Kind, number | null>;
}

/** Per-round bookkeeping the panels share. Call `tick` after every `Round.tick`. */
export class Activity {
  readonly growthStarts: number[] = [];
  first: Record<Kind, number | null> = { lplc2: null, relay: null, gf: null, parallel: null };
  private growing = false;
  private seen = -1;
  private circuit: Circuit;
  constructor(circuit: Circuit) { this.circuit = circuit; }

  tick(round: Round): void {
    const g = round.lastRate >= GROWTH_EPS;
    if (g && !this.growing) this.growthStarts.push(round.timeMs);
    this.growing = g;
    const eng = round.engine;
    if (eng.spikeCount !== this.seen) {
      this.seen = eng.spikeCount;
      const f: Record<Kind, number | null> = { lplc2: null, relay: null, gf: null, parallel: null };
      for (const [s, j] of eng.spikes()) {
        const k = this.circuit.kind[j], ms = s * eng.dtMs;
        if (f[k] === null || ms < f[k]!) f[k] = ms;
      }
      this.first = f;
    }
  }

  summary(round: Round, windowMs = 20): ActivitySummary {
    const active: Record<Kind, number> = { lplc2: 0, relay: 0, gf: 0, parallel: 0 };
    for (let j = 0; j < this.circuit.n; j++) if (round.timeMs - round.lastSpikeMs[j] < windowMs) active[this.circuit.kind[j]]++;
    return { total: round.engine.spikeCount, active, first: this.first };
  }
}

// -------------------------------------------------------------------------------------------------- draw

const rgb = (hex: string, a: number) => {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${(v >> 16) & 255},${(v >> 8) & 255},${v & 255},${a})`;
};

/** How long, in simulated ms, a spike keeps glowing. The replay runs at 0.1x, so this is about a second of wall time there. */
const GLOW_MS = 60;
const EDGE_MS = 45;

export function drawCircuit(canvas: HTMLCanvasElement, round: Round, c: Circuit, afterMs = 0): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (cw === 0 || ch === 0) return;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
  const g = canvas.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.globalCompositeOperation = "source-over";
  const bg = g.createRadialGradient(cw * 0.5, ch * 0.55, 10, cw * 0.5, ch * 0.55, Math.max(cw, ch) * 0.75);
  bg.addColorStop(0, "#0c1a1f"); bg.addColorStop(1, "#04080a");
  g.fillStyle = bg;
  g.fillRect(0, 0, cw, ch);

  const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const cyan = css("--accent"), hot = css("--bad"), amber = css("--warn");
  const eng = round.engine, lif = eng.manifest.lif;
  const now = round.timeMs + afterMs;
  const s = Math.max(0.85, Math.min(1.7, cw / 520)); // dot and line sizes follow the panel
  const PX = 30, TOP = 74, BOT = 16;
  const px = (j: number) => PX + c.x[j] * (cw - 2 * PX), py = (j: number) => TOP + c.y[j] * (ch - TOP - BOT);
  const delayMs = lif["n_delay"] * lif["dt_ms"];
  const wmax = Math.max(1e-6, Math.abs(c.edges[0]?.w ?? 1));

  // connections: always visible, strongest brightest; bright and travelling while the sending neuron has just fired
  g.lineCap = "round";
  for (const e of c.edges) {
    const ax = px(e.a), ay = py(e.a), bx = px(e.b), by = py(e.b);
    const strength = Math.pow(Math.abs(e.w) / wmax, 0.6);
    g.strokeStyle = rgb(e.w >= 0 ? cyan : amber, 0.1 + 0.3 * strength);
    g.lineWidth = (0.7 + 0.5 * strength) * s;
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
  }
  g.globalCompositeOperation = "lighter";
  for (const e of c.edges) {
    const age = now - round.lastSpikeMs[e.a];
    if (!(age >= 0 && age < EDGE_MS)) continue;
    const ax = px(e.a), ay = py(e.a), bx = px(e.b), by = py(e.b);
    const fade = 1 - age / EDGE_MS;
    g.strokeStyle = rgb(e.w >= 0 ? cyan : amber, 0.9 * fade);
    g.lineWidth = 1.6 * s;
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
    const p = Math.min(1, age / Math.max(delayMs, 0.5));
    g.fillStyle = rgb("#ffffff", fade);
    g.beginPath(); g.arc(ax + (bx - ax) * p, ay + (by - ay) * p, 2.6 * s, 0, 7); g.fill();
  }

  // neurons: brightness follows membrane voltage; neurons that fired this round stay warm; a spike blooms
  const level = (j: number) => Math.max(0, Math.min(1, (eng.voltage(j) - lif["v_rst"]) / (lif["v_th"] - lif["v_rst"])));
  for (let j = 0; j < c.n; j++) {
    const k = c.kind[j], big = k === "gf" || k === "parallel";
    const age = now - round.lastSpikeMs[j];
    const r = (big ? 6.5 : k === "lplc2" ? 2.4 : 2.2) * s;
    const lv = level(j);
    const base = k === "lplc2" ? amber : cyan;
    if (age >= 0 && age < GLOW_MS) {
      const a = 1 - age / GLOW_MS;
      g.fillStyle = rgb(hot, 0.16 * a + 0.04);
      g.beginPath(); g.arc(px(j), py(j), r * (big ? 5.5 : 6) * (0.7 + 0.5 * (1 - a)), 0, 7); g.fill();
      g.fillStyle = rgb(hot, 0.55 * a);
      g.beginPath(); g.arc(px(j), py(j), r * 2.6, 0, 7); g.fill();
    } else if (age >= GLOW_MS && age < Infinity) {
      g.fillStyle = rgb(hot, 0.28); // fired earlier this round
      g.beginPath(); g.arc(px(j), py(j), r * 1.9, 0, 7); g.fill();
    }
    g.fillStyle = age >= 0 && age < 14 ? "#ffffff" : rgb(base, 0.55 + 0.45 * lv);
    g.beginPath(); g.arc(px(j), py(j), r * (0.9 + 0.35 * lv), 0, 7); g.fill();
  }
  g.globalCompositeOperation = "source-over";

  // labels
  const f = (px: number, w = 400) => `${w} ${px}px ui-sans-serif, system-ui, sans-serif`;
  const col = (cx: number) => PX + cx * (cw - 2 * PX);
  const head = (cx: number, title: string, sub: string, color: string) => {
    g.textAlign = "center";
    g.fillStyle = color; g.font = f(11, 700); g.fillText(title.toUpperCase(), col(cx), 46);
    g.fillStyle = "#8ea3a6"; g.font = f(11.5); g.fillText(sub, col(cx), 62);
  };
  head(0.11, "Looming detectors", `${c.counts.lplc2} LPLC2 neurons`, amber);
  head(0.45, "Relay neurons", `${c.counts.relay} neurons in between`, cyan);
  head(0.76, "Escape neurons", `${c.counts.gf + c.counts.parallel} descending neurons`, hot);
  g.textAlign = "center"; g.fillStyle = "#5f7478"; g.font = f(10.5);
  g.fillText("\u2191 pushes toward the right", col(0.45), TOP + 2);
  g.fillText("\u2193 pushes toward the left", col(0.45), ch - 3);
  // the signal runs left to right
  g.strokeStyle = "#35565b"; g.fillStyle = "#35565b"; g.lineWidth = 1;
  for (const [a, b] of [[0.2, 0.36], [0.55, 0.69]] as const) {
    const y = 24;
    g.beginPath(); g.moveTo(col(a), y); g.lineTo(col(b), y); g.stroke();
    g.beginPath(); g.moveTo(col(b), y); g.lineTo(col(b) - 5, y - 3.5); g.lineTo(col(b) - 5, y + 3.5); g.fill();
  }
  g.textAlign = "left";
  for (const j of [...eng.manifest.roles.gf, ...eng.manifest.roles.parallel]) {
    const age = now - round.lastSpikeMs[j];
    g.fillStyle = age >= 0 && age < GLOW_MS ? "#ffffff" : "#9fb4b7";
    g.font = f(12, age >= 0 && age < GLOW_MS ? 700 : 500);
    g.fillText(c.label(j), px(j) + 13 * s, py(j) + 4);
  }
}
