// Glue between the game (or the leaderboard's Node re-sim) and the Rust/WASM engine.
// Same calls as offline/wasm_host.py. Written to run unchanged in a browser and in Node:
// it takes bytes, not file paths. Uses only erasable TypeScript syntax, so Node can run it directly.

export interface ArraySpec { dtype: string; shape: number[]; offset: number }
export interface Manifest {
  binary: string;
  neurons: number;
  edges: number;
  arrays: Record<string, ArraySpec>;
  lif: Record<string, number>;
  drive: { rf_sigma_deg: number };
  roles: { gf: number[]; parallel: number[]; target_side: string[]; short_window_ms: number };
}

/** One 5 ms frame: swatter growth rate (deg/ms) and its position on each eye's plane (deg). */
export type Frame = readonly [rate: number, xR: number, yR: number, xL: number, yL: number];

export interface Outcome {
  gfFirstMs: number | null;
  parallelFirstMs: number | null;
  escapeFirstMs: number | null;
  /** (R - L) / (R + L) over escape-DN spikes in the 20 ms after the first one. */
  A: number;
  /** Heading is away from the stronger side; null if there is none. */
  headingSide: "left" | "right" | null;
  mode: "short" | "long" | "none";
}

interface Exports {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  init(...args: number[]): void;
  reset(): void;
  frame(rate: number, xr: number, yr: number, xl: number, yl: number): void;
  frame_n(rate: number, xr: number, yr: number, xl: number, yl: number, substeps: number): void;
  finish(): void;
  voltage(j: number): number;
  step_count(): number;
  spike_count(): number;
  spikes_ptr(): number;
}

const ELEMENT_BYTES: Record<string, number> = { "<i4": 4, "<u2": 2, "<f4": 4, "|u1": 1 };
const ARRAYS = ["csr_indptr", "csr_post", "csr_weight_mv", "drive_neuron", "drive_rf_deg", "drive_eye"] as const;

export class FlyBrain {
  readonly dtMs: number;
  readonly manifest: Manifest;
  private x: Exports;
  // explicit fields, not constructor parameter properties: those are not erasable syntax
  private constructor(x: Exports, manifest: Manifest) {
    this.x = x;
    this.manifest = manifest;
    this.dtMs = manifest.lif["dt_ms"];
  }

  static async create(manifest: Manifest, brainBin: ArrayBuffer, wasm: BufferSource): Promise<FlyBrain> {
    const { instance } = await WebAssembly.instantiate(wasm, {});
    const x = instance.exports as unknown as Exports;
    const put = (bytes: Uint8Array): number => {
      const ptr = x.alloc(bytes.length);
      new Uint8Array(x.memory.buffer).set(bytes, ptr); // re-read the buffer: alloc may have grown memory
      return ptr;
    };
    const ptr: Record<string, number> = {};
    for (const name of ARRAYS) {
      const s = manifest.arrays[name];
      const bytes = s.shape.reduce((a, b) => a * b, 1) * ELEMENT_BYTES[s.dtype];
      ptr[name] = put(new Uint8Array(brainBin, s.offset, bytes));
    }
    const p = manifest.lif;
    const params = new Float64Array([p["v0"], p["v_rst"], p["v_th"], p["t_mbr"], p["tau"], p["dt_ms"], p["n_delay"],
      p["n_refractory"], p["input_gain"], manifest.drive.rf_sigma_deg, 0, 0]);
    const pParams = put(new Uint8Array(params.buffer));
    x.init(manifest.neurons, manifest.edges, manifest.arrays["drive_neuron"].shape[0], ptr["csr_indptr"], ptr["csr_post"],
      ptr["csr_weight_mv"], ptr["drive_neuron"], ptr["drive_rf_deg"], ptr["drive_eye"], pParams);
    return new FlyBrain(x, manifest);
  }

  reset(): void { this.x.reset(); }
  /** Feed the next 5 ms frame. The engine is one frame behind: the first call only latches it. */
  frame(f: Frame): void { this.x.frame(f[0], f[1], f[2], f[3], f[4]); }
  /** As `frame`, for an interval of `substeps` 0.2 ms steps (1 to 25), e.g. the last frame of a strike. */
  frameFor(f: Frame, substeps: number): void { this.x.frame_n(f[0], f[1], f[2], f[3], f[4], substeps); }
  finish(): void { this.x.finish(); }
  /** Membrane potential of neuron j, mV. */
  voltage(j: number): number { return this.x.voltage(j); }
  get steps(): number { return this.x.step_count(); }
  get spikeCount(): number { return this.x.spike_count(); }

  /** All spikes so far as [step, neuron] pairs. */
  spikes(): Array<[number, number]> {
    const n = this.x.spike_count();
    if (n === 0) return [];
    const a = new Uint32Array(this.x.memory.buffer, this.x.spikes_ptr(), 2 * n);
    const out: Array<[number, number]> = [];
    for (let i = 0; i < n; i++) out.push([a[2 * i], a[2 * i + 1]]);
    return out;
  }

  /** Whole trial: reset, every frame, finish. */
  run(frames: readonly Frame[]): Array<[number, number]> {
    this.reset();
    for (const f of frames) this.frame(f);
    this.finish();
    return this.spikes();
  }

  /** Mode and heading from the spikes so far, by the plan's fixed rules (offline/takeoff.py, heading.py). */
  outcome(): Outcome {
    const r = this.manifest.roles;
    const gf = new Set(r.gf), par = new Set(r.parallel);
    const side = new Map<number, string>();
    [...r.gf, ...r.parallel].forEach((j, i) => side.set(j, r.target_side[i]));
    const sp = this.spikes();
    const first = (ids: Set<number>): number | null => {
      let t: number | null = null;
      for (const [s, j] of sp) if (ids.has(j) && (t === null || s < t)) t = s;
      return t === null ? null : t * this.dtMs;
    };
    const tGf = first(gf), tPar = first(par);
    const escapes = sp.filter(([, j]) => side.has(j));
    const out: Outcome = { gfFirstMs: tGf, parallelFirstMs: tPar, escapeFirstMs: null, A: 0, headingSide: null, mode: "none" };
    if (escapes.length) {
      const t0 = Math.min(...escapes.map(([s]) => s));
      out.escapeFirstMs = t0 * this.dtMs;
      const win = escapes.filter(([s]) => s >= t0 && s < t0 + Math.round(20 / this.dtMs));
      const right = win.filter(([, j]) => side.get(j) === "R").length;
      const left = win.filter(([, j]) => side.get(j) === "L").length;
      out.A = (right - left) / Math.max(1, right + left);
      out.headingSide = right > left ? "left" : left > right ? "right" : null;
    }
    const w = r.short_window_ms;
    out.mode = tGf === null && tPar === null ? "none" : tPar === null ? "short" : tGf === null ? "long" : tGf <= tPar + w ? "short" : "long";
    return out;
  }
}
