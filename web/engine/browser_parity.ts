// In-browser parity run: the 50 parity stimuli through the WASM engine in a real browser, results written to
// the page so a headless browser (or a person) can read them. Entry point for web/parity.html.
import { FlyBrain, type Frame, type Manifest } from "./brain.ts";

interface Trial { id: number; frames: Frame[]; expected: { gf_first_ms: number | null; parallel_first_ms: number | null; escape_first_ms: number | null; heading_side: string | null; mode: string } }

const out = document.getElementById("result")!;
try {
  const [manifest, parity, wasm] = await Promise.all([
    fetch("brain/manifest.json").then((r) => r.json() as Promise<Manifest>),
    fetch("brain/parity.json").then((r) => r.json() as Promise<{ n: number; dt_ms: number; trials: Trial[] }>),
    fetch("brain/engine.wasm").then((r) => r.arrayBuffer()),
  ]);
  const bin = await fetch("brain/" + manifest.binary).then((r) => r.arrayBuffer());
  const brain = await FlyBrain.create(manifest, bin, wasm);
  const near = (a: number | null, b: number | null, tol: number) => (a === null) === (b === null) && (a === null || Math.abs(a - (b as number)) <= tol + 1e-9);
  let bad = 0;
  const t0 = performance.now();
  for (const t of parity.trials) {
    brain.run(t.frames);
    const g = brain.outcome(), e = t.expected;
    const ok = g.mode === e.mode && g.headingSide === e.heading_side && near(g.gfFirstMs, e.gf_first_ms, parity.dt_ms) &&
      near(g.parallelFirstMs, e.parallel_first_ms, parity.dt_ms) && near(g.escapeFirstMs, e.escape_first_ms, parity.dt_ms);
    if (!ok) bad++;
  }
  const ms = (performance.now() - t0) / parity.trials.length;
  out.textContent = JSON.stringify({ ok: bad === 0, trials: parity.trials.length, mismatches: bad, msPer400msTrial: +ms.toFixed(2), userAgent: navigator.userAgent });
} catch (err) {
  out.textContent = JSON.stringify({ ok: false, error: String(err) });
}
