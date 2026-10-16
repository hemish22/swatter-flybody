// The plan's population panel: what each eye gets, which LPLC2 neurons it drives, and where the escape neurons are.
// Honest labelling matters here: the eyes show the swatter's footprint on a hexagonal lattice, computed from geometry.
// The optic lobe is not run (docs/week1_gate.md), so nothing in the eye discs is a neural response; the LPLC2 rings and
// the descending neurons are the engine's own state.
import { eyeView, type Round } from "./world.ts";

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const ECC_MAX = 120; // degrees of eccentricity drawn from each eye's axis

/** Hex lattice points inside a disc of `radius` px with spacing `step`, as [x, y] offsets. */
export function hexLattice(radius: number, step: number): Array<[number, number]> {
  const out: Array<[number, number]> = [], dy = step * Math.sqrt(3) / 2;
  for (let row = -Math.floor(radius / dy); row <= radius / dy; row++) {
    const y = row * dy, shift = (row & 1) * step / 2;
    for (let x = -radius - step; x <= radius + step; x += step) { const px = x + shift; if (Math.hypot(px, y) <= radius) out.push([px, y]); }
  }
  return out;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * Math.max(0, Math.min(1, t));

/** Logical drawing size; the canvas is scaled to its CSS width, so the same picture stays sharp at any size. */
const LW = 360, LH = 330;

export function drawPopulation(canvas: HTMLCanvasElement, round: Round): void {
  const cssW = canvas.clientWidth;
  if (cssW === 0) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1), pw = Math.round(cssW * dpr), ph = Math.round((pw * LH) / LW);
  if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
  const g = canvas.getContext("2d")!;
  g.setTransform(pw / LW, 0, 0, pw / LW, 0, 0);
  g.clearRect(0, 0, LW, LH);
  const W = LW;
  const brain = round.engine, m = brain.manifest, lif = m.lif;
  const view = eyeView(round.fly, round.swatter);
  const nowMs = round.timeMs;
  const R = 76, step = 8, cy = 84, cx = [W / 4 + 2, (3 * W) / 4 - 2];
  const lattice = hexLattice(R, step);
  const px = (deg: number) => (deg / ECC_MAX) * R; // plane degrees to pixels
  const ink = "#e5eeee", muted = "#8ea3a6", pink = "#ff4fa3", cyan = "#4fe3d1", amber = "#f0a95b";
  g.textAlign = "center";

  // --- two eyes: the swatter's footprint on the lattice (pink = covered), LPLC2 receptive fields on top
  const pos = [view.right, view.left], names = ["Right eye", "Left eye"];
  for (let e = 0; e < 2; e++) {
    const [ox, oy] = [cx[e], cy];
    const [sx, sy] = pos[e];
    const half = view.thetaDeg / 2;
    for (const [lx, ly] of lattice) {
      const dx = (lx / R) * ECC_MAX - sx, dy = (-ly / R) * ECC_MAX - sy; // plane y is up, screen y down
      const covered = Math.hypot(dx, dy) <= half;
      g.fillStyle = covered ? "#ff7fbf" : "#2d4a50";
      g.beginPath(); g.arc(ox + lx, oy + ly, step * (covered ? 0.46 : 0.38), 0, 7); g.fill();
    }
    g.strokeStyle = "#5b8089"; g.lineWidth = 1.2;
    g.beginPath(); g.arc(ox, oy, R, 0, 7); g.stroke();
    // LPLC2 neurons of this eye: ring at the receptive-field centre, filled by membrane potential
    const d = brain.drive;
    for (let i = 0; i < d.neuron.length; i++) {
      if (d.eye[i] !== e) continue;
      const rx = d.rf[2 * i], ry = d.rf[2 * i + 1];
      if (Math.hypot(rx, ry) > ECC_MAX) continue;
      const v = brain.voltage(d.neuron[i]);
      const level = (v - lif["v_rst"]) / (lif["v_th"] - lif["v_rst"]);
      const fired = nowMs - round.lastSpikeMs[d.neuron[i]] < 15;
      const x = ox + px(rx), y = oy - px(ry);
      if (fired) { g.fillStyle = "rgba(255,79,163,.45)"; g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); }
      g.fillStyle = fired ? "#ffffff" : `rgba(240,169,91,${lerp(0.05, 1, level)})`;
      g.strokeStyle = fired ? pink : amber; g.lineWidth = 1.4;
      g.beginPath(); g.arc(x, y, 3.3, 0, 7); g.fill(); g.stroke();
    }
    g.fillStyle = ink; g.font = "600 13px system-ui, sans-serif";
    g.fillText(names[e], ox, oy + R + 17);
  }

  // --- the escape neurons: giant fibers and the parallel DNs, voltage as a bar, pink when spiking
  const roles = m.roles, ids = [...roles.gf, ...roles.parallel], kinds = m.neuron_type ?? [];
  const top = cy + R + 50, bw = Math.min(24, (W - 24) / ids.length - 6), h = 62, slot = (W - 24) / ids.length;
  g.font = "600 12px system-ui, sans-serif"; g.textAlign = "left"; g.fillStyle = ink;
  g.fillText("Escape neurons: bar = voltage", 12, top - 10);
  g.font = "11.5px system-ui, sans-serif"; g.textAlign = "center";
  ids.forEach((j, k) => {
    const x = 12 + k * slot + slot / 2;
    const level = (brain.voltage(j) - lif["v_rst"]) / (lif["v_th"] - lif["v_rst"]);
    const fired = nowMs - round.lastSpikeMs[j] < 15;
    g.fillStyle = "#0d171a"; g.fillRect(x - bw / 2, top, bw, h);
    const fh = Math.max(0, Math.min(1, level)) * h;
    g.fillStyle = fired ? pink : cyan;
    g.fillRect(x - bw / 2, top + h - fh, bw, fh);
    g.strokeStyle = fired ? pink : "#5b8089"; g.lineWidth = 1.2; g.strokeRect(x - bw / 2, top, bw, h);
    g.fillStyle = fired ? "#fff" : muted;
    g.fillText((kinds[j] ?? "?").replace("DNp", "") + roles.target_side[k], x, top + h + 13);
  });
  // which bars are the giant fibers
  const nGf = roles.gf.length;
  if (nGf > 0 && nGf < ids.length) {
    const bracket = (from: number, to: number, label: string) => {
      const x0 = 12 + from * slot + 3, x1 = 12 + to * slot - 3, y = top + h + 22;
      g.strokeStyle = "#35565b"; g.lineWidth = 1.2; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
      g.fillStyle = muted; g.font = "11px system-ui, sans-serif"; g.fillText(label, (x0 + x1) / 2, y + 13);
    };
    bracket(0, nGf, "giant fibers");
    bracket(nGf, ids.length, "parallel escape neurons");
  }
}
