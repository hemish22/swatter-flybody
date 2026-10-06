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

export function drawPopulation(canvas: HTMLCanvasElement, round: Round): void {
  const g = canvas.getContext("2d")!, W = canvas.width, H = canvas.height;
  g.clearRect(0, 0, W, H);
  const brain = round.engine, m = brain.manifest, lif = m.lif;
  const view = eyeView(round.fly, round.swatter);
  const nowMs = round.timeMs;
  const R = 80, step = 8, cy = 92, cx = [W / 4 + 2, (3 * W) / 4 - 2];
  const lattice = hexLattice(R, step);
  const px = (deg: number) => (deg / ECC_MAX) * R; // plane degrees to pixels
  g.font = "10px system-ui"; g.textAlign = "center";

  // --- two eyes: the swatter's footprint on the lattice (dark = covered), LPLC2 receptive fields on top
  const pos = [view.right, view.left], names = ["right eye", "left eye"];
  for (let e = 0; e < 2; e++) {
    const [ox, oy] = [cx[e], cy];
    const [sx, sy] = pos[e];
    const half = view.thetaDeg / 2;
    for (const [lx, ly] of lattice) {
      const dx = (lx / R) * ECC_MAX - sx, dy = (-ly / R) * ECC_MAX - sy; // plane y is up, screen y down
      const covered = Math.hypot(dx, dy) <= half;
      g.fillStyle = covered ? "#2a2c31" : css("--panel") === "#fff" ? "#ece9e1" : "#32353c";
      g.beginPath(); g.arc(ox + lx, oy + ly, step * 0.42, 0, 7); g.fill();
    }
    g.strokeStyle = css("--muted"); g.lineWidth = 1; g.globalAlpha = 0.5;
    g.beginPath(); g.arc(ox, oy, R, 0, 7); g.stroke(); g.globalAlpha = 1;
    // LPLC2 neurons of this eye: ring at the receptive-field centre, filled by membrane potential
    const d = brain.drive;
    for (let i = 0; i < d.neuron.length; i++) {
      if (d.eye[i] !== e) continue;
      const rx = d.rf[2 * i], ry = d.rf[2 * i + 1];
      if (Math.hypot(rx, ry) > ECC_MAX) continue;
      const v = brain.voltage(d.neuron[i]);
      const level = (v - lif["v_rst"]) / (lif["v_th"] - lif["v_rst"]);
      const fired = nowMs - round.lastSpikeMs[d.neuron[i]] < 15;
      g.fillStyle = fired ? css("--bad") : `rgba(240,160,75,${lerp(0.08, 0.95, level)})`;
      g.strokeStyle = css("--warn"); g.lineWidth = 1;
      g.beginPath(); g.arc(ox + px(rx), oy - px(ry), 2.6, 0, 7); g.fill(); g.stroke();
    }
    g.fillStyle = css("--muted");
    g.fillText(names[e], ox, oy + R + 12);
  }
  g.fillStyle = css("--muted");
  g.fillText("dark: the swatter's footprint (geometry, not a neural response)", W / 2, cy + R + 26);
  g.fillText("rings: LPLC2 neurons, brighter = higher voltage, red = spiking", W / 2, cy + R + 38);

  // --- the escape neurons: giant fibers and the parallel DNs, voltage as a bar, red when spiking
  const roles = m.roles, ids = [...roles.gf, ...roles.parallel], kinds = m.neuron_type ?? [];
  const top = cy + R + 66, bw = Math.min(26, (W - 20) / ids.length - 4);
  g.textAlign = "center";
  ids.forEach((j, k) => {
    const x = 10 + k * ((W - 20) / ids.length) + bw / 2, h = 56;
    const level = (brain.voltage(j) - lif["v_rst"]) / (lif["v_th"] - lif["v_rst"]);
    const fired = nowMs - round.lastSpikeMs[j] < 15;
    g.strokeStyle = css("--muted"); g.lineWidth = 1; g.strokeRect(x - bw / 2, top, bw, h);
    g.fillStyle = fired ? css("--bad") : css("--accent");
    const fh = Math.max(0, Math.min(1, level)) * h;
    g.fillRect(x - bw / 2, top + h - fh, bw, fh);
    g.fillStyle = css("--muted");
    const label = (kinds[j] ?? "?").replace("DNp", "") + roles.target_side[k];
    g.fillText(label, x, top + h + 11);
  });
  g.textAlign = "left"; g.fillText("escape neurons (DNp): bar = voltage, red = spiking", 10, top - 6);
}
