// Browser entry: input, the 5 ms game loop, drawing, the post-round card and the live overlay.
// All rules live in world.ts; this file only turns mouse events into `Input`s and a `Round` into pixels.
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { CONFIG, Round, simulateRound, type Card, type Input } from "./world.ts";
import { encodeTrace, nextSeed, quantise, runSeeds, rulesFor } from "./trace.ts";
import { Lab } from "../lab/ui.ts";

const arena = document.getElementById("arena") as HTMLCanvasElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const ctx = arena.getContext("2d")!;
const octx = overlay.getContext("2d")!;
const SCALE = arena.width / CONFIG.arenaMm.w; // px per mm
const FLY_SPRITE_SCALE = 8; // a 2.5 mm fly is 4 px at this scale; the sprite is drawn larger, the hit test is not
const CLASSIC_SWATS = 20;
const STREAK_LIMIT = 3; // Streak ends at this many swats in a row that do not hit
type Mode = "classic" | "streak" | "lab";
const $ = (id: string) => document.getElementById(id)!;
const store = {
  get: (k: string): string | null => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string): void => { try { localStorage.setItem(k, v); } catch { /* private window: the best streak just is not kept */ } },
};
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const toPx = (x: number, y: number): [number, number] => [arena.width / 2 + x * SCALE, arena.height / 2 - y * SCALE];

async function loadBrain(): Promise<FlyBrain> {
  const manifest: Manifest = await fetch("brain/manifest.json").then((r) => r.json());
  const [bin, wasm] = await Promise.all([
    fetch("brain/" + manifest.binary).then((r) => r.arrayBuffer()),
    fetch("brain/engine.wasm").then((r) => r.arrayBuffer()),
  ]);
  return FlyBrain.create(manifest, bin, wasm);
}

// ------------------------------------------------------------------------------------------ drawing

function drawArena(round: Round, input: Input, showSwatter = true): void {
  const w = arena.width, h = arena.height;
  ctx.fillStyle = css("--panel") === "#fff" ? "#d9d6cd" : "#3a3d44";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(128,128,128,.18)";
  ctx.lineWidth = 1;
  for (let x = -300; x <= 300; x += 50) { const [px] = toPx(x, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke(); }
  for (let y = -200; y <= 200; y += 50) { const [, py] = toPx(0, y); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke(); }

  // fly, unless it has already left
  const gone = round.card ? round.card.outcome === "escaped" || round.card.outcome === "spooked" : round.leftAtMs !== null && round.leftAtMs <= round.timeMs;
  const [fx, fy] = toPx(round.fly.x, round.fly.y);
  if (!gone) {
    ctx.save();
    ctx.translate(fx, fy);
    ctx.rotate(-round.fly.heading);
    ctx.fillStyle = "#15161a";
    ctx.beginPath(); ctx.ellipse(0, 0, 1.25 * SCALE * FLY_SPRITE_SCALE, 0.7 * SCALE * FLY_SPRITE_SCALE, 0, 0, 7); ctx.fill();
    ctx.fillStyle = "#5aa9ff";
    ctx.beginPath(); ctx.arc(1.1 * SCALE * FLY_SPRITE_SCALE, 0, 0.45 * SCALE * FLY_SPRITE_SCALE, 0, 7); ctx.fill(); // head: forward is +x in the body frame
    ctx.restore();
  } else {
    ctx.strokeStyle = css("--warn"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(fx, fy, 14, 0, 7); ctx.setLineDash([3, 4]); ctx.stroke(); ctx.setLineDash([]);
  }

  if (!showSwatter) return;
  const sw = round.swatter;
  const [sx, sy] = toPx(sw.x, sw.y);
  const r = CONFIG.swatterHalfWidthMm * SCALE;
  const lift = 1 + sw.z / 250; // a higher swatter looks a little bigger
  ctx.fillStyle = "rgba(0,0,0,.18)";
  ctx.beginPath(); ctx.arc(sx, sy, r, 0, 7); ctx.fill(); // shadow = the footprint that decides the hit
  ctx.fillStyle = "rgba(20,22,26,.78)";
  ctx.strokeStyle = round.phase === "strike" ? css("--bad") : css("--ink");
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(sx, sy - sw.z * 0.15, r * lift, 0, 7); ctx.fill(); ctx.stroke();
  if (round.charge > 0) {
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 5;
    ctx.beginPath(); ctx.arc(sx, sy, r * 1.25, -Math.PI / 2, -Math.PI / 2 + round.charge * 2 * Math.PI); ctx.stroke();
    const v = CONFIG.strikeSpeedMin + (CONFIG.strikeSpeedMax - CONFIG.strikeSpeedMin) * round.charge;
    ctx.fillStyle = css("--ink"); ctx.font = "13px system-ui"; ctx.textAlign = "center";
    ctx.fillText(`${v.toFixed(1)} m/s  (r/v ${(CONFIG.swatterHalfWidthMm / v).toFixed(0)} ms)`, sx, sy + r * 1.25 + 18);
  }
  void input;
}

function drawOverlay(round: Round): void {
  const w = overlay.width, h = overlay.height, pad = { l: 34, r: 6, t: 6, b: 6 };
  octx.clearRect(0, 0, w, h);
  const n = Math.max(round.gfVoltage.right.length, 40);
  const x = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const top = 0.55, split = pad.t + (h - pad.t - pad.b) * top;
  const yV = (v: number) => pad.t + ((-44 - v) / (-44 + 53)) * (split - pad.t - 6);
  const yT = (deg: number) => h - pad.b - (deg / 180) * (h - pad.b - split - 6);
  octx.font = "10px system-ui"; octx.fillStyle = css("--muted"); octx.strokeStyle = css("--muted");
  octx.textAlign = "right";
  for (const v of [-52, -48, -44]) octx.fillText(String(v), pad.l - 4, yV(v) + 4);
  for (const d of [0, 90, 180]) octx.fillText(`${d}°`, pad.l - 4, yT(d) + 4);
  octx.setLineDash([2, 3]); octx.lineWidth = 1;
  octx.beginPath(); octx.moveTo(pad.l, yV(-45)); octx.lineTo(w - pad.r, yV(-45)); octx.stroke(); octx.setLineDash([]);

  const line = (xs: number[], f: (v: number) => number, color: string, dash: number[] = []) => {
    octx.strokeStyle = color; octx.lineWidth = 1.5; octx.setLineDash(dash); octx.beginPath();
    xs.forEach((v, i) => (i ? octx.lineTo(x(i), f(v)) : octx.moveTo(x(i), f(v))));
    octx.stroke(); octx.setLineDash([]);
  };
  line(round.thetaDeg, yT, "#8a8f98");
  line(round.gfVoltage.left, yV, css("--accent"), [4, 3]);
  line(round.gfVoltage.right, yV, css("--accent"));
  const { gf } = round.firstSpikes;
  if (gf !== null) {
    const i = gf / CONFIG.tickMs;
    octx.strokeStyle = css("--warn"); octx.lineWidth = 1.5; octx.beginPath(); octx.moveTo(x(i), pad.t); octx.lineTo(x(i), h - pad.b); octx.stroke();
    octx.fillStyle = css("--warn"); octx.textAlign = "left"; octx.fillText("GF spike", Math.min(x(i) + 4, w - 48), pad.t + 10);
  }
  octx.fillStyle = css("--muted"); octx.textAlign = "left"; octx.textAlign = "left"; octx.fillText("mV", pad.l + 4, pad.t + 20); octx.fillText("deg", pad.l + 4, split + 14);
}

// -------------------------------------------------------------------------------------------- the card

const fmt = (v: number | null, d = 1, unit = "") => (v === null ? "–" : v.toFixed(d) + unit);

function showCard(card: Card, seed: number): void {
  const el = document.getElementById("card")!;
  const label: Record<string, string> = { hit: "Hit", escaped: "Escaped", spooked: "Spooked", miss: "Missed", timeout: "Timed out" };
  const margin = card.marginMs === null ? "–" : card.marginMs >= 0 ? `${card.marginMs.toFixed(1)} ms before the swat landed` : `${(-card.marginMs).toFixed(1)} ms too late`;
  el.innerHTML = `<h2>Last swat</h2><div class="outcome ${card.outcome}">${label[card.outcome]}</div><table>
    <tr><td>Your strike r/v</td><td>${fmt(card.strikeRvMs, 0, " ms")}</td></tr>
    <tr><td>Swatter size when the giant fiber fired</td><td>${fmt(card.thetaAtGfDeg, 0, "°")}</td></tr>
    <tr><td>Takeoff</td><td>${card.mode === "none" ? "none" : card.mode + "-mode"}</td></tr>
    <tr><td>Fly left</td><td>${margin}</td></tr>
    <tr><td>Heading</td><td>${card.headingSide ? "turned " + card.headingSide : "no side"}</td></tr></table>
    <div class="note">Seed ${seed}. The same seed and your mouse trace replay exactly.</div>
    <p><button id="again">Next swat</button> <button id="replay">Slow replay</button></p>`;
}

// --------------------------------------------------------------------------------------------- the game

async function main(): Promise<void> {
  const brain = await loadBrain();
  const rules = rulesFor(brain.manifest); // sent with a submission: the server refuses a page running other rules
  const input: Input = { x: 0, y: 0, down: false };
  let mode: Mode = "classic";
  let seeds: number[] = runSeeds((Date.now() & 0x7fffffff) >>> 0, CLASSIC_SWATS);
  let run: { id: string } | null = null; // a server-issued run: only these can be submitted
  let log: Array<{ trace: number[]; claimed: { outcome: string; mode: string; headingSide: string | null } }> = [];
  let seed = seeds[0];
  let round = new Round(brain, seed);
  let trace: Input[] = [];
  let last = performance.now(), acc = 0;
  let swats = 0, hits = 0, misses = 0, replaying: { round: Round; trace: Input[]; i: number } | null = null;
  let streakEnded = false;
  const params = new URLSearchParams(location.search);
  const demo = params.get("demo");
  const lab = new Lab(brain, { arena, panel: $("labpanel") });

  const cursorToMm = (e: PointerEvent) => {
    const rect = arena.getBoundingClientRect();
    input.x = ((e.clientX - rect.left) / rect.width - 0.5) * CONFIG.arenaMm.w;
    input.y = (0.5 - (e.clientY - rect.top) / rect.height) * CONFIG.arenaMm.h;
  };
  arena.addEventListener("pointermove", cursorToMm);
  arena.addEventListener("pointerdown", (e) => { cursorToMm(e); input.down = true; arena.setPointerCapture(e.pointerId); });
  arena.addEventListener("pointerup", () => { input.down = false; });

  async function startClassicRun(): Promise<void> {
    swats = 0; hits = 0; log = []; run = null;
    $("submit").innerHTML = "";
    try {
      const r = await fetch("api/run").then((x) => { if (!x.ok) throw new Error(String(x.status)); return x.json(); }) as { runId: string; seed: number; rules: string };
      run = { id: r.runId };
      seeds = runSeeds(r.seed, CLASSIC_SWATS);
      $("mode-note").textContent = "Ranked run: the server issued the seed and will replay your 20 swats.";
    } catch {
      seeds = runSeeds((Date.now() & 0x7fffffff) >>> 0, CLASSIC_SWATS);
      $("mode-note").textContent = "Practice run: no leaderboard server reachable, so this score cannot be submitted.";
    }
    newRound();
  }

  function setMode(m: Mode): void {
    mode = m;
    for (const t of document.querySelectorAll<HTMLElement>("[data-mode]")) t.setAttribute("aria-pressed", String(t.dataset.mode === m));
    const isLab = m === "lab";
    $("labpanel").hidden = !isLab;
    for (const id of ["stats", "card", "brainpanel", "board"]) $(id).hidden = isLab;
    $("sub").textContent = isLab
      ? "Lab: fire the standard stimuli at a tethered fly and check the circuit's response yourself."
      : "Swat the fly. Its escape reflex is wired from a real fly connectome. Move the mouse to hover the swatter; hold the button to charge, release to strike.";
    if (isLab) return;
    $("stats-title").textContent = m === "classic" ? "Classic" : "Streak";
    $("stats-rows").innerHTML = m === "classic"
      ? `<tr><td>Swats</td><td id="n">0 / ${CLASSIC_SWATS}</td></tr><tr><td>Hits</td><td id="hits">0</td></tr>`
      : `<tr><td>Hits</td><td id="hits">0</td></tr><tr><td>Swats that did not hit, in a row</td><td id="n">0 / ${STREAK_LIMIT}</td></tr><tr><td>Best</td><td id="best">${store.get("swatter-best-streak") ?? "0"}</td></tr>`;
    hits = 0; swats = 0; misses = 0; streakEnded = false;
    if (m === "classic") { if (demo) newRound(); else void startClassicRun(); } // a scripted demo must not race the server fetch
    else {
      seeds = runSeeds((Date.now() & 0x7fffffff) >>> 0, 1); run = null; $("submit").innerHTML = "";
      $("mode-note").textContent = "Streak: play until three swats in a row fail to hit.";
      newRound();
    }
  }
  for (const t of document.querySelectorAll<HTMLElement>("[data-mode]")) t.addEventListener("click", () => setMode(t.dataset.mode as Mode));

  function newRound(): void {
    if (mode === "classic") seed = seeds[Math.min(swats, CLASSIC_SWATS - 1)];
    else { seed = nextSeed(seeds[0]); seeds[0] = seed; } // Streak: an endless local chain, never submitted
    round = new Round(brain, seed);
    trace = [];
    replaying = null;
    $("card").innerHTML = "<h2>Last swat</h2><div class='note'>Hover, hold to charge, release to strike.</div>";
  }

  function finished(): void {
    const card = round.card!;
    swats++;
    if (card.outcome === "hit") { hits++; misses = 0; } else misses++;
    if (mode === "classic") {
      log.push({ trace: encodeTrace(trace), claimed: { outcome: card.outcome, mode: card.mode, headingSide: card.headingSide } });
      $("n").textContent = `${Math.min(swats, CLASSIC_SWATS)} / ${CLASSIC_SWATS}`;
    } else {
      $("n").textContent = `${misses} / ${STREAK_LIMIT}`;
      streakEnded = misses >= STREAK_LIMIT;
      if (streakEnded && hits > Number(store.get("swatter-best-streak") ?? 0)) { store.set("swatter-best-streak", String(hits)); $("best").textContent = String(hits); }
    }
    $("hits").textContent = String(hits);
    showCard(card, seed);
    const over = mode === "classic" ? swats >= CLASSIC_SWATS : streakEnded;
    if (over) $("again").textContent = mode === "classic" ? "New run" : "New streak";
    if (mode === "streak" && streakEnded) $("card").insertAdjacentHTML("beforeend", `<div class="note"><b>Streak over: ${hits} hit${hits === 1 ? "" : "s"}.</b></div>`);
    if (mode === "classic" && over) offerSubmit();
    $("again").addEventListener("click", () => {
      if (mode === "classic" && over) void startClassicRun();
      else if (mode === "streak" && over) { hits = 0; swats = 0; misses = 0; streakEnded = false; $("hits").textContent = "0"; $("n").textContent = `0 / ${STREAK_LIMIT}`; newRound(); }
      else newRound();
    });
    $("replay").addEventListener("click", () => { replaying = { round: new Round(brain, seed), trace: trace.slice(), i: 0 }; });
  }

  function offerSubmit(): void {
    const el = $("submit");
    if (!run) { el.innerHTML = `<div class="note">Run finished: ${hits} of ${CLASSIC_SWATS}. Practice runs are not ranked.</div>`; return; }
    el.innerHTML = `<h2>Submit ${hits} of ${CLASSIC_SWATS}</h2><p><label for="name">Name</label> <input id="name" maxlength="24" value="${esc(store.get("swatter-name") ?? "")}"> <button id="send">Submit</button></p><div id="send-note" class="note">The server replays all 20 swats from your seed and cursor trace and keeps the score only if they match.</div>`;
    $("send").addEventListener("click", async () => {
      const name = ($("name") as HTMLInputElement).value;
      store.set("swatter-name", name);
      ($("send") as HTMLButtonElement).disabled = true;
      try {
        const res = await fetch("api/submit", { method: "POST", body: JSON.stringify({ runId: run!.id, name, rules, swats: log }) });
        const j = await res.json() as { accepted: boolean; hits?: number; rank?: number; reason?: string };
        $("send-note").textContent = j.accepted ? `Accepted: ${j.hits} of ${CLASSIC_SWATS}, rank ${j.rank}.` : `Not accepted: ${j.reason}`;
        if (j.accepted) void loadBoard();
      } catch (e) { $("send-note").textContent = `Could not reach the server: ${String(e)}`; ($("send") as HTMLButtonElement).disabled = false; }
    });
  }

  async function loadBoard(): Promise<void> {
    try {
      const j = await fetch("api/leaderboard?limit=10").then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); }) as { rows: Array<{ name: string; hits: number; swats: number }> };
      $("board-rows").innerHTML = j.rows.length ? j.rows.map((r, i) => `<tr><td>${i + 1}. ${esc(r.name)}</td><td>${r.hits} / ${r.swats}</td></tr>`).join("") : `<tr><td colspan="2" class="note">No verified runs yet.</td></tr>`;
    } catch { $("board-rows").innerHTML = `<tr><td colspan="2" class="note">Leaderboard unavailable here. It is served by server/server.ts.</td></tr>`; }
  }
  void loadBoard();

  function frame(now: number): void {
    if (mode === "lab") { lab.frame(now); last = now; requestAnimationFrame(frame); return; }
    acc += Math.min(now - last, 100); last = now;
    if (replaying) {
      // slow motion: one 5 ms tick every 20 ms of wall time, i.e. 0.25x
      while (acc >= 20 && replaying.round.phase !== "done") { acc -= 20; replaying.round.tick(replaying.trace[Math.min(replaying.i++, replaying.trace.length - 1)]); }
      if (replaying.round.phase === "done") acc = 0;
      drawArena(replaying.round, input, true); drawOverlay(replaying.round);
    } else {
      const over = mode === "classic" ? swats >= CLASSIC_SWATS : streakEnded;
      while (!over && acc >= CONFIG.tickMs && round.phase !== "done") {
        acc -= CONFIG.tickMs;
        const inp = quantise(input); // the round is played on exactly what the server will replay
        trace.push(inp);
        round.tick(inp);
        if ((round.phase as string) === "done") finished(); // tick() moves the phase; TypeScript cannot see that
      }
      if ((round.phase as string) === "done" || over) acc = 0;
      drawArena(round, input); drawOverlay(round);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  const first = params.get("mode");
  setMode(first === "lab" || first === "streak" ? first : "classic");

  // Scripted rounds for screenshots and tests: ?demo=hit | escape | spook, or ?mode=lab&demo=lab (a fired loom)
  if (demo === "lab") { lab.demo(); document.title = "SWATTER demo lab"; }
  else if (demo) {
    const s = 7, fly = new Round(brain, s).fly, at = { x: fly.x + 20, y: fly.y };
    const tr: Input[] = [];
    if (demo === "spook") { for (let i = 0; i < 100; i++) tr.push({ x: fly.x + 200 - i * 2, y: fly.y, down: false }); }
    const hold = demo === "escape" ? 1 : 200;
    if (demo !== "spook") { for (let i = 0; i < 40; i++) tr.push({ ...at, down: false }); for (let i = 0; i < hold; i++) tr.push({ ...at, down: true }); }
    for (let i = 0; i < 200; i++) tr.push({ ...(demo === "spook" ? { x: fly.x, y: fly.y } : at), down: false });
    const r = simulateRound(brain, s, tr);
    seed = s; round = r; trace = tr; swats = 1; hits = r.card!.outcome === "hit" ? 1 : 0;
    document.getElementById("n")!.textContent = `1 / ${CLASSIC_SWATS}`; document.getElementById("hits")!.textContent = String(hits);
    showCard(r.card!, s); drawArena(r, input); drawOverlay(r);
    (window as unknown as { __swatterDemo: unknown }).__swatterDemo = r.card;
    document.title = `SWATTER demo ${demo}: ${r.card!.outcome}`;
  }
}

main().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<pre style="color:#f66">Could not start: ${String(e)}</pre>`); });
