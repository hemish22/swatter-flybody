// Browser entry: input, the 5 ms game loop, the 3D stage, the explanation rail, the post-round card and the leaderboard.
// All rules live in world.ts; this file only turns pointer events into `Input`s and a `Round` into pictures.
// Nothing drawn here is read back by the rules, so the leaderboard server's replay is unaffected.
import { FlyBrain, type Manifest } from "../engine/brain.ts";
import { CONFIG, Round, decideTakeoff, simulateRound, type Card, type Input } from "./world.ts";
import { encodeTrace, nextSeed, quantise, runSeeds, rulesFor } from "./trace.ts";
import { Lab } from "../lab/ui.ts";
import { drawPopulation } from "./overlay.ts";
import { Stage } from "./scene3d.ts";
import { Activity, buildCircuit, drawCircuit } from "./circuit.ts";
import { deriveStory, resultLine, type Snapshot } from "./story.ts";
import { Sfx, diff as soundDiff, type SoundState } from "./sound.ts";

const arena = document.getElementById("arena") as HTMLCanvasElement; // Lab's 2D view
const sceneCanvas = document.getElementById("scene") as HTMLCanvasElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const circuitCanvas = document.getElementById("circuit") as HTMLCanvasElement;
const pop = document.getElementById("pop") as HTMLCanvasElement;
const CLASSIC_SWATS = 20;
const STREAK_LIMIT = 3; // Streak ends at this many swats in a row that do not hit
/** The replay runs one 5 ms tick every 50 ms of wall time: 0.1x, slow enough to watch the neurons decide. */
const REPLAY_MS_PER_TICK = 50;
/** Only the last stretch of a swat is replayed in slow motion; the hover before it is fast-forwarded. */
const REPLAY_TICKS = 60;
/** How long, in simulated ms, the scene keeps playing after a round ends so the takeoff can finish. */
const AFTER_CAP_MS = 520;
/** After the round ends nothing is being decided any more, so the takeoff plays at 0.35x: slow enough to watch the fly climb. */
const AFTER_SPEED = 0.35;
/** The result banner waits until a fly that left is visibly airborne, so it never names a takeoff over a fly still on the table. */
const AIRBORNE_MS = 140;
type Mode = "classic" | "streak" | "lab";
const $ = (id: string) => document.getElementById(id)!;
const store = {
  get: (k: string): string | null => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string): void => { try { localStorage.setItem(k, v); } catch { /* private window: the setting just is not kept */ } },
};
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function loadBrain(): Promise<{ brain: FlyBrain; bin: ArrayBuffer }> {
  const manifest: Manifest = await fetch("brain/manifest.json").then((r) => r.json());
  const [bin, wasm] = await Promise.all([
    fetch("brain/" + manifest.binary).then((r) => r.arrayBuffer()),
    fetch("brain/engine.wasm").then((r) => r.arrayBuffer()),
  ]);
  return { brain: await FlyBrain.create(manifest, bin, wasm), bin };
}

// ------------------------------------------------------------------------------------- giant-fiber trace

function drawOverlay(round: Round): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1), w = overlay.clientWidth, h = overlay.clientHeight;
  if (w === 0 || h === 0) return;
  if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) { overlay.width = Math.round(w * dpr); overlay.height = Math.round(h * dpr); }
  const octx = overlay.getContext("2d")!;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pad = { l: 40, r: 10, t: 8, b: 6 };
  octx.clearRect(0, 0, w, h);
  const n = Math.max(round.gfVoltage.right.length, 40);
  const x = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const top = 0.6, split = pad.t + (h - pad.t - pad.b) * top;
  const yV = (v: number) => pad.t + ((-44 - v) / (-44 + 53)) * (split - pad.t - 6);
  const yT = (deg: number) => h - pad.b - (deg / 180) * (h - pad.b - split - 6);
  const accent = css("--accent"), spike = css("--bad");
  octx.font = "11px ui-sans-serif, system-ui, sans-serif"; octx.textAlign = "right";
  octx.fillStyle = "#8ea3a6";
  for (const v of [-52, -48, -44]) octx.fillText(String(v), pad.l - 6, yV(v) + 4);
  for (const d of [0, 90, 180]) octx.fillText(`${d}°`, pad.l - 6, yT(d) + 4);
  octx.strokeStyle = "rgba(255,255,255,.07)"; octx.lineWidth = 1;
  for (const v of [-52, -48]) { octx.beginPath(); octx.moveTo(pad.l, yV(v)); octx.lineTo(w - pad.r, yV(v)); octx.stroke(); }
  for (const d of [0, 90, 180]) { octx.beginPath(); octx.moveTo(pad.l, yT(d)); octx.lineTo(w - pad.r, yT(d)); octx.stroke(); }
  octx.strokeStyle = "#ffd9ec"; octx.lineWidth = 1.2; octx.setLineDash([3, 4]);
  octx.beginPath(); octx.moveTo(pad.l, yV(-45)); octx.lineTo(w - pad.r, yV(-45)); octx.stroke(); octx.setLineDash([]);
  octx.fillStyle = "#ffd9ec"; octx.textAlign = "right"; octx.fillText("threshold -45 mV", w - pad.r, yV(-45) - 5);

  const line = (xs: number[], f: (v: number) => number, color: string, width: number, dash: number[] = [], glow = 0) => {
    octx.save();
    octx.strokeStyle = color; octx.lineWidth = width; octx.setLineDash(dash); octx.lineJoin = "round";
    if (glow) { octx.shadowColor = color; octx.shadowBlur = glow; }
    octx.beginPath();
    xs.forEach((v, i) => (i ? octx.lineTo(x(i), f(v)) : octx.moveTo(x(i), f(v))));
    octx.stroke();
    octx.restore();
  };
  // swatter size: filled so the growth reads as a ramp
  if (round.thetaDeg.length > 1) {
    octx.beginPath();
    round.thetaDeg.forEach((v, i) => (i ? octx.lineTo(x(i), yT(v)) : octx.moveTo(x(i), yT(v))));
    octx.lineTo(x(round.thetaDeg.length - 1), yT(0)); octx.lineTo(x(0), yT(0)); octx.closePath();
    octx.fillStyle = "rgba(143,165,168,.16)"; octx.fill();
  }
  line(round.thetaDeg, yT, "#a9bfc2", 2);
  line(round.gfVoltage.left, yV, accent, 2, [5, 4]);
  line(round.gfVoltage.right, yV, accent, 2.6, [], 8);
  const { gf } = round.firstSpikes;
  if (gf !== null) {
    const i = gf / CONFIG.tickMs;
    octx.strokeStyle = spike; octx.lineWidth = 2; octx.shadowColor = spike; octx.shadowBlur = 10;
    octx.beginPath(); octx.moveTo(x(i), pad.t); octx.lineTo(x(i), h - pad.b); octx.stroke(); octx.shadowBlur = 0;
    octx.fillStyle = spike; octx.font = "600 12px ui-sans-serif, system-ui, sans-serif"; octx.textAlign = "left";
    const label = `giant fiber spike, ${gf.toFixed(1)} ms`;
    const tw = octx.measureText(label).width;
    if (x(i) + 8 + tw > w - pad.r) { octx.textAlign = "right"; octx.fillText(label, x(i) - 8, pad.t + 22); }
    else octx.fillText(label, x(i) + 8, pad.t + 22);
  }
  octx.fillStyle = "#8ea3a6"; octx.font = "11px ui-sans-serif, system-ui, sans-serif"; octx.textAlign = "left";
  octx.fillText("mV", pad.l + 4, yV(-48) - 5); octx.fillText("swatter size", pad.l + 4, split + 14);
}

// -------------------------------------------------------------------------------------------- the card

const fmt = (v: number | null, d = 1, unit = "") => (v === null ? "–" : v.toFixed(d) + unit);

function showCard(card: Card, seed: number, autoReplay: boolean): void {
  const margin = card.marginMs === null ? "–" : card.marginMs >= 0 ? `left ${card.marginMs.toFixed(1)} ms early` : `${(-card.marginMs).toFixed(1)} ms too late`;
  $("card").innerHTML = `<div class="detail">
      <span>Your strike speed <b>${card.strikeRvMs === null ? "–" : `r/v ${fmt(card.strikeRvMs, 0, " ms")}`}</b></span>
      <span>Swatter size at the spike <b>${fmt(card.thetaAtGfDeg, 0, "°")}</b></span>
      <span>Takeoff <b>${card.mode === "none" ? "none" : card.mode + "-mode"}</b></span>
      <span>Timing <b>${margin}</b></span>
      <span>Heading <b>${card.headingSide ? "turned " + card.headingSide : "no side"}</b></span></div>
    <div class="actions"><button class="primary" id="again">Next swat</button><button id="replay">Replay in slow motion</button>
      <label class="check"><input type="checkbox" id="auto"${autoReplay ? " checked" : ""}> Replay automatically</label></div>
    <div class="note" style="margin:0">Seed ${seed}. The same seed and your mouse trace replay exactly.</div>`;
}

function showHint(): void {
  $("card").innerHTML = `<div class="note" style="margin:0">Move to hover the swatter over the fly. Hold the button to charge a strike, release to swing.</div>`;
}

// --------------------------------------------------------------------------------------------- the game

async function main(): Promise<void> {
  const { brain, bin } = await loadBrain();
  const rules = rulesFor(brain.manifest); // sent with a submission: the server refuses a page running other rules
  const circuit = buildCircuit(brain.manifest, bin, brain.drive);
  const lif = brain.manifest.lif;
  const stage = new Stage(sceneCanvas);
  const stageEl = $("stage"), pipEl = $("pip");
  const layout = () => {
    const r = stageEl.getBoundingClientRect();
    stage.resize(r.width, r.height, { w: pipEl.offsetWidth, h: pipEl.offsetHeight });
  };
  new ResizeObserver(layout).observe(stageEl);
  layout();

  const input: Input = { x: 0, y: 0, down: false };
  let mode: Mode = "classic";
  let seeds: number[] = runSeeds((Date.now() & 0x7fffffff) >>> 0, CLASSIC_SWATS);
  let run: { id: string } | null = null; // a server-issued run: only these can be submitted
  let log: Array<{ trace: number[]; claimed: { outcome: string; mode: string; headingSide: string | null } }> = [];
  let seed = seeds[0];
  let round = new Round(brain, seed);
  let activity = new Activity(circuit);
  let trace: Input[] = [];
  let last = performance.now(), acc = 0, liveAfter = 0, frameNo = 0;
  let swats = 0, hits = 0, misses = 0;
  let replaying: { round: Round; trace: Input[]; i: number; act: Activity; after: number } | null = null;
  let streakEnded = false;
  let autoReplay = store.get("swatter-auto-replay") !== "0";
  let autoTimer: ReturnType<typeof setTimeout> | undefined;
  const params = new URLSearchParams(location.search);
  const demo = params.get("demo");
  const frozenAfter = !!demo && params.has("after"); // a screenshot of a chosen moment must not drift
  const lab = new Lab(brain, { arena, panel: $("labpanel") });

  // sound: plays back what the engine produced (see sound.ts); a browser only allows it after a click or key press
  const sfx = new Sfx();
  sfx.setEnabled(store.get("swatter-sound") !== "0");
  const soundBtn = $("sound");
  const showSound = () => { soundBtn.setAttribute("aria-pressed", String(sfx.enabled)); soundBtn.textContent = sfx.enabled ? "Sound on" : "Sound off"; };
  showSound();
  soundBtn.addEventListener("click", () => { sfx.unlock(); sfx.setEnabled(!sfx.enabled); store.set("swatter-sound", sfx.enabled ? "1" : "0"); showSound(); });
  for (const ev of ["pointerdown", "keydown"] as const) window.addEventListener(ev, () => sfx.unlock(), { passive: true });
  const roundIds = new WeakMap<Round, number>();
  let nextRoundId = 1;
  let prevSound: SoundState | null = null;

  if (params.get("eyes") === "1") ($("poppanel").querySelector("details") as HTMLDetailsElement).open = true; // for screenshots

  const cursorToMm = (e: PointerEvent) => { const p = stage.pointerToMm(e.clientX, e.clientY); input.x = p.x; input.y = p.y; };
  sceneCanvas.addEventListener("pointermove", cursorToMm);
  sceneCanvas.addEventListener("pointerdown", (e) => { cursorToMm(e); input.down = true; sceneCanvas.setPointerCapture(e.pointerId); });
  sceneCanvas.addEventListener("pointerup", () => { input.down = false; });
  sceneCanvas.addEventListener("pointercancel", () => { input.down = false; });
  sceneCanvas.addEventListener("contextmenu", (e) => e.preventDefault()); // a long press on touch must charge, not open a menu

  /** Play a whole trace silently, keeping the activity bookkeeping the panels need. */
  function playThrough(r: Round, act: Activity, tr: readonly Input[], upTo = tr.length): number {
    let i = 0;
    while (i < upTo && r.phase !== "done") { r.tick(tr[i++]); act.tick(r); }
    return i;
  }

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
      $("mode-note").textContent = "Classic: 20 swats, your score is the number of hits."; // the static site has no leaderboard server
    }
    newRound();
  }

  const LAB_HIDDEN = ["stats", "card", "mode-note", "story", "circuitpanel", "brainpanel", "poppanel", "board"];
  let boardOk = true; // false once the leaderboard server has failed to answer: the static site has none, so the card goes away
  function setMode(m: Mode): void {
    mode = m;
    for (const t of document.querySelectorAll<HTMLElement>("[data-mode]")) t.setAttribute("aria-pressed", String(t.dataset.mode === m));
    const isLab = m === "lab";
    $("labpanel").hidden = !isLab;
    for (const id of LAB_HIDDEN) $(id).hidden = isLab || (id === "board" && !boardOk);
    arena.hidden = !isLab; sceneCanvas.hidden = isLab; $("pip").hidden = isLab; $("banner").hidden = true; $("replaying").hidden = true; $("hud-charge").hidden = true;
    $("sub").textContent = isLab
      ? "" // the Lab panel explains itself, and the arena draws its own legend in this corner
      : "Hover over the fly. Hold the button to charge, release to strike.";
    if (isLab) return;
    $("stats-rows").innerHTML = m === "classic"
      ? `<div><span class="big" id="n">0 / ${CLASSIC_SWATS}</span><span class="lab">swats</span></div><div><span class="big" id="hits">0</span><span class="lab">hits</span></div>`
      : `<div><span class="big" id="hits">0</span><span class="lab">hits</span></div><div><span class="big" id="n">0 / ${STREAK_LIMIT}</span><span class="lab">swats in a row that did not hit</span></div><div><span class="big" id="best">${store.get("swatter-best-streak") ?? "0"}</span><span class="lab">best</span></div>`;
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
    clearTimeout(autoTimer);
    if (mode === "classic") seed = seeds[Math.min(swats, CLASSIC_SWATS - 1)];
    else { seed = nextSeed(seeds[0]); seeds[0] = seed; } // Streak: an endless local chain, never submitted
    round = new Round(brain, seed);
    activity = new Activity(circuit);
    trace = [];
    replaying = null;
    liveAfter = 0;
    $("banner").hidden = true;
    showHint();
  }

  function startReplay(): void {
    clearTimeout(autoTimer);
    if (!round.card) return;
    const r = new Round(brain, seed), act = new Activity(circuit), tr = trace.slice();
    const i = playThrough(r, act, tr, Math.max(0, tr.length - REPLAY_TICKS)); // fast-forward the hover
    replaying = { round: r, trace: tr, i, act, after: 0 };
    acc = 0;
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
    showCard(card, seed, autoReplay);
    const over = mode === "classic" ? swats >= CLASSIC_SWATS : streakEnded;
    if (over) $("again").textContent = mode === "classic" ? "New run" : "New streak";
    if (mode === "streak" && streakEnded) $("card").insertAdjacentHTML("beforeend", `<div class="note"><b>Streak over: ${hits} hit${hits === 1 ? "" : "s"}.</b></div>`);
    if (mode === "classic" && over) offerSubmit();
    $("again").addEventListener("click", () => {
      if (mode === "classic" && over) void startClassicRun();
      else if (mode === "streak" && over) { hits = 0; swats = 0; misses = 0; streakEnded = false; $("hits").textContent = "0"; $("n").textContent = `0 / ${STREAK_LIMIT}`; newRound(); }
      else newRound();
    });
    $("replay").addEventListener("click", startReplay);
    $("auto").addEventListener("change", (e) => { autoReplay = (e.target as HTMLInputElement).checked; store.set("swatter-auto-replay", autoReplay ? "1" : "0"); });
    if (autoReplay && !demo) autoTimer = setTimeout(startReplay, 900);
  }

  function offerSubmit(): void {
    const el = $("submit");
    if (!run) { el.innerHTML = `<div class="note"><b>Run finished: ${hits} of ${CLASSIC_SWATS}.</b></div>`; return; }
    el.innerHTML = `<h2 style="margin-top:12px">Submit ${hits} of ${CLASSIC_SWATS}</h2><p><label for="name">Name</label> <input id="name" maxlength="24" value="${esc(store.get("swatter-name") ?? "")}"> <button id="send" class="primary">Submit</button></p><div id="send-note" class="note">The server replays all 20 swats from your seed and cursor trace and keeps the score only if they match.</div>`;
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
    } catch { boardOk = false; $("board").hidden = true; }
  }
  void loadBoard();

  // ------------------------------------------------------------------------------------- what to draw

  const headings = new WeakMap<Round, { n: number; side: "left" | "right" | null }>();
  /** The side the spikes chose, once there are enough of them; the card's own answer once the round is over. */
  function headingSide(r: Round): "left" | "right" | null {
    if (r.card) return r.card.headingSide;
    if (r.leftAtMs === null) return null;
    const hit = headings.get(r), n = r.engine.spikeCount;
    if (hit && hit.n === n) return hit.side;
    const side = r.engine.outcome().headingSide;
    headings.set(r, { n, side });
    return side;
  }

  const gfIds = brain.manifest.roles.gf;
  function crouchLevel(r: Round): number {
    let v = -Infinity;
    for (const j of gfIds) v = Math.max(v, r.engine.voltage(j));
    return Math.pow(Math.max(0, Math.min(1, (v - lif["v0"]) / (lif["v_th"] - lif["v0"]))), 1.6);
  }

  function snapshot(r: Round, act: Activity): Snapshot {
    const sum = act.summary(r);
    const { gf, parallel } = r.firstSpikes;
    const tk = decideTakeoff(gf, parallel, r.timeMs);
    const thetaAt = r.card?.thetaAtGfDeg ?? (gf === null ? null : r.thetaDeg[Math.min(r.thetaDeg.length - 1, Math.round(gf / CONFIG.tickMs))] ?? null);
    return {
      timeMs: r.timeMs, thetaDeg: r.thetaDeg[r.thetaDeg.length - 1] ?? 0, rateDegPerMs: r.lastRate, growthStartsMs: act.growthStarts,
      lplc2Active: sum.active.lplc2, lplc2Driven: circuit.driven, firstLplc2Ms: sum.first.lplc2, firstGfMs: gf, thetaAtGfDeg: thetaAt,
      thresholdMv: lif["v_th"], takeoff: tk, headingSide: headingSide(r),
    };
  }

  const chain = $("chain");
  chain.innerHTML = [0, 1, 2, 3].map(() => `<li data-state="idle"><i></i><b></b><span></span><em></em></li>`).join("");
  const chainItems = [...chain.children] as HTMLElement[];
  function renderRail(r: Round, act: Activity, after: number): void {
    const snap = snapshot(r, act);
    deriveStory(snap).forEach((s, i) => {
      const li = chainItems[i];
      const state = r.phase === "done" && s.state === "live" ? "idle" : s.state; // a finished round has nothing still in progress
      if (li.dataset.state !== state) li.dataset.state = state;
      const [, b, span, em] = li.children as unknown as HTMLElement[];
      if (b.textContent !== s.title) b.textContent = s.title;
      if (span.textContent !== s.detail) span.textContent = s.detail;
      const cav = s.caveat ?? "";
      if (em.textContent !== cav) em.textContent = cav;
    });
    const leave = snap.takeoff?.leaveMs ?? null;
    const start = leave === null ? null : [...act.growthStarts].reverse().find((t) => t <= leave) ?? act.growthStarts[0] ?? null;
    $("reaction").innerHTML = leave !== null && start !== null ? `Reaction time: <b>${(leave - start).toFixed(1)} ms</b> from first growth to leaving the table` : "";
    const sum = act.summary(r);
    const vNow = r.gfVoltage.right[r.gfVoltage.right.length - 1] ?? lif["v_rst"];
    const active = sum.active.lplc2 + sum.active.relay + sum.active.gf + sum.active.parallel;
    $("counts").innerHTML = `<div><b class="${sum.total > 0 ? "hot" : ""}">${sum.total}</b>spikes this round</div>`
      + `<div><b class="${active > 0 ? "cool" : ""}">${active}<small> of ${circuit.n}</small></b>neurons firing now</div>`
      + `<div><b>${sum.active.gf + sum.active.parallel}<small> of ${circuit.counts.gf + circuit.counts.parallel}</small></b>escape neurons firing</div>`
      + `<div><b>${vNow.toFixed(1)}<small> mV</small></b>giant fiber, jumps at ${lif["v_th"]}</div>`;
    void after;
  }

  /** What the page currently shows: the live round, or its slow-motion replay. */
  function view(): { r: Round; act: Activity; after: number } {
    return replaying ? { r: replaying.round, act: replaying.act, after: replaying.after } : { r: round, act: activity, after: liveAfter };
  }

  function draw(now: number): void {
    const { r, act, after } = view();
    stage.render({ round: r, afterMs: after, crouch: crouchLevel(r), headingSide: headingSide(r), nowMs: now });
    drawCircuit(circuitCanvas, r, circuit, after);
    drawOverlay(r);
    if ((frameNo & 1) === 0) renderRail(r, act, after);
    const pe = ($("poppanel").querySelector("details") as HTMLDetailsElement);
    if (pe.open) drawPopulation(pop, r);

    // sound for what just changed on screen
    if (!roundIds.has(r)) roundIds.set(r, nextRoundId++);
    const t = r.timeMs + after;
    const now_: SoundState = {
      roundId: roundIds.get(r)!, phase: r.phase, spikeCount: r.engine.spikeCount, gfSpiked: r.firstSpikes.gf !== null,
      left: r.leftAtMs !== null && t >= r.leftAtMs, outcome: r.card?.outcome ?? null,
    };
    sfx.play(soundDiff(prevSound, now_));
    prevSound = now_;
    sfx.setCharge(r.phase === "hover" && !replaying ? r.charge : 0);

    const tag = $("hud-charge");
    if (r.phase === "hover" && r.charge > 0 && !replaying) {
      const v = CONFIG.strikeSpeedMin + (CONFIG.strikeSpeedMax - CONFIG.strikeSpeedMin) * r.charge;
      tag.hidden = false;
      tag.textContent = `${v.toFixed(1)} m/s, r/v ${(CONFIG.swatterHalfWidthMm / v).toFixed(0)} ms`;
      const p = stage.project(r.swatter.x, r.swatter.y - CONFIG.swatterHalfWidthMm - 14, 0);
      tag.style.left = `${p.x}px`; tag.style.top = `${p.y}px`;
    } else tag.hidden = true;

    const banner = $("banner");
    const flew = r.card?.outcome === "escaped" || r.card?.outcome === "spooked";
    const showBanner = !!r.card && (!replaying || replaying.round.phase === "done") && (!flew || after >= AIRBORNE_MS);
    banner.hidden = !showBanner;
    if (showBanner) {
      const [title, ...rest] = resultLine(r.card!.outcome, r.card!.marginMs).split(". ");
      const html = `<b>${title}</b><span>${rest.join(". ")}</span>`;
      if (banner.innerHTML !== html) banner.innerHTML = html;
      banner.dataset.outcome = r.card!.outcome;
    }
    $("replaying").hidden = !(replaying && replaying.round.phase !== "done");
  }

  function frame(now: number): void {
    frameNo++;
    const wall = Math.min(now - last, 100);
    last = now;
    if (mode === "lab") { lab.frame(now); requestAnimationFrame(frame); return; }
    acc += wall;
    if (replaying) {
      while (acc >= REPLAY_MS_PER_TICK && replaying.round.phase !== "done") {
        acc -= REPLAY_MS_PER_TICK;
        replaying.round.tick(replaying.trace[Math.min(replaying.i++, replaying.trace.length - 1)]);
        replaying.act.tick(replaying.round);
      }
      if (replaying.round.phase === "done") { acc = 0; replaying.after = Math.min(AFTER_CAP_MS, replaying.after + wall * AFTER_SPEED); }
    } else {
      const over = mode === "classic" ? swats >= CLASSIC_SWATS : streakEnded;
      while (!over && acc >= CONFIG.tickMs && round.phase !== "done") {
        acc -= CONFIG.tickMs;
        const inp = quantise(input); // the round is played on exactly what the server will replay
        trace.push(inp);
        round.tick(inp);
        activity.tick(round);
        if ((round.phase as string) === "done") finished(); // tick() moves the phase; TypeScript cannot see that
      }
      if ((round.phase as string) === "done" || over) { acc = 0; if (!frozenAfter) liveAfter = Math.min(AFTER_CAP_MS, liveAfter + wall * AFTER_SPEED); }
    }
    draw(now);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  const first = params.get("mode");
  setMode(first === "lab" || first === "streak" ? first : "classic");

  // Scripted rounds for screenshots and tests: ?demo=hit | escape | spook, or ?mode=lab&demo=lab (a fired loom).
  // Add &after=<ms> to freeze the scene that many simulated ms past the end of the round, e.g. mid-takeoff.
  if (demo === "lab") { lab.demo(); document.title = "SWATTER demo lab"; }
  else if (demo) {
    const s = 7, fly = new Round(brain, s).fly, at = { x: fly.x + 20, y: fly.y };
    const tr: Input[] = [];
    if (demo === "spook") { for (let i = 0; i < 100; i++) tr.push({ x: fly.x + 200 - i * 2, y: fly.y, down: false }); }
    const hold = demo === "escape" ? 1 : 200;
    if (demo !== "spook") { for (let i = 0; i < 40; i++) tr.push({ ...at, down: false }); for (let i = 0; i < hold; i++) tr.push({ ...at, down: true }); }
    for (let i = 0; i < 200; i++) tr.push({ ...(demo === "spook" ? { x: fly.x, y: fly.y } : at), down: false });
    const r = simulateRound(brain, s, tr);
    seed = s; round = new Round(brain, s); activity = new Activity(circuit);
    playThrough(round, activity, tr);
    trace = tr.slice(0, round.tickIndex + 1); // what a played round would have recorded: the replay starts from its end
    swats = 1; hits = r.card!.outcome === "hit" ? 1 : 0;
    liveAfter = Number(params.get("after") ?? 0);
    document.getElementById("n")!.textContent = `1 / ${CLASSIC_SWATS}`; document.getElementById("hits")!.textContent = String(hits);
    showCard(round.card!, s, false);
    (window as unknown as { __swatterDemo: unknown }).__swatterDemo = r.card;
    document.title = `SWATTER demo ${demo}: ${r.card!.outcome}`;
    if (params.get("replay") === "1") startReplay();
  }
}

main().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<pre style="color:#f66">Could not start: ${String(e)}</pre>`); });
