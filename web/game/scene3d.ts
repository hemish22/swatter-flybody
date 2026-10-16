// The 3D stage: a table, the swatter, the fly, and a small fly's-eye view.
//
// Rendering only. Everything it shows is read from a `Round` (positions, phase, the spikes that decided the takeoff);
// it writes nothing back, so the rules in world.ts and the server's replay are untouched.
// Two things on screen are scaled up for legibility and are NOT to scale: the fly (FLY_SCALE x real size) and the
// swatter's height above the table is drawn true, but the camera is angled so a 40 mm hover reads clearly.
// The footprint ring on the table is the exact disc that decides a hit.

import * as THREE from "three";
import { FlyModel } from "./flymodel.ts";
import { CONFIG, type Round } from "./world.ts";

const FLY_SCALE = 13;
/** How high the fly climbs (mm above the table) before it hovers; the picture stays in frame and the fly stays visibly airborne. */
const CLIMB_MM = 170;
const TRAIL_POINTS = 26;
/** Takeoff speed of the illustration, mm per ms of simulated time. Cosmetic. */
const FLIGHT_MMS = 1.1;
const CHARGE_SEGMENTS = 72;
const COLORS = { void: 0x070d10, table: 0x1b2d32, grid: 0x2c444a, accent: 0x4fe3d1, spike: 0xff4fa3 };

export interface StageState {
  round: Round;
  /** Simulated ms past the end of the round, so the fly can finish its takeoff after the result. */
  afterMs: number;
  /** 0 to 1, from the giant fiber's voltage. */
  crouch: number;
  headingSide: "left" | "right" | null;
  /** Wall-clock ms, for idle motion only. */
  nowMs: number;
}

function paddleTexture(): THREE.CanvasTexture {
  // a swatter's mesh: white = solid, black = hole
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff"; g.fillRect(0, 0, 256, 256);
  g.fillStyle = "#000";
  const step = 19, dy = step * Math.sqrt(3) / 2;
  for (let row = 0; row * dy < 256; row++) for (let x = (row & 1) * step / 2; x < 256; x += step) {
    if (Math.hypot(x - 128, row * dy - 128) < 108) { g.beginPath(); g.arc(x, row * dy, 5.2, 0, 7); g.fill(); }
  }
  return new THREE.CanvasTexture(c);
}

function gridTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 1200; c.height = 800;
  const g = c.getContext("2d")!;
  g.fillStyle = "#1b2d32"; g.fillRect(0, 0, 1200, 800);
  g.strokeStyle = "#2a4248"; g.lineWidth = 2;
  for (let x = 0; x <= 1200; x += 100) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 800); g.stroke(); }
  for (let y = 0; y <= 800; y += 100) { g.beginPath(); g.moveTo(0, y); g.lineTo(1200, y); g.stroke(); }
  g.strokeStyle = "#3b5a61"; g.lineWidth = 6; g.strokeRect(3, 3, 1194, 794);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

export class Stage {
  readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private cam: THREE.PerspectiveCamera;
  private eyeCam = new THREE.PerspectiveCamera(105, 1.4, 0.2, 2000);
  private fly = new FlyModel();
  private flyHolder = new THREE.Group();
  private marker: THREE.Mesh;
  private trail: THREE.Points;
  private trailPos = new Float32Array(TRAIL_POINTS * 3);
  private swatter = new THREE.Group();
  private footprint = new THREE.Group();
  private charge: THREE.Mesh[] = [];
  private pipRect = { w: 0, h: 0 };
  private width = 0;
  private height = 0;
  private ray = new THREE.Raycaster();
  private plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -CONFIG.hoverHeightMm);

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(COLORS.void);
    this.scene.fog = new THREE.Fog(COLORS.void, 900, 1500);

    this.cam = new THREE.PerspectiveCamera(38, 1.5, 10, 3000);
    this.cam.up.set(0, 0, 1);
    this.cam.position.set(0, -300, 560);
    this.cam.lookAt(0, 8, 0);
    this.cam.updateMatrixWorld();

    // the table
    const { w, h } = CONFIG.arenaMm;
    const table = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: gridTexture(), roughness: 0.95 }));
    table.receiveShadow = true;
    this.scene.add(table);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ color: COLORS.void, roughness: 1 }));
    floor.position.z = -3;
    this.scene.add(floor);

    // light: hemisphere fill plus one key light from nearly overhead, so the swatter's shadow stays under it
    this.scene.add(new THREE.HemisphereLight(0xbfe9ee, 0x10181b, 1.05));
    const key = new THREE.DirectionalLight(0xfff3e0, 2.1);
    key.position.set(80, -120, 520);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -360, right: 360, top: 260, bottom: -260, near: 50, far: 1200 });
    key.shadow.bias = -0.0004;
    key.shadow.radius = 4;
    this.scene.add(key);

    // the swatter: a perforated paddle with a rim and handle
    const r = CONFIG.swatterHalfWidthMm;
    const paddle = new THREE.Mesh(
      new THREE.CircleGeometry(r, 64),
      new THREE.MeshStandardMaterial({ color: 0xd7e6e8, alphaMap: paddleTexture(), alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.5, metalness: 0.1 }),
    );
    paddle.castShadow = true;
    const rim = new THREE.Mesh(new THREE.TorusGeometry(r, 2.2, 8, 72), new THREE.MeshStandardMaterial({ color: 0x2a3b40, roughness: 0.4, metalness: 0.4 }));
    rim.castShadow = true;
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(3, 3.6, 150, 12), new THREE.MeshStandardMaterial({ color: COLORS.spike, roughness: 0.4 }));
    handle.geometry.translate(0, 75, 0);
    handle.position.set(0, r, 0);
    handle.rotation.x = -0.28; // leans up and away from the camera
    handle.castShadow = true;
    this.swatter.add(paddle, rim, handle);
    this.scene.add(this.swatter);

    // the footprint: the exact disc that decides a hit, and the charge ring around it
    const fill = new THREE.Mesh(new THREE.CircleGeometry(r, 64), new THREE.MeshBasicMaterial({ color: COLORS.spike, transparent: true, opacity: 0.1, depthWrite: false }));
    const edge = new THREE.Mesh(new THREE.RingGeometry(r - 0.9, r, 96), new THREE.MeshBasicMaterial({ color: COLORS.spike, transparent: true, opacity: 0.55, depthWrite: false }));
    this.footprint.add(fill, edge);
    const step = (Math.PI * 2) / CHARGE_SEGMENTS;
    for (let i = 0; i < CHARGE_SEGMENTS; i++) {
      const seg = new THREE.Mesh(new THREE.RingGeometry(r + 4, r + 9, 3, 1, Math.PI / 2 - (i + 1) * step + 0.01, step - 0.02), new THREE.MeshBasicMaterial({ color: COLORS.accent, depthWrite: false }));
      seg.visible = false;
      this.charge.push(seg);
      this.footprint.add(seg);
    }
    this.footprint.position.z = 0.4;
    this.scene.add(this.footprint);

    // a soft ring on the table under the fly, so it is easy to find on a big table
    this.marker = new THREE.Mesh(new THREE.RingGeometry(21, 24, 64), new THREE.MeshBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.5, depthWrite: false }));
    this.marker.position.z = 0.3;
    this.scene.add(this.marker);

    // the path the fly took after it left, as pink dots, so a takeoff reads even in a still frame
    const tg = new THREE.BufferGeometry();
    tg.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
    this.trail = new THREE.Points(tg, new THREE.PointsMaterial({ color: COLORS.spike, size: 5, sizeAttenuation: false, transparent: true, opacity: 0.85, depthWrite: false }));
    this.trail.frustumCulled = false;
    this.trail.visible = false;
    this.scene.add(this.trail);

    this.flyHolder.add(this.fly.root);
    this.fly.root.scale.setScalar(FLY_SCALE);
    this.scene.add(this.flyHolder);
  }

  resize(width: number, height: number, pip: { w: number; h: number }): void {
    this.width = width; this.height = height; this.pipRect = pip;
    this.renderer.setSize(width, height, false);
    this.cam.aspect = width / height;
    // keep the same horizontal view at any stage shape, wide enough that a fly at the edge of its spawn area (and the
    // climb after it leaves) stays in frame; the vertical fov follows from the aspect
    const hFov = (56 * Math.PI) / 180;
    this.cam.fov = Math.max(34, Math.min(78, (2 * Math.atan(Math.tan(hFov / 2) / (width / height)) * 180) / Math.PI));
    this.cam.updateProjectionMatrix();
    this.cam.updateMatrixWorld();
    this.eyeCam.aspect = pip.w / Math.max(1, pip.h);
    this.eyeCam.updateProjectionMatrix();
  }

  /** Where the pointer is on the plane the swatter hovers in, in table mm, clamped to the arena. */
  pointerToMm(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -(((clientY - rect.top) / rect.height) * 2 - 1));
    this.ray.setFromCamera(ndc, this.cam);
    const hit = new THREE.Vector3();
    if (!this.ray.ray.intersectPlane(this.plane, hit)) return { x: 0, y: 0 };
    const { w, h } = CONFIG.arenaMm;
    return { x: Math.max(-w / 2, Math.min(w / 2, hit.x)), y: Math.max(-h / 2, Math.min(h / 2, hit.y)) };
  }

  /** Screen position (CSS px inside the stage) of a table point, for labels that follow the swatter. */
  project(x: number, y: number, z: number): { x: number; y: number } {
    const v = new THREE.Vector3(x, y, z).project(this.cam);
    return { x: (v.x * 0.5 + 0.5) * this.width, y: (-v.y * 0.5 + 0.5) * this.height };
  }

  render(s: StageState): void {
    const { round } = s;
    const t = round.timeMs + s.afterMs;
    const sw = round.swatter;
    const card = round.card;

    // swatter
    // after the swat lands the paddle lifts away, so the result under it can be seen
    const peel = card !== null && card.strikeRvMs !== null ? Math.min(1, s.afterMs / 450) * 38 : 0;
    this.swatter.position.set(sw.x, sw.y, sw.z + 1.5 + peel);
    this.footprint.position.set(sw.x, sw.y, 0.4);
    const n = Math.round(round.charge * CHARGE_SEGMENTS);
    this.charge.forEach((seg, i) => { seg.visible = i < n; });
    const striking = round.phase === "strike" || (card !== null && card.strikeRvMs !== null);
    ((this.footprint.children[0] as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = striking ? 0.2 : 0.1;

    // fly
    const leave = round.leftAtMs;
    const left = leave !== null && t >= leave;
    const since = left ? t - leave! : 0;
    const f = round.fly;
    const lateral = s.headingSide === "left" ? 1 : s.headingSide === "right" ? -1 : 0; // model left is +y in this frame
    this.flyHolder.position.set(f.x, f.y, 0);
    this.flyHolder.rotation.set(0, 0, f.heading);
    this.fly.body.position.set(0, 0, 0);
    this.fly.body.rotation.set(0, 0, 0);
    this.fly.body.scale.set(1, 1, 1);
    this.flyHolder.visible = true;

    const squashed = card !== null && card.outcome === "hit";
    let crouch = s.crouch, wingOpen = 0, flap = 0;
    if (squashed) {
      this.fly.body.scale.set(1.35, 1.35, 0.12);
      crouch = 1;
    } else if (left) {
      // the jump: a push-off along the side the spikes chose (or straight up when they chose none), wings out, away
      const dir = new THREE.Vector3(0, lateral * 0.55, 0.85).normalize();
      // climb at the real push-off speed, then hover and bob in the air: the fly never sits on the table once it has left
      const reach = (ms: number) => Math.min(ms * FLIGHT_MMS, CLIMB_MM) + (ms * FLIGHT_MMS > CLIMB_MM ? 6 * Math.sin(ms / 70) : 0);
      const dist = reach(since) / FLY_SCALE;
      this.fly.body.position.set(dir.x * dist, dir.y * dist, dir.z * dist);
      this.fly.body.rotation.x = lateral * -0.5 * Math.min(1, since / 40); // roll into the turn
      wingOpen = 1;
      flap = since * 0.34;
      // trail: where the fly has been, in table mm
      const ch = Math.cos(f.heading), sh = Math.sin(f.heading);
      for (let i = 0; i < TRAIL_POINTS; i++) {
        const d = reach((since * i) / (TRAIL_POINTS - 1)), ox = dir.x * d, oy = dir.y * d, oz = dir.z * d + 13;
        this.trailPos[3 * i] = f.x + ch * ox - sh * oy;
        this.trailPos[3 * i + 1] = f.y + sh * ox + ch * oy;
        this.trailPos[3 * i + 2] = oz;
      }
      (this.trail.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      this.trail.visible = since > 6;
      crouch = Math.max(0, 1 - since / 12);
    } else {
      crouch = Math.max(crouch, 0);
      const breathe = 1 + 0.012 * Math.sin(s.nowMs / 420);
      this.fly.body.scale.set(1, 1, breathe);
    }
    if (!left) this.trail.visible = false;
    this.fly.pose({ crouch, wingOpen, flap, groom: s.nowMs / 260 * (Math.sin(s.nowMs / 2300) > 0.55 ? 1 : 0) });

    this.marker.position.set(f.x, f.y, 0.3);
    this.marker.visible = !left && !squashed;
    (this.marker.material as THREE.MeshBasicMaterial).opacity = 0.35 + 0.2 * Math.sin(s.nowMs / 380);

    // main view
    const W = this.width, H = this.height;
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, W, H);
    this.renderer.render(this.scene, this.cam);

    // the fly's-eye view, aimed at the swatter, drawn in the corner
    if (this.pipRect.w > 0) {
      this.eyeCam.position.set(f.x, f.y, CONFIG.eyeHeightMm + 1.5);
      this.eyeCam.up.set(0, 0, 1);
      this.eyeCam.lookAt(sw.x, sw.y, sw.z);
      this.eyeCam.updateMatrixWorld();
      const hidden = this.flyHolder.visible;
      this.flyHolder.visible = false; // the camera sits inside the (enlarged) body
      const { w, h } = this.pipRect;
      this.renderer.setScissorTest(true);
      // GL's origin is the bottom-left corner: y = 14 puts the inset 14 px above the stage's bottom edge, where the CSS box is
      this.renderer.setViewport(W - w - 14, 14, w, h);
      this.renderer.setScissor(W - w - 14, 14, w, h);
      this.renderer.render(this.scene, this.eyeCam);
      this.renderer.setScissorTest(false);
      this.flyHolder.visible = hidden;
    }
  }
}
