// A procedural fruit fly for the 3D stage, built from primitives so no mesh asset is needed.
//
// Modelled in millimetres in a right-handed frame: x forward, y to the fly's LEFT, z up. (The game's own body frame
// in world.ts puts the fly's right on +y and is left-handed once drawn; the stage maps that to -y here.)
// This is an illustration. The only things that move it are values the engine already produced: the giant-fiber
// voltage (the crouch) and the takeoff the spikes decided. It never decides anything.

import * as THREE from "three";

export interface Pose {
  /** 0 to 1: how far the legs have folded for the push-off. Driven by the giant fiber's membrane potential. */
  crouch: number;
  /** 0 folded over the abdomen, 1 fully raised. */
  wingOpen: number;
  /** Radians. Wing beat phase while flying. */
  flap: number;
  /** Radians. Small idle motion of the front legs. */
  groom: number;
}

const BODY = 0xb9772f, DARK = 0x4a2c14, EYE = 0xc0213a, LEG = 0x8a5424;

function stripes(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 8; c.height = 128;
  const g = c.getContext("2d")!;
  for (let i = 0; i < 8; i++) { g.fillStyle = i % 2 ? "#3b2410" : "#c98a3c"; g.fillRect(0, i * 16, 8, 16); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function segment(len: number, r: number, color: number): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(r * 0.7, r, len, 6);
  geo.translate(0, len / 2, 0); // grows along local +y from its joint
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.7 }));
}

interface Leg { yaw: THREE.Group; hip: THREE.Group; knee: THREE.Group; ankle: THREE.Group; side: 1 | -1; baseYaw: number; front: boolean }

export class FlyModel {
  readonly root = new THREE.Group();
  /** Everything that tilts with the body in flight. */
  readonly body = new THREE.Group();
  private legs: Leg[] = [];
  private wings: Array<{ pivot: THREE.Group; side: 1 | -1 }> = [];
  private meshes: THREE.Mesh[] = [];

  constructor() {
    this.root.add(this.body);
    const mat = (color: number, rough = 0.6) => new THREE.MeshStandardMaterial({ color, roughness: rough });
    const add = (m: THREE.Mesh) => { this.body.add(m); this.meshes.push(m); m.castShadow = true; return m; };

    const thorax = add(new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), mat(BODY)));
    thorax.scale.set(0.62, 0.46, 0.46); thorax.position.set(0, 0, 1.15);
    const abdomen = add(new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshStandardMaterial({ map: stripes(), roughness: 0.6 })));
    abdomen.rotation.z = Math.PI / 2; // the texture's bands run along the body
    abdomen.scale.set(0.4, 0.95, 0.4); abdomen.position.set(-1.1, 0, 1.0); abdomen.rotation.y = 0;
    const head = add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), mat(DARK)));
    head.scale.set(0.34, 0.34, 0.32); head.position.set(0.72, 0, 1.08);
    for (const s of [1, -1]) {
      const eye = add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshStandardMaterial({ color: EYE, roughness: 0.3, emissive: 0x3a0510 })));
      eye.scale.set(0.27, 0.22, 0.27); eye.position.set(0.76, s * 0.27, 1.14);
    }

    // six legs: front, middle, hind on each side
    const hips: Array<[number, number, boolean]> = [[0.42, 0.55, true], [0.05, 0, false], [-0.35, -0.6, false]];
    for (const side of [1, -1] as const) {
      for (const [x, baseYaw, front] of hips) {
        const yaw = new THREE.Group(); yaw.position.set(x, side * 0.28, 0.98);
        const hip = new THREE.Group(); yaw.add(hip);
        const femur = segment(0.95, 0.07, LEG); hip.add(femur); this.meshes.push(femur);
        const knee = new THREE.Group(); knee.position.y = 0.95; hip.add(knee);
        const tibia = segment(1.05, 0.05, LEG); knee.add(tibia); this.meshes.push(tibia);
        const ankle = new THREE.Group(); ankle.position.y = 1.05; knee.add(ankle);
        const tarsus = segment(0.45, 0.035, DARK); ankle.add(tarsus); this.meshes.push(tarsus);
        this.body.add(yaw);
        this.legs.push({ yaw, hip, knee, ankle, side, baseYaw: baseYaw * side * -1, front });
      }
    }

    // wings: translucent ellipses pivoting at the top of the thorax
    const shape = new THREE.Shape();
    shape.absellipse(1.1, 0, 1.2, 0.46, 0, Math.PI * 2, false, 0); // extends along +x from the pivot
    const wingMat = new THREE.MeshStandardMaterial({ color: 0xcfe9f0, transparent: true, opacity: 0.42, side: THREE.DoubleSide, roughness: 0.2 });
    for (const side of [1, -1] as const) {
      const pivot = new THREE.Group(); pivot.position.set(0.05, side * 0.18, 1.55);
      const w = new THREE.Mesh(new THREE.ShapeGeometry(shape, 24), wingMat);
      pivot.add(w); this.body.add(pivot);
      this.wings.push({ pivot, side });
    }
    this.pose({ crouch: 0, wingOpen: 0, flap: 0, groom: 0 });
  }

  /** Every mesh, so the stage can tint the whole fly or hide it for the fly-cam pass. */
  get parts(): readonly THREE.Mesh[] { return this.meshes; }

  pose(p: Pose): void {
    const crouch = Math.max(0, Math.min(1, p.crouch));
    for (const l of this.legs) {
      const spread = 0.55 + 0.35 * crouch; // legs splay and fold as the fly loads up
      const pitch = l.side > 0 ? -spread : Math.PI + spread;
      l.yaw.rotation.z = l.baseYaw * 0.5;
      l.hip.rotation.x = pitch;
      const bend = (0.95 + 0.7 * crouch) * (l.side > 0 ? -1 : 1);
      l.knee.rotation.x = bend;
      l.ankle.rotation.x = -bend * 0.45;
      if (l.front) l.hip.rotation.z = Math.sin(p.groom) * 0.35 * (1 - crouch);
    }
    for (const w of this.wings) {
      const open = Math.max(0, Math.min(1, p.wingOpen));
      // folded: lying back over the abdomen; open: swept out and up, then beating
      w.pivot.rotation.z = w.side * (Math.PI + 0.15 - open * 1.1);
      w.pivot.rotation.x = w.side * (-0.05 + open * (0.9 + 0.45 * Math.sin(p.flap)));
    }
  }
}
