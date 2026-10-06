import * as THREE from "three";
import { ChipStacks } from './ChipStacks';
import type { ChipTransfer } from '../game/chips';


import type { SceneQuality, TablePhase, TableView } from "../game/model";


const SCENE_COLORS = {
  background: 0x11140d,
  floor: 0x26291a,
  wall: 0x1b2013,
  ink: 0x1b1b11,
  olive: 0x48512d,
  felt: 0x12422f,
  parchment: 0xe4d6ad,
  amber: 0xd28b32,
  signal: 0x91c54a,
  red: 0x9d2e25,
} as const;

const WING_SPEED_BY_PHASE: Record<TablePhase, number> = {
  dealing: 24,
  "fly-thinking": 62,
  "player-turn": 18,
  showdown: 12,
};

export class CasinoScene {
  readonly #host: HTMLElement;
  readonly #chips: ChipStacks;
  readonly #renderer: THREE.WebGLRenderer;
  readonly #scene = new THREE.Scene();
  readonly #camera = new THREE.PerspectiveCamera(38, 1, 0.1, 80);
  readonly #clock = new THREE.Clock();
  readonly #fly = new THREE.Group();
  readonly #wingPivots: THREE.Group[] = [];
  #wingSpeed: number;
  readonly #resize = () => this.resize();
  readonly #observer = new ResizeObserver(this.#resize);
  #frame = 0;

  constructor(host: HTMLElement, view: TableView, quality: SceneQuality) {
    this.#host = host;
    this.#wingSpeed = WING_SPEED_BY_PHASE[view.phase];
    this.#renderer = new THREE.WebGLRenderer({ antialias: true });
    this.#renderer.setPixelRatio(quality.pixelRatio);
    this.#renderer.shadowMap.enabled = quality.shadows;
    this.#renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.#renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.#renderer.toneMappingExposure = 1;
    host.append(this.#renderer.domElement);

    this.#scene.background = new THREE.Color(SCENE_COLORS.background);
    this.#scene.fog = new THREE.FogExp2(SCENE_COLORS.background, 0.038);
    this.#camera.position.set(0, 6.8, 9);
    this.#camera.lookAt(0, 1.5, 0);

    this.addRoom();
    this.addTable(view);
    this.addCasinoDetails();
    this.addFly();
    this.#chips = new ChipStacks(this.#scene,view);
    this.resize();
    this.#observer.observe(host);
    this.tick();
  }

  update(view: TableView): void { this.#wingSpeed = WING_SPEED_BY_PHASE[view.phase]; this.#chips.update(view); }
  animate(stages: readonly (readonly ChipTransfer[])[]): Promise<boolean> { if(stages.length) this.#wingSpeed = WING_SPEED_BY_PHASE['fly-thinking']; return this.#chips.animate(stages); }
  cancelAnimation(): void { this.#chips.cancel(); }
  destroy(): void {
    cancelAnimationFrame(this.#frame);
    this.#observer.disconnect();
    this.#chips.destroy();
    this.#scene.traverse(part => {
      if (part instanceof THREE.Mesh) {
        part.geometry.dispose();
        const materials = Array.isArray(part.material) ? part.material : [part.material];
        materials.forEach(material => {
          if (material instanceof THREE.MeshStandardMaterial) material.map?.dispose();
          material.dispose();
        });
      }
    });
    this.#renderer.dispose();
    this.#host.replaceChildren();
  }

  private addCasinoDetails(): void {
    const brass = new THREE.MeshStandardMaterial({ color: 0xb88a41, metalness: .72, roughness: .36 });
    const walnut = new THREE.MeshStandardMaterial({ color: 0x342719, roughness: .68 });
    const leather = new THREE.MeshStandardMaterial({ color: 0x442d25, roughness: .8 });
    const warm = new THREE.MeshStandardMaterial({ color: 0xffcf75, emissive: 0xf6a92e, emissiveIntensity: .65 });
    const box = (width: number, height: number, depth: number, x: number, y: number, z: number, material: THREE.MeshStandardMaterial) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
      mesh.position.set(x, y, z); mesh.castShadow = true; mesh.receiveShadow = true;
      this.#scene.add(mesh); return mesh;
    };
    box(28, 1.5, .25, 0, .75, -6.8, walnut);
    box(28, .05, .35, 0, 1.55, -6.6, brass);
    for (const x of [-8, -5, 5, 8]) {
      box(.1, 4, .1, x, 3.4, -6.6, brass);
      box(1.8, .07, .1, x, 5.4, -6.6, brass);
      box(1.8, .07, .1, x, 1.6, -6.6, brass);
      for (const edge of [-.9, .9]) box(.05, 3.8, .08, x + edge, 3.5, -6.6, brass);
      const shade = new THREE.Mesh(new THREE.CylinderGeometry(.35,.52,.62,24,1,true), warm);
      shade.position.set(x,3.75,-5.95); this.#scene.add(shade);
      box(.055,.9,.055,x,3.2,-6.15,brass);
      const bulb = new THREE.PointLight(0xffb84f,8,6,2); bulb.position.set(x,3.5,-5.5); this.#scene.add(bulb);
    }
    for (const side of [-1]) {
      const z = side * 3.4;
      const seat = new THREE.Mesh(new THREE.CylinderGeometry(.7,.7,.3,32),leather);
      seat.position.set(0,1.25,z); seat.castShadow = true; this.#scene.add(seat);
      box(1.35,1.5,.2,0,2,z + side * .55,leather);
      for (const x of [-.45,.45]) {
        for (const offset of [-.4,.4]) box(.09,1.2,.09,x,.55,z + offset,brass);
      }
      box(1.4,.06,.22,0,2.77,z + side * .55,brass);
    }
    const lightRing = new THREE.Mesh(new THREE.TorusGeometry(1.7,.065,8,64),brass);
    lightRing.rotation.x = Math.PI/2; lightRing.position.set(0,5.7,0); this.#scene.add(lightRing);
    for (let i=0;i<8;i++) {
      const angle = i*Math.PI/4;
      const bulb = new THREE.Mesh(new THREE.SphereGeometry(.095,12,8),warm);
      bulb.position.set(Math.cos(angle)*1.7,5.68,Math.sin(angle)*1.7); this.#scene.add(bulb);
    }
    const railInlay = new THREE.Mesh(new THREE.TorusGeometry(3.55,.012,8,96),brass);
    railInlay.rotation.x = Math.PI/2; railInlay.scale.y=.64; railInlay.position.y=1.685; this.#scene.add(railInlay);
    for (const x of [-2.9,2.9]) {
      const cup = new THREE.Mesh(new THREE.CylinderGeometry(.14,.14,.015,24),brass);
      cup.position.set(x,1.97,.95); this.#scene.add(cup);
      const center = new THREE.Mesh(new THREE.CylinderGeometry(.11,.11,.02,24),walnut);
      center.position.set(x,1.975,.95); this.#scene.add(center);
    }
    const cloth = document.createElement('canvas'); cloth.width=256; cloth.height=256;
    const ctx = cloth.getContext('2d');
    if (ctx) {
      ctx.fillStyle='#123b2a';ctx.fillRect(0,0,256,256);
      for(let x=0;x<256;x+=4)for(let y=0;y<256;y+=4){ctx.fillStyle=(x+y)%8?'#183f2d':'#1c4933';ctx.fillRect(x,y,1,1);}
      const texture = new THREE.CanvasTexture(cloth); texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(8,8);
      this.#scene.traverse(part => {if(part instanceof THREE.Mesh && part.geometry instanceof THREE.CylinderGeometry && part.geometry.parameters.radiusTop===3.85 && part.material instanceof THREE.MeshStandardMaterial){part.material.map=texture;part.material.needsUpdate=true;}});
    }
  }
  private addRoom(): void {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ color: SCENE_COLORS.floor, roughness: 1 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.#scene.add(floor);

    const wallMaterial = new THREE.MeshStandardMaterial({ color: SCENE_COLORS.wall, roughness: 1 });
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(28, 12), wallMaterial);
    wall.position.set(0, 5, -7);
    this.#scene.add(wall);

    const trimMaterial = new THREE.MeshStandardMaterial({ color: SCENE_COLORS.amber, roughness: 0.68 });
    for (const x of [-7, -3.5, 0, 3.5, 7]) {
      const trim = new THREE.Mesh(new THREE.BoxGeometry(0.045, 8, 0.08), trimMaterial);
      trim.position.set(x, 4, -6.9);
      this.#scene.add(trim);
    }

    this.#scene.add(new THREE.HemisphereLight(0xd6c68f, 0x15190d, 1.6));
    const key = new THREE.SpotLight(0xf0c46e, 95, 30, 0.62, 0.7, 1.3);
    key.position.set(0, 9, 2);
    key.target.position.set(0, 0, 0);
    key.castShadow = true;
    this.#scene.add(key, key.target);
    const signal = new THREE.PointLight(SCENE_COLORS.signal, 14, 12);
    signal.position.set(-6, 3, -2);
    this.#scene.add(signal);
  }

  private addTable(_view: TableView): void {
    const pedestal = new THREE.Mesh(
      new THREE.CylinderGeometry(1.3, 1.7, 2.2, 48),
      new THREE.MeshStandardMaterial({ color: SCENE_COLORS.ink, roughness: 0.85 }),
    );
    pedestal.position.y = 0.4;
    pedestal.castShadow = true;
    this.#scene.add(pedestal);

    const felt = new THREE.Mesh(
      new THREE.CylinderGeometry(3.85, 3.85, 0.22, 64),
      new THREE.MeshStandardMaterial({ color: SCENE_COLORS.felt, roughness: 0.95 }),
    );
    felt.scale.z = 0.64;
    felt.position.y = 1.55;
    felt.receiveShadow = true;
    felt.castShadow = true;
    this.#scene.add(felt);

    const rail = new THREE.Mesh(
      new THREE.TorusGeometry(3.85, 0.28, 16, 64),
      new THREE.MeshStandardMaterial({ color: SCENE_COLORS.ink, roughness: 0.8 }),
    );
    rail.rotation.x = Math.PI / 2;
    rail.scale.y = 0.64;
    rail.position.y = 1.72;
    rail.castShadow = true;
    this.#scene.add(rail);
  }

  private addFly(): void {
    const amber = new THREE.MeshStandardMaterial({ color: SCENE_COLORS.amber, roughness: 0.78 });
    const dark = new THREE.MeshStandardMaterial({ color: SCENE_COLORS.ink, roughness: 0.9 });
    const red = new THREE.MeshStandardMaterial({ color: SCENE_COLORS.red, roughness: 0.48 });
    const wingMaterial = new THREE.MeshStandardMaterial({
      color: 0xd7dbb4,
      transparent: true,
      opacity: 0.38,
      side: THREE.DoubleSide,
      depthWrite: false,
      roughness: 0.5,
    });
    const legGeometry = new THREE.CylinderGeometry(0.018, 0.026, 0.38, 8);

    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.36, 24, 16), amber);
    abdomen.scale.set(0.72, 0.68, 1.35);
    abdomen.position.z = -0.28;
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.34, 24, 16), amber);
    thorax.scale.set(0.92, 0.84, 1.05);
    thorax.position.z = 0.15;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.28, 24, 16), amber);
    head.scale.set(1.08, 0.92, 0.95);
    head.position.z = 0.57;
    this.#fly.add(abdomen, thorax, head);

    for (const z of [-0.52, -0.3, -0.08]) {
      const stripe = new THREE.Mesh(new THREE.TorusGeometry(0.265, 0.035, 8, 24), dark);
      stripe.scale.y = 0.82;
      stripe.position.z = z;
      this.#fly.add(stripe);
    }

    for (const side of [-1, 1]) {
      const flyEye = new THREE.Mesh(new THREE.SphereGeometry(0.17, 18, 12), red);
      flyEye.scale.set(0.7, 1.05, 0.75);
      flyEye.position.set(side * 0.21, 0.03, 0.72);
      this.#fly.add(flyEye);

      const antenna = new THREE.Mesh(legGeometry, dark);
      antenna.scale.set(0.65, 0.72, 0.65);
      antenna.position.set(side * 0.17, 0.24, 0.82);
      antenna.rotation.z = side * -0.52;
      antenna.rotation.x = -0.45;
      const antennaTip = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), dark);
      antennaTip.position.set(side * 0.28, 0.35, 0.94);
      this.#fly.add(antenna, antennaTip);

      [-0.16, 0.1, 0.34].forEach((z, legIndex) => {
        const upper = new THREE.Mesh(legGeometry, dark);
        upper.position.set(side * 0.25, -0.17, z);
        upper.rotation.z = side * -0.72;
        upper.rotation.x = (legIndex - 1) * 0.32;
        const knee = new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 8), dark);
        knee.position.set(side * 0.39, -0.31, z + (legIndex - 1) * 0.08);
        const lower = new THREE.Mesh(legGeometry, dark);
        lower.position.set(side * 0.53, -0.44, z + (legIndex - 1) * 0.14);
        lower.rotation.z = side * -0.82;
        lower.rotation.x = (legIndex - 1) * 0.48;
        this.#fly.add(upper, knee, lower);
      });

      const wingPivot = new THREE.Group();
      wingPivot.position.set(side * 0.22, 0.2, -0.02);
      const wing = new THREE.Mesh(
        new THREE.CircleGeometry(0.48, 28),
        wingMaterial,
      );
      wing.scale.set(0.58, 1.35, 1);
      wing.position.x = side * 0.35;
      wing.rotation.x = -Math.PI / 2;
      wing.rotation.z = side * -0.42;
      wingPivot.add(wing);
      this.#wingPivots.push(wingPivot);
      this.#fly.add(wingPivot);
    }

    this.#fly.traverse((part) => {
      if (part instanceof THREE.Mesh && part.material !== wingMaterial) {
        part.castShadow = true;
      }
    });
    this.#fly.position.set(0, 2.1, -3.4);
    this.#fly.rotation.x = -0.18;
    this.#fly.scale.setScalar(1.12);
    this.#scene.add(this.#fly);
  }

  private resize(): void {
    const width = this.#host.clientWidth;
    const height = this.#host.clientHeight;
    this.#camera.aspect = width / Math.max(1, height);
    this.#camera.fov = width < 700 ? 65 : 38;
    this.#camera.position.set(0, width < 700 ? 10 : 6.8, width < 700 ? 12 : 9);
    this.#camera.lookAt(0, 1.5, 0);
    this.#camera.updateProjectionMatrix();
    this.#renderer.setSize(width, height, false);
  }

  private tick = (): void => {
    const time = this.#clock.getElapsedTime();
    this.#wingPivots.forEach((pivot, index) => {
      pivot.rotation.z = Math.sin(time * this.#wingSpeed) * 0.28 * (index === 0 ? -1 : 1);
    });
    this.#renderer.render(this.#scene, this.#camera);
    this.#frame = requestAnimationFrame(this.tick);
  };
}









