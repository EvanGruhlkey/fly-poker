import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { Card, SceneQuality, TablePhase, TableView } from "../game/model";

const CARD_RED = new Set(["D", "H"]);
const SCENE_COLORS = {
  background: 0x11140d,
  floor: 0x26291a,
  wall: 0x1b2013,
  ink: 0x1b1b11,
  olive: 0x48512d,
  felt: 0x344729,
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
  readonly #renderer: THREE.WebGLRenderer;
  readonly #scene = new THREE.Scene();
  readonly #camera = new THREE.PerspectiveCamera(38, 1, 0.1, 80);
  readonly #clock = new THREE.Clock();
  readonly #fly = new THREE.Group();
  readonly #wingPivots: THREE.Group[] = [];
  readonly #wingSpeed: number;
  readonly #resize = () => this.resize();
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
    this.#camera.position.set(0, 5.5, 9.6);
    this.#camera.lookAt(0, 0.6, 0);

    const controls = new OrbitControls(this.#camera, this.#renderer.domElement);
    controls.target.set(0, 0.7, 0);
    controls.enablePan = false;
    controls.minDistance = 6;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.update();

    this.addRoom();
    this.addTable(view);
    this.addFly();
    this.resize();
    window.addEventListener("resize", this.#resize);
    this.tick();
  }

  destroy(): void {
    cancelAnimationFrame(this.#frame);
    window.removeEventListener("resize", this.#resize);
    this.#renderer.dispose();
    this.#host.replaceChildren();
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

  private addTable(view: TableView): void {
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

    view.board.forEach((card, index) => this.addCard(card, (index - 1) * 0.72, 0.2));
    view.playerCards.forEach((card, index) => this.addCard(card, (index - 0.5) * 0.58, 2.55));
    this.addChips(0, -0.55, 10, SCENE_COLORS.amber);
    this.addChips(0.75, 2.05, 7, SCENE_COLORS.signal);
  }

  private addCard(card: Card, x: number, z: number): void {
    const canvas = document.createElement("canvas");
    canvas.width = 180;
    canvas.height = 250;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = "#e4d6ad";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#665f3a";
    context.lineWidth = 8;
    context.strokeRect(8, 8, canvas.width - 16, canvas.height - 16);
    context.fillStyle = CARD_RED.has(card[1]) ? "#9d2e25" : "#1b1b11";
    context.font = "bold 68px Georgia";
    context.fillText(card, 22, 86);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.56, 0.035, 0.78),
      [
        new THREE.MeshStandardMaterial({ color: SCENE_COLORS.parchment, roughness: 0.9 }),
        new THREE.MeshStandardMaterial({ color: SCENE_COLORS.parchment, roughness: 0.9 }),
        new THREE.MeshStandardMaterial({ map: texture, roughness: 0.9 }),
        new THREE.MeshStandardMaterial({ color: SCENE_COLORS.olive, roughness: 0.9 }),
        new THREE.MeshStandardMaterial({ color: SCENE_COLORS.parchment, roughness: 0.9 }),
        new THREE.MeshStandardMaterial({ color: SCENE_COLORS.parchment, roughness: 0.9 }),
      ],
    );
    mesh.position.set(x, 1.82, z);
    mesh.castShadow = true;
    this.#scene.add(mesh);
  }

  private addChips(x: number, z: number, count: number, color: number): void {
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.82 });
    for (let index = 0; index < count; index += 1) {
      const chip = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.045, 24), material);
      chip.position.set(x, 1.82 + index * 0.045, z);
      chip.castShadow = true;
      this.#scene.add(chip);
    }
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
    this.#fly.position.set(-2, 2.65, -2.3);
    this.#fly.rotation.x = -0.18;
    this.#fly.scale.setScalar(1.12);
    this.#scene.add(this.#fly);
  }

  private resize(): void {
    const width = this.#host.clientWidth;
    const height = this.#host.clientHeight;
    this.#camera.aspect = width / Math.max(1, height);
    this.#camera.updateProjectionMatrix();
    this.#renderer.setSize(width, height, false);
  }

  private tick = (): void => {
    const time = this.#clock.getElapsedTime();
    this.#fly.position.y = 2.65 + Math.sin(time * 2) * 0.08;
    this.#wingPivots.forEach((pivot, index) => {
      pivot.rotation.z = Math.sin(time * this.#wingSpeed) * 0.28 * (index === 0 ? -1 : 1);
    });
    this.#renderer.render(this.#scene, this.#camera);
    this.#frame = requestAnimationFrame(this.tick);
  };
}
