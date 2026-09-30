import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { Card, SceneQuality, TableView } from "../game/model";

const CARD_RED = new Set(["D", "H"]);

export class CasinoScene {
  readonly #host: HTMLElement;
  readonly #renderer: THREE.WebGLRenderer;
  readonly #scene = new THREE.Scene();
  readonly #camera = new THREE.PerspectiveCamera(38, 1, 0.1, 80);
  readonly #clock = new THREE.Clock();
  readonly #fly = new THREE.Group();
  readonly #wings: THREE.Mesh[] = [];
  readonly #resize = () => this.resize();
  #frame = 0;

  constructor(host: HTMLElement, view: TableView, quality: SceneQuality) {
    this.#host = host;
    this.#renderer = new THREE.WebGLRenderer({ antialias: true });
    this.#renderer.setPixelRatio(quality.pixelRatio);
    this.#renderer.shadowMap.enabled = quality.shadows;
    this.#renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.#renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.#renderer.toneMappingExposure = 1.15;
    host.append(this.#renderer.domElement);

    this.#scene.background = new THREE.Color(0x09070a);
    this.#scene.fog = new THREE.FogExp2(0x09070a, 0.045);
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
      new THREE.MeshStandardMaterial({ color: 0x150f12, roughness: 0.82 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.#scene.add(floor);

    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0x241217, roughness: 0.72 });
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(28, 12), wallMaterial);
    wall.position.set(0, 5, -7);
    this.#scene.add(wall);

    const gold = new THREE.MeshStandardMaterial({ color: 0xb88736, metalness: 0.7, roughness: 0.28 });
    for (const x of [-7, -3.5, 0, 3.5, 7]) {
      const trim = new THREE.Mesh(new THREE.BoxGeometry(0.05, 8, 0.08), gold);
      trim.position.set(x, 4, -6.9);
      this.#scene.add(trim);
    }

    this.#scene.add(new THREE.HemisphereLight(0x7b6384, 0x170d0a, 1.2));
    const key = new THREE.SpotLight(0xffc66b, 130, 30, 0.62, 0.6, 1.2);
    key.position.set(0, 9, 2);
    key.target.position.set(0, 0, 0);
    key.castShadow = true;
    this.#scene.add(key, key.target);
    const red = new THREE.PointLight(0xa8323f, 25, 15);
    red.position.set(-6, 3, -2);
    this.#scene.add(red);
  }

  private addTable(view: TableView): void {
    const pedestal = new THREE.Mesh(
      new THREE.CylinderGeometry(1.3, 1.7, 2.2, 48),
      new THREE.MeshStandardMaterial({ color: 0x251615, roughness: 0.4 }),
    );
    pedestal.position.y = 0.4;
    pedestal.castShadow = true;
    this.#scene.add(pedestal);

    const felt = new THREE.Mesh(
      new THREE.CylinderGeometry(3.85, 3.85, 0.22, 64),
      new THREE.MeshStandardMaterial({ color: 0x0f5b3c, roughness: 0.72 }),
    );
    felt.scale.z = 0.64;
    felt.position.y = 1.55;
    felt.receiveShadow = true;
    felt.castShadow = true;
    this.#scene.add(felt);

    const rail = new THREE.Mesh(
      new THREE.TorusGeometry(3.85, 0.28, 16, 64),
      new THREE.MeshStandardMaterial({ color: 0x3b211b, roughness: 0.38 }),
    );
    rail.rotation.x = Math.PI / 2;
    rail.scale.y = 0.64;
    rail.position.y = 1.72;
    rail.castShadow = true;
    this.#scene.add(rail);

    view.board.forEach((card, index) => this.addCard(card, (index - 1) * 0.72, 0.2));
    view.playerCards.forEach((card, index) => this.addCard(card, (index - 0.5) * 0.58, 2.55));
    this.addChips(0, -0.55, 10, 0xb83842);
    this.addChips(0.75, 2.05, 7, 0x3153a4);
  }

  private addCard(card: Card, x: number, z: number): void {
    const canvas = document.createElement("canvas");
    canvas.width = 180;
    canvas.height = 250;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = "#f5f0e6";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = CARD_RED.has(card[1]) ? "#b52d39" : "#15131a";
    context.font = "bold 72px Georgia";
    context.fillText(card, 22, 86);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.56, 0.035, 0.78),
      [
        new THREE.MeshStandardMaterial({ color: 0xe9dfcb }),
        new THREE.MeshStandardMaterial({ color: 0xe9dfcb }),
        new THREE.MeshStandardMaterial({ map: texture }),
        new THREE.MeshStandardMaterial({ color: 0x501824 }),
        new THREE.MeshStandardMaterial({ color: 0xe9dfcb }),
        new THREE.MeshStandardMaterial({ color: 0xe9dfcb }),
      ],
    );
    mesh.position.set(x, 1.82, z);
    mesh.castShadow = true;
    this.#scene.add(mesh);
  }

  private addChips(x: number, z: number, count: number, color: number): void {
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.35 });
    for (let index = 0; index < count; index += 1) {
      const chip = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.045, 24), material);
      chip.position.set(x, 1.82 + index * 0.045, z);
      chip.castShadow = true;
      this.#scene.add(chip);
    }
  }

  private addFly(): void {
    const brown = new THREE.MeshStandardMaterial({ color: 0x8e5f31, roughness: 0.62 });
    const eye = new THREE.MeshStandardMaterial({ color: 0x8b1119, roughness: 0.2 });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.38, 24, 16), brown);
    body.scale.set(0.75, 0.72, 1.4);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 16), brown);
    head.position.z = 0.52;
    this.#fly.add(body, head);
    for (const side of [-1, 1]) {
      const flyEye = new THREE.Mesh(new THREE.SphereGeometry(0.17, 18, 12), eye);
      flyEye.position.set(side * 0.21, 0.04, 0.72);
      this.#fly.add(flyEye);
      const wing = new THREE.Mesh(
        new THREE.CircleGeometry(0.48, 28),
        new THREE.MeshPhysicalMaterial({ color: 0xc9e5df, transparent: true, opacity: 0.45, side: THREE.DoubleSide }),
      );
      wing.scale.set(0.55, 1.35, 1);
      wing.position.set(side * 0.45, 0.18, -0.15);
      wing.rotation.y = side * 0.55;
      this.#wings.push(wing);
      this.#fly.add(wing);
    }
    this.#fly.position.set(-1.15, 2.5, -2.35);
    this.#fly.rotation.x = -0.18;
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
    this.#fly.position.y = 2.5 + Math.sin(time * 2) * 0.08;
    this.#wings.forEach((wing, index) => {
      wing.rotation.z = Math.sin(time * 42) * 0.22 * (index === 0 ? -1 : 1);
    });
    this.#renderer.render(this.#scene, this.#camera);
    this.#frame = requestAnimationFrame(this.tick);
  };
}
