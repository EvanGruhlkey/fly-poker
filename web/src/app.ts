import { createDemoTable, qualityForWidth } from "./game/model";
import { CasinoScene } from "./scene/CasinoScene";
import { hudMarkup } from "./ui/hud";

export function createAppTitle(): string {
  return "Poker vs. a Fruit Fly";
}

export function mountApp(root: HTMLElement): () => void {
  const view = createDemoTable();
  root.innerHTML = hudMarkup(view);
  const sceneHost = root.querySelector<HTMLElement>("[data-scene]");
  if (!sceneHost) throw new Error("Casino scene host is missing");

  let scene: CasinoScene | undefined;
  try {
    scene = new CasinoScene(sceneHost, view, qualityForWidth(window.innerWidth));
  } catch {
    sceneHost.innerHTML = '<div class="webgl-error">This casino needs WebGL. Poker controls remain available.</div>';
  }

  root.querySelector("[data-collapse]")?.addEventListener("click", () => {
    root.querySelector(".activity")?.classList.toggle("is-collapsed");
  });
  const toggleHud = (event: KeyboardEvent) => {
    if (event.key.toLowerCase() === "h") root.classList.toggle("hud-hidden");
  };
  window.addEventListener("keydown", toggleHud);

  return () => {
    window.removeEventListener("keydown", toggleHud);
    scene?.destroy();
  };
}
