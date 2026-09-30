import { createAppTitle } from "./app";

const root = document.querySelector<HTMLElement>("#app");

if (!root) {
  throw new Error("Missing #app root");
}

root.textContent = createAppTitle();
