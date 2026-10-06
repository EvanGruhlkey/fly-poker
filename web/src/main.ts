import { mountApp } from "./app";
import "./casino.css";

const root = document.querySelector<HTMLElement>("#app");

if (!root) {
  throw new Error("Missing #app root");
}

mountApp(root);
