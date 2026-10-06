import { createSession, act } from './game/demo';
import { qualityForWidth } from './game/model';
import { CasinoScene } from './scene/CasinoScene';
import { hudMarkup } from './ui/hud';
export function createAppTitle(): string { return 'Poker vs. a Fruit Fly'; }
export function mountApp(root: HTMLElement): () => void {
 let state = createSession();
 let scene: CasinoScene | undefined;
 let raise = 8;
 const render = () => {
  scene?.destroy();
  root.innerHTML = hudMarkup(state.table, state);
  const host = root.querySelector<HTMLElement>('[data-scene]');
  if (!host) throw new Error('Casino scene host is missing');
  try { scene = new CasinoScene(host, state.table, qualityForWidth(window.innerWidth)); }
  catch { host.innerHTML = '<div class="webgl-error">3D view unavailable. The table controls and cards remain playable.</div>'; }
  const slider = root.querySelector<HTMLInputElement>('[data-raise-slider]');
  if (slider) { slider.value = String(raise); slider.addEventListener('input', () => {
   raise = slider.valueAsNumber;
   root.querySelectorAll('[data-raise-value],[data-raise-button]').forEach(node => { node.textContent = `${raise} BB`; });
  }); }
  root.querySelectorAll<HTMLElement>('[data-action]').forEach(button => button.addEventListener('click', () => {
   const action = button.dataset.action;
   if (action === 'fold' || action === 'call' || action === 'raise') { state = act(state, action, raise); render(); }
  }));
  root.querySelector('[data-new-game]')?.addEventListener('click', () => { state = createSession(state.table.handNumber + 1); raise = 8; render(); });
  const dialog = root.querySelector<HTMLDialogElement>('[data-about]');
  root.querySelectorAll('[data-info]').forEach(button => button.addEventListener('click', () => dialog?.showModal()));
  root.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => dialog?.close()));
  dialog?.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
 };
 const keyboard = (event: KeyboardEvent) => {
  if (event.target instanceof HTMLInputElement || root.querySelector('dialog[open]')) return;
  const action = { f: 'fold', c: 'call', r: 'raise' }[event.key.toLowerCase()];
  if (action === 'fold' || action === 'call' || action === 'raise') { event.preventDefault(); state = act(state, action, raise); render(); }
 };
 render(); window.addEventListener('keydown', keyboard);
 return () => { window.removeEventListener('keydown', keyboard); scene?.destroy(); };
}
