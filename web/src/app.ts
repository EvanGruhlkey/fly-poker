import { createSession, act, type DemoAction } from './game/demo';
import { planTransfers } from './game/chips';
import { qualityForWidth } from './game/model';
import { CasinoScene } from './scene/CasinoScene';
import { hudMarkup } from './ui/hud';
export function createAppTitle(): string { return 'Poker vs. a Fruit Fly'; }
export function mountApp(root: HTMLElement): () => void {
 let state=createSession();let scene:CasinoScene|undefined;let host:HTMLElement|undefined;
 let raise=8.5;let busy=false;let generation=0;
 const lock=()=>{
  root.querySelectorAll<HTMLButtonElement|HTMLInputElement>('[data-action],[data-raise-slider]').forEach(control=>{control.disabled=busy||state.kind==='finished';});
  root.querySelector('[data-controls]')?.setAttribute('aria-busy',String(busy));
 };
 const play=async(action:DemoAction)=>{
  if(busy||state.kind==='finished')return;
  const ticket=++generation;const next=act(state,action,raise);const transfers=planTransfers(state,next,action,raise);
  busy=true;lock();
  if(transfers.length){const status=root.querySelector('[role="status"]');if(status)status.textContent='Moving chips…';}
  const complete=scene?await scene.animate(transfers):true;
  if(ticket!==generation||!complete)return;
  state=next;busy=false;scene?.update(state.table);render();
 };
 const render=()=>{
  host?.remove();root.innerHTML=hudMarkup(state.table,state);
  const placeholder=root.querySelector<HTMLElement>('[data-scene]');
  if(!placeholder)throw new Error('Casino scene host is missing');
  if(host){host.dataset.phase=state.table.phase;placeholder.replaceWith(host);}
  else{
   host=placeholder;
   try{scene=new CasinoScene(host,state.table,qualityForWidth(window.innerWidth));}
   catch{host.innerHTML='<div class="webgl-error">3D view unavailable. Cards and controls remain playable.</div>';}
  }
  const slider=root.querySelector<HTMLInputElement>('[data-raise-slider]');
  if(slider){
   const sync=()=>root.querySelectorAll('[data-raise-value],[data-raise-button]').forEach(node=>{node.textContent=`${raise} BB`;});
   raise=Math.min(Number(slider.max),Math.max(Number(slider.min),raise));slider.value=String(raise);sync();
   slider.addEventListener('input',()=>{raise=slider.valueAsNumber;sync();});
  }
  root.querySelectorAll<HTMLElement>('[data-action]').forEach(button=>button.addEventListener('click',()=>{
   const action=button.dataset.action;if(action==='fold'||action==='call'||action==='raise')void play(action);
  }));
  root.querySelector('[data-new-game]')?.addEventListener('click',()=>{
   generation++;scene?.cancelAnimation();busy=false;state=createSession(state.table.handNumber+1);raise=8.5;scene?.update(state.table);render();
  });
  const dialog=root.querySelector<HTMLDialogElement>('[data-about]');
  root.querySelectorAll('[data-info]').forEach(button=>button.addEventListener('click',()=>dialog?.showModal()));
  root.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>dialog?.close()));
  dialog?.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});lock();
 };
 const keyboard=(event:KeyboardEvent)=>{
  if(event.target instanceof HTMLInputElement||root.querySelector('dialog[open]'))return;
  const action={f:'fold',c:'call',r:'raise'}[event.key.toLowerCase()];
  if(action==='fold'||action==='call'||action==='raise'){event.preventDefault();void play(action);}
 };
 render();window.addEventListener('keydown',keyboard);
 return()=>{generation++;window.removeEventListener('keydown',keyboard);scene?.destroy();};
}

