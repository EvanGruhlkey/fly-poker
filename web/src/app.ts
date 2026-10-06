import { startHand,play,nextHand,legalActions } from './game/poker';
import { observeFly,chooseFly } from './game/bot';
import { qualityForWidth } from './game/model';
import type { PokerAction,PokerState,Transition } from './game/pokerTypes';
import { CasinoScene } from './scene/CasinoScene';
import { pokerMarkup } from './ui/pokerHud';
import { tableFor,statusFor,type Presentation } from './ui/pokerView';
export function createAppTitle():string{return 'Poker vs. a Fruit Fly';}
export function mountApp(root:HTMLElement):()=>void{
 let initial=startHand();let state:PokerState=initial.state;let scene:CasinoScene|undefined;let host:HTMLElement|undefined;
 let display:Presentation={table:tableFor(state),opponentCards:[],winningCards:[],animation:'none',dealingCards:state.holes.player,winner:'none',message:'Posting blinds…',busy:true};
 let raise=8;let generation=0;let timer:number|undefined;let resolveWait:((valid:boolean)=>void)|undefined;
 const reduced=()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches;
 const wait=(ms:number,ticket:number)=>new Promise<boolean>(resolve=>{resolveWait=resolve;timer=window.setTimeout(()=>{resolveWait=undefined;timer=undefined;resolve(ticket===generation);},reduced()?0:ms);});
 const cancel=()=>{generation++;if(timer!==undefined)window.clearTimeout(timer);timer=undefined;resolveWait?.(false);resolveWait=undefined;scene?.cancelAnimation();};
 const render=()=>{
  host?.remove();root.innerHTML=pokerMarkup(state,display);
  const placeholder=root.querySelector<HTMLElement>('[data-scene]');if(!placeholder)throw new Error('Casino scene host is missing');
  if(host){host.dataset.phase=display.table.phase;placeholder.replaceWith(host);}
  else{host=placeholder;try{scene=new CasinoScene(host,display.table,qualityForWidth(window.innerWidth));}catch{host.innerHTML='<div class="webgl-error">3D view unavailable. Cards and controls remain playable.</div>';}}
  const slider=root.querySelector<HTMLInputElement>('[data-raise-slider]');
  if(slider){raise=Math.min(Number(slider.max),Math.max(Number(slider.min),raise));slider.value=String(raise);
   const sync=()=>root.querySelectorAll('[data-raise-value],[data-raise-button]').forEach(node=>{node.textContent=`${raise/4} BB`;});sync();slider.addEventListener('input',()=>{raise=slider.valueAsNumber;sync();});}
  root.querySelectorAll<HTMLElement>('[data-action]').forEach(button=>button.addEventListener('click',()=>{
   const action=button.dataset.action;if(action==='fold'||action==='call'||action==='raise'||action==='all-in')void humanAction(action);
  }));
  root.querySelector('[data-reset]')?.addEventListener('click',()=>{cancel();void begin(startHand(),{player:400,fly:400});});
  root.querySelector('[data-next]')?.addEventListener('click',()=>{
   if(display.busy||state.kind!=='complete')return;cancel();
   const busted=!state.stacks.player||!state.stacks.fly;const stacks=busted?{player:400,fly:400}:state.stacks;
   void begin(busted?startHand():nextHand(state),stacks);
  });
  const dialog=root.querySelector<HTMLDialogElement>('[data-about]');
  root.querySelectorAll('[data-info]').forEach(button=>button.addEventListener('click',()=>dialog?.showModal()));
  root.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>dialog?.close()));
  dialog?.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
 };
 const execute=async(transition:Transition,ticket:number):Promise<boolean>=>{
  display={...display,busy:true,dealingCards:[]};render();
  for(const effect of transition.effects){
   if(ticket!==generation)return false;
   if(effect.kind==='chips'){
    display={...display,message:effect.moves[0]?.from==='pot'?'Chips return to the stacks…':'Chips into the pot…'};render();
    if(scene&&!await scene.animate([effect.moves]))return false;
    if(ticket!==generation)return false;
    const accounts={player:display.table.playerStackBb,fly:display.table.flyStackBb,pot:display.table.potBb};
    effect.moves.forEach(move=>{accounts[move.from]-=move.amountBb;accounts[move.to]+=move.amountBb;});
    display={...display,table:{...display.table,playerStackBb:accounts.player,flyStackBb:accounts.fly,potBb:accounts.pot}};render();
   }else if(effect.kind==='deal'){
    const newCards=effect.board.filter(card=>!display.table.board.includes(card));
    display={...display,table:{...display.table,board:effect.board},dealingCards:newCards,message:`Dealing the ${effect.board.length===3?'flop':effect.board.length===4?'turn':'river'}…`};render();
    if(!await wait(effect.board.length===3?750:650,ticket))return false;
   }else{
    const result=transition.state.kind==='complete'?transition.state.result:undefined;
    display={...display,opponentCards:transition.state.holes.fly,animation:'reveal',dealingCards:[],winningCards:result?.cards??[],winner:result?.winner??'none',message:result?`${result.winner==='split'?'Split pot':result.winner==='player'?'You win':'Fly wins'} · ${result.label}`:'Showdown.'};render();
    if(!await wait(1000,ticket))return false;
    display={...display,animation:'none'};
   }
  }
  if(ticket!==generation)return false;
  state=transition.state;display={...display,table:tableFor(state),dealingCards:[],message:statusFor(state),busy:false,winner:state.kind==='complete'?state.result.winner:'none'};
  display={...display,animation:state.kind==='complete'?'win':'none'};scene?.update(display.table);render();
  if(state.kind==='complete'){if(!await wait(750,ticket))return false;display={...display,animation:'none'};render();}return true;
 };
 const flyTurns=async(ticket:number)=>{
  while(ticket===generation&&state.kind==='betting'&&state.turn==='fly'){
   display={...display,busy:true,message:'The fly is thinking…'};render();
   if(!await wait(650+Math.random()*350,ticket))return;
   if(state.kind!=='betting'||state.turn!=='fly')return;
   const action=chooseFly(observeFly(state));if(!await execute(play(state,action),ticket))return;
  }
 };
 const humanAction=async(kind:'fold'|'call'|'raise'|'all-in')=>{
  if(display.busy||state.kind!=='betting'||state.turn!=='player')return;
  const legal=legalActions(state);let action:PokerAction;
  if(kind==='raise'){if(legal.raise.kind==='none')return;action={kind:'raise',to:raise};}
  else if(kind==='call'){action=legal.canCheck?{kind:'check'}:{kind:'call'};}
  else if(kind==='fold'){if(!legal.canFold)return;action={kind:'fold'};}
  else{if(!legal.canAllIn)return;action={kind:'all-in'};}
  const ticket=++generation;if(await execute(play(state,action),ticket))await flyTurns(ticket);
 };
 const begin=async(transition:Transition,stacks:{readonly player:number;readonly fly:number})=>{
  const ticket=++generation;state=transition.state;raise=8;
  display={table:{...tableFor(state),playerStackBb:stacks.player/4,flyStackBb:stacks.fly/4,potBb:0,board:[]},opponentCards:[],winningCards:[],animation:'none',dealingCards:state.holes.player,winner:'none',message:'Dealing a new hand…',busy:true};
  render();scene?.update(display.table);if(!await wait(500,ticket))return;
  if(await execute(transition,ticket))await flyTurns(ticket);
 };
 const keyboard=(event:KeyboardEvent)=>{
  if(event.target instanceof HTMLInputElement||root.querySelector('dialog[open]')||event.repeat)return;
  const action={f:'fold',c:'call',r:'raise',a:'all-in'}[event.key.toLowerCase()];
  if(action==='fold'||action==='call'||action==='raise'||action==='all-in'){event.preventDefault();void humanAction(action);}
 };
 void begin(initial,{player:400,fly:400});window.addEventListener('keydown',keyboard);
 return()=>{cancel();window.removeEventListener('keydown',keyboard);scene?.destroy();};
}

