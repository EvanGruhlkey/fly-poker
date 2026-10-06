import { shuffledDeck, orderedDeck } from './deck';
import { compareHands,evaluate } from './evaluate';
import { other,bb,streetName,type Seat,type Street,type Betting,type PokerState,type PokerAction,type LegalActions,type Transition,type Effect,type Balances } from './pokerTypes';
import type { Card } from './model';
export function legalActions(state:PokerState):LegalActions{
 if(state.kind==='complete')return {toCall:0,call:0,canCheck:false,canFold:false,canAllIn:false,raise:{kind:'none'}};
 const actor=state.turn;const toCall=Math.max(0,state.currentBet-state.committed[actor]);
 const maxTo=state.committed[actor]+state.stacks[actor];
 const canRaise=state.raiseRights.includes(actor)&&state.stacks[other(actor)]>0&&maxTo>state.currentBet;
 return {toCall,call:Math.min(state.stacks[actor],toCall),canCheck:toCall===0,canFold:toCall>0,
  canAllIn:state.stacks[actor]>0&&(state.stacks[actor]<=toCall||canRaise),
  raise:canRaise?{kind:'raise',minTo:Math.min(maxTo,state.currentBet+state.lastFullRaise),maxTo}:{kind:'none'}};
}
const chips=(from:Seat|'pot',to:Seat|'pot',amount:number):Effect=>({kind:'chips',moves:[{from,to,amountBb:amount/4}]});
function refund(state:Betting,effects:Effect[]):Betting{
 const difference=state.committed.player-state.committed.fly;
 if(!difference)return state;
 const seat:Seat=difference>0?'player':'fly';const amount=Math.abs(difference);
 effects.push(chips('pot',seat,amount));
 return {...state,pot:state.pot-amount,stacks:{...state.stacks,[seat]:state.stacks[seat]+amount},committed:{...state.committed,[seat]:state.committed[seat]-amount},
  history:[...state.history,`${seat==='player'?'You':'Fly'} get ${bb(amount)} BB uncalled back.`]};
}
function showdown(state:Betting,effects:Effect[]):Transition{
 const hands={player:evaluate([...state.holes.player,...state.board]),fly:evaluate([...state.holes.fly,...state.board])};
 const comparison=compareHands(hands.player,hands.fly);const winner:Seat|'split'=comparison>0?'player':comparison<0?'fly':'split';
 effects.push({kind:'reveal',hands});
 const stacks={...state.stacks};
 if(winner==='split'){
  const half=Math.floor(state.pot/2);const odd=state.pot%2;
  stacks.player+=half+(other(state.dealer)==='player'?odd:0);stacks.fly+=half+(other(state.dealer)==='fly'?odd:0);
  effects.push({kind:'chips',moves:[{from:'pot',to:'player',amountBb:(stacks.player-state.stacks.player)/4},{from:'pot',to:'fly',amountBb:(stacks.fly-state.stacks.fly)/4}]});
 }else{stacks[winner]+=state.pot;effects.push(chips('pot',winner,state.pot));}
 const rank=winner==='split'?hands.player:hands[winner];const description=winner==='split'?`Split pot · ${rank.label}.`:`${winner==='player'?'You win':'Fly wins'} ${bb(state.pot)} BB · ${rank.label}.`;
 return {state:{...state,kind:'complete',stacks,pot:0,result:{winner,reason:'showdown',label:rank.label,cards:rank.cards,awarded:state.pot},history:[...state.history,description]},effects};
}
function dealStreet(state:Betting,effects:Effect[]):Betting{
 const street:Street=state.street===0?1:state.street===1?2:3;const count=street===1?3:1;
 const board=[...state.board,...state.deck.slice(1,count+1)];
 effects.push({kind:'deal',board});
 return {...state,street,board,deck:state.deck.slice(count+1),committed:{player:0,fly:0},currentBet:0,lastFullRaise:4,
  turn:other(state.dealer),pending:['player','fly'],raiseRights:['player','fly'],history:[...state.history,`${streetName(street)} · ${board.slice(-count).join(' ')}.`]};
}
function settle(state:Betting,effects:Effect[]):Transition{
 state=refund(state,effects);
 const allIn=state.stacks.player===0||state.stacks.fly===0;
 if(allIn){while(state.street<3)state=dealStreet(state,effects);return showdown(state,effects);}
 if(state.street===3)return showdown(state,effects);
 return {state:dealStreet(state,effects),effects};
}
function noResponseNeeded(state:Betting):boolean{
 if(state.stacks.player===0&&state.committed.fly>=state.committed.player)return true;
 if(state.stacks.fly===0&&state.committed.player>=state.committed.fly)return true;
 return false;
}
export function startHand(options:{readonly stacks?:Balances;readonly dealer?:Seat;readonly handNumber?:number;readonly deck?:readonly Card[]}={}):Transition{
 const stacks={...(options.stacks??{player:400,fly:400})};const dealer=options.dealer??'player';const deck=options.deck??shuffledDeck();
 if(Object.values(stacks).some(value=>!Number.isInteger(value)||value<=0)||stacks.player+stacks.fly!==800)throw new Error('A live match has 800 quarter-blind units');
 if(deck.length!==52||new Set(deck).size!==52||deck.some(card=>!orderedDeck().includes(card)))throw new Error('Deal from a unique standard 52-card deck');
 const holes:Record<Seat,readonly [Card,Card]>={player:dealer==='player'?[deck[1],deck[3]]:[deck[0],deck[2]],fly:dealer==='fly'?[deck[1],deck[3]]:[deck[0],deck[2]]};
 const committed={player:0,fly:0};const effects:Effect[]=[];
 for(const seat of [dealer,other(dealer)]){const amount=Math.min(stacks[seat],seat===dealer?2:4);stacks[seat]-=amount;committed[seat]=amount;}
 effects.push({kind:'chips',moves:[{from:dealer,to:'pot',amountBb:committed[dealer]/4},{from:other(dealer),to:'pot',amountBb:committed[other(dealer)]/4}]});
 const state:Betting={kind:'betting',handNumber:options.handNumber??1,dealer,stacks,committed,pot:committed.player+committed.fly,holes,deck:deck.slice(4),board:[],street:0,
  turn:dealer,currentBet:Math.max(committed.player,committed.fly),lastFullRaise:4,pending:[dealer,other(dealer)],raiseRights:['player','fly'],
  history:[`Hand ${options.handNumber??1} · ${dealer==='player'?'You':'Fly'} on the button.`,`${dealer==='player'?'You':'Fly'} post ${bb(committed[dealer])} BB. ${other(dealer)==='player'?'You':'Fly'} post ${bb(committed[other(dealer)])} BB.`]};
 return noResponseNeeded(state)?settle(state,effects):{state,effects};
}
export function play(state:PokerState,action:PokerAction):Transition{
 if(state.kind==='complete')throw new Error('The hand is already complete');
 const actor=state.turn;const opponent=other(actor);const legal=legalActions(state);const effects:Effect[]=[];
 if(action.kind==='all-in'){
  if(!legal.canAllIn)throw new Error('All-in is not legal');
  action=state.stacks[actor]<=legal.toCall?{kind:'call'}:{kind:'raise',to:state.committed[actor]+state.stacks[actor]};
 }
 if(action.kind==='fold'){
  if(!legal.canFold)throw new Error('Check when no bet is faced');
  const matched=refund(state,effects);effects.push(chips('pot',opponent,matched.pot));
  return {state:{...matched,kind:'complete',pot:0,stacks:{...matched.stacks,[opponent]:matched.stacks[opponent]+matched.pot},
   result:{winner:opponent,reason:'fold',label:'Fold',cards:[],awarded:matched.pot},history:[...matched.history,`${actor==='player'?'You fold':'Fly folds'}. ${opponent==='player'?'You win':'Fly wins'} ${bb(matched.pot)} BB.`]},effects};
 }
 let paid=0;let currentBet=state.currentBet;let lastFullRaise=state.lastFullRaise;let pending=state.pending.filter(seat=>seat!==actor);let rights=state.raiseRights.filter(seat=>seat!==actor);let description='';
 if(action.kind==='check'){if(!legal.canCheck)throw new Error('Cannot check facing a bet');description='check';}
 else if(action.kind==='call'){if(!legal.toCall)throw new Error('No bet to call');paid=legal.call;description=`call ${bb(paid)} BB${paid===state.stacks[actor]?' all-in':''}`;}
 else{
  if(legal.raise.kind==='none'||!Number.isInteger(action.to)||action.to<legal.raise.minTo||action.to>legal.raise.maxTo)throw new Error('Illegal raise amount');
  paid=action.to-state.committed[actor];const increase=action.to-state.currentBet;const full=increase>=state.lastFullRaise;
  currentBet=action.to;pending=[opponent];
  if(full){lastFullRaise=increase;rights=[opponent];}
  description=`${state.currentBet?'raise to':'bet'} ${bb(action.to)} BB${paid===state.stacks[actor]?' all-in':''}`;
 }
 if(paid)effects.push(chips(actor,'pot',paid));
 const next:Betting={...state,pot:state.pot+paid,stacks:{...state.stacks,[actor]:state.stacks[actor]-paid},committed:{...state.committed,[actor]:state.committed[actor]+paid},
  turn:opponent,currentBet,lastFullRaise,pending,raiseRights:rights,history:[...state.history,`${actor==='player'?'You':'Fly'} ${description}.`]};
 return !pending.length||noResponseNeeded(next)?settle(next,effects):{state:next,effects};
}
export function nextHand(state:PokerState,deck?:readonly Card[]):Transition{
 if(state.kind!=='complete')throw new Error('Finish this hand first');
 return startHand({stacks:state.stacks,dealer:other(state.dealer),handNumber:state.handNumber+1,deck});
}
