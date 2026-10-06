import { describe,expect,it } from 'vitest';
import { startHand,legalActions,play,nextHand } from './poker';
import { orderedDeck,shuffledDeck } from './deck';
import type { Card } from './model';
import type { PokerAction,PokerState,Transition } from './pokerTypes';
const rig=(prefix:readonly Card[])=>[...prefix,...orderedDeck().filter(card=>!prefix.includes(card))];
function ledger(before:{player:number;fly:number;pot:number},transition:Transition):void{
 const accounts={...before};
 transition.effects.forEach(effect=>{if(effect.kind==='chips')effect.moves.forEach(move=>{
  const amount=move.amountBb*4;expect(Number.isInteger(amount)).toBe(true);
  accounts[move.from]-=amount;accounts[move.to]+=amount;expect(accounts[move.from]).toBeGreaterThanOrEqual(0);
 });});
 expect(accounts).toEqual({...transition.state.stacks,pot:transition.state.pot});
 expect(accounts.player+accounts.fly+accounts.pot).toBe(800);
}
describe('complete poker matches',()=>{
 it('splits a royal-board-equivalent tie and returns both buy-ins',()=>{
  let state=startHand({deck:rig(['2C','3D','4H','5S','6C','7H','8H','9H','AC','TH','AD','JH'])}).state;
  while(state.kind==='betting')state=play(state,legalActions(state).canCheck?{kind:'check'}:{kind:'call'}).state;
  expect(state.result.winner).toBe('split');expect(state.result.label).toBe('Straight flush');
  expect(state.stacks).toEqual({player:400,fly:400});expect(state.pot).toBe(0);
 });
 it('rejects checking a bet and leaves the input state unchanged',()=>{
  const state=startHand({}).state;const snapshot=JSON.stringify(state);
  expect(()=>play(state,{kind:'check'})).toThrow();expect(JSON.stringify(state)).toBe(snapshot);
  expect(()=>startHand({deck:orderedDeck().fill('AS')})).toThrow();
 });
 it('runs randomized legal actions with exact transfer-ledger conservation',()=>{
  let seed=991;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  let transition=startHand({deck:shuffledDeck(random)});ledger({player:400,fly:400,pot:0},transition);let state=transition.state;
  for(let step=0;step<1500;step++){
   if(state.kind==='complete'){
    if(!state.stacks.player||!state.stacks.fly){transition=startHand({deck:shuffledDeck(random)});ledger({player:400,fly:400,pot:0},transition);}
    else{transition=nextHand(state,shuffledDeck(random));ledger({...state.stacks,pot:0},transition);}
   }else{
    const legal=legalActions(state);let action:PokerAction=legal.canCheck?{kind:'check'}:{kind:'call'};
    const roll=random();
    if(legal.canFold&&roll<.1)action={kind:'fold'};
    else if(legal.raise.kind==='raise'&&roll>.65)action={kind:'raise',to:legal.raise.minTo+Math.floor(random()*(legal.raise.maxTo-legal.raise.minTo+1))};
    transition=play(state,action);ledger({...state.stacks,pot:state.pot},transition);
   }
   state=transition.state;
  }
 });
 it('finishes twenty seeded all-in matches without manufacturing chips',()=>{
  for(let seed=1;seed<=20;seed++){
   let number=seed;const random=()=>{number=(Math.imul(number,1664525)+1013904223)>>>0;return number/4294967296;};
   let state:PokerState=startHand({deck:shuffledDeck(random)}).state;
   for(let hand=0;hand<100&&state.stacks.player>0&&state.stacks.fly>0;hand++){
    while(state.kind==='betting'){
     const legal=legalActions(state);const action:PokerAction=legal.canAllIn?{kind:'all-in'}:legal.canCheck?{kind:'check'}:{kind:'call'};
     const transition=play(state,action);ledger({...state.stacks,pot:state.pot},transition);state=transition.state;
    }
    if(state.stacks.player&&state.stacks.fly)state=nextHand(state,shuffledDeck(random)).state;
   }
   expect(state.kind).toBe('complete');expect(Math.min(state.stacks.player,state.stacks.fly)).toBe(0);
   expect(state.stacks.player+state.stacks.fly).toBe(800);
   expect(()=>nextHand(state)).toThrow();
  }
 });
});
