import {describe,it,expect} from 'vitest';
import {startHand,play} from '../game/poker';
import {observeBrain,brainInput} from './observation';
import {roundEven} from './domain.mjs';

describe('private neural observation',()=>{
 it('contains only own cards/public state and masks legal wagers',()=>{
  const state=play(startHand().state,{kind:'call'}).state;
  if(state.kind!=='betting')throw new Error('Expected betting');
  const view=observeBrain(state);const input=brainInput(view);
  expect(view).not.toHaveProperty('deck');expect(view).not.toHaveProperty('holes');
  expect(view.holeCards).toEqual(state.holes.fly);
  expect(input.features.length).toBe(118);
  expect(input.features[108]).toBe(0);
  for(const action of input.actions)if(action)expect(()=>play(state,action)).not.toThrow();
 });
 it('matches Python banker rounding for half-quarter pot bets',()=>{
  expect([1.5,2.5,3.5,4.5].map(roundEven)).toEqual([2,2,4,4]);
 });
 it('deduplicates clamped short all-ins',()=>{
  const state=startHand({dealer:'fly',stacks:{player:795,fly:5}}).state;
  if(state.kind!=='betting')throw new Error('Expected betting');
  const input=brainInput(observeBrain(state));
  expect(input.actions.filter(action=>action?.kind==='raise')).toHaveLength(1);
 });
});

it('matches trainer-sized wagers in real engine states, including underraises and capped calls',()=>{
 const inputFor=(state:ReturnType<typeof startHand>['state'])=>{
  if(state.kind!=='betting')throw new Error('Expected betting');return brainInput(observeBrain(state));
 };
 const preflop=inputFor(startHand({dealer:'fly'}).state);
 expect(preflop.actions).toEqual([{kind:'fold'},{kind:'call'},{kind:'raise',to:8},null,
  {kind:'raise',to:12},{kind:'raise',to:20},{kind:'raise',to:400}]);
 let flop=play(startHand().state,{kind:'call'}).state;
 flop=play(flop,{kind:'check'}).state;
 const checked=inputFor(flop);
 expect(checked.actions).toEqual([null,{kind:'check'},{kind:'raise',to:4},null,
  {kind:'raise',to:8},{kind:'raise',to:16},{kind:'raise',to:396}]);
 flop=play(flop,{kind:'raise',to:4}).state;flop=play(flop,{kind:'raise',to:12}).state;
 expect(inputFor(flop).actions).toEqual([{kind:'fold'},{kind:'call'},{kind:'raise',to:20},
  {kind:'raise',to:28},{kind:'raise',to:44},{kind:'raise',to:76},{kind:'raise',to:396}]);
 const short=play(startHand({stacks:{player:795,fly:5}}).state,{kind:'raise',to:8}).state;
 const shortInput=inputFor(short);
 expect(shortInput.mask).toEqual([true,true,false,false,false,false,false]);
 expect(shortInput.features[110]).toBeCloseTo(1/400);
 let under=play(startHand({dealer:'fly',stacks:{player:11,fly:789}}).state,{kind:'raise',to:8}).state;
 under=play(under,{kind:'all-in'}).state;
 expect(inputFor(under).actions).toEqual([{kind:'fold'},{kind:'call'},null,null,null,null,null]);
});
