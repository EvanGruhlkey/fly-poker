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
