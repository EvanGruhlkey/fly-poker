import { describe,expect,it } from 'vitest';
import { observeFly,chooseFly } from './bot';
import { startHand,play } from './poker';
describe('local fly opponent',()=>{
 it('receives only its own cards and public information',()=>{
  const state=play(startHand({}).state,{kind:'call'}).state;
  if(state.kind!=='betting')throw new Error('Expected betting');
  const view=observeFly(state);
  expect(view.holeCards).toEqual(state.holes.fly);expect(view).not.toHaveProperty('holes');expect(view).not.toHaveProperty('deck');
  expect(Object.keys(view).sort()).toEqual(['board','committed','holeCards','legal','opponentStack','pot','stack']);
 });
 it('only chooses a legal action over different random choices',()=>{
  const state=play(startHand({}).state,{kind:'call'}).state;
  if(state.kind!=='betting')throw new Error('Expected betting');
  for(let i=0;i<100;i++)expect(()=>play(state,chooseFly(observeFly(state),()=>i/100))).not.toThrow();
 });
});
