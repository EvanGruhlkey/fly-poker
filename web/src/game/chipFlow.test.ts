import { describe,expect,it } from 'vitest';
import { startHand,play } from './poker';

describe('poker chip movement',()=>{
 it('leaves blinds and raises in front of their owners until the betting round closes',()=>{
  const opening=startHand();
  expect(opening.effects[0]).toEqual({kind:'chips',moves:[
   {from:'player',to:'player-wager',amountBb:.5},{from:'fly',to:'fly-wager',amountBb:1},
  ]});
  const raise=play(opening.state,{kind:'raise',to:20});
  expect(raise.effects).toEqual([{kind:'chips',moves:[{from:'player',to:'player-wager',amountBb:4.5}]}]);
  const call=play(raise.state,{kind:'call'});
  expect(call.effects.slice(0,2)).toEqual([
   {kind:'chips',moves:[{from:'fly',to:'fly-wager',amountBb:4}]},
   {kind:'chips',moves:[{from:'player-wager',to:'pot',amountBb:5},{from:'fly-wager',to:'pot',amountBb:5}]},
  ]);
  expect(call.effects[2].kind).toBe('deal');
 });
 it('returns an uncalled bet before collecting and awarding a folded pot',()=>{
  const raise=play(startHand().state,{kind:'raise',to:40});
  const fold=play(raise.state,{kind:'fold'});
  expect(fold.effects).toEqual([
   {kind:'chips',moves:[{from:'player-wager',to:'player',amountBb:9}]},
   {kind:'chips',moves:[{from:'player-wager',to:'pot',amountBb:1},{from:'fly-wager',to:'pot',amountBb:1}]},
   {kind:'chips',moves:[{from:'pot',to:'player',amountBb:2}]},
  ]);
  expect(fold.state.stacks).toEqual({player:404,fly:396});
 });
 it('collects the final matched wagers before the all-in runout and showdown',()=>{
  const shove=play(startHand({stacks:{player:760,fly:40}}).state,{kind:'all-in'});
  const call=play(shove.state,{kind:'call'});
  expect(call.effects.slice(0,3)).toEqual([
   {kind:'chips',moves:[{from:'fly',to:'fly-wager',amountBb:9}]},
   {kind:'chips',moves:[{from:'player-wager',to:'player',amountBb:180}]},
   {kind:'chips',moves:[{from:'player-wager',to:'pot',amountBb:10},{from:'fly-wager',to:'pot',amountBb:10}]},
  ]);
  expect(call.effects.slice(3).map(effect=>effect.kind)).toEqual(['deal','deal','deal','reveal','chips']);
 });
});
