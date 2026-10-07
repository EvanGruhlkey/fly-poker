import { describe,expect,it } from 'vitest';
import { startHand,play } from '../game/poker';
import { chipBalances,moveChips,tableFor } from './pokerView';

it('shows exact current bets while the pot total includes those bets',()=>{
 const view=tableFor(startHand().state);
 expect(chipBalances(view)).toEqual({player:99.5,fly:99,pot:0,'player-wager':.5,'fly-wager':1});
 expect(view.potBb).toBe(1.5);
 const collect=moveChips(view,[{from:'player-wager',to:'pot',amountBb:.5},{from:'fly-wager',to:'pot',amountBb:1}]);
 expect(chipBalances(collect)).toEqual({player:99.5,fly:99,pot:1.5,'player-wager':0,'fly-wager':0});
 expect(collect.potBb).toBe(1.5);
});

describe('animated table accounting',()=>{
 it('ends each effect sequence at the actual engine balances without chip duplication',()=>{
  const opening=startHand();
  let view={...tableFor(opening.state),playerStackBb:100,flyStackBb:100,potBb:0,playerWagerBb:0,flyWagerBb:0};
  for(const effect of opening.effects)if(effect.kind==='chips')view=moveChips(view,effect.moves);
  expect(view).toEqual(tableFor(opening.state));
  const fold=play(opening.state,{kind:'fold'});
  for(const effect of fold.effects)if(effect.kind==='chips'){
   view=moveChips(view,effect.moves);
   expect(Object.values(chipBalances(view)).reduce((sum,value)=>sum+value,0)).toBe(200);
   expect(Math.min(...Object.values(chipBalances(view)))).toBeGreaterThanOrEqual(0);
  }
  expect(chipBalances(view)).toEqual({player:99.5,fly:100.5,pot:0,'player-wager':0,'fly-wager':0});
  expect(chipBalances(tableFor(fold.state))).toEqual(chipBalances(view));
 });
});
