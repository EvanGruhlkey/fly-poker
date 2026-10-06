import { describe,expect,it } from 'vitest';
import { startHand, legalActions, play, nextHand } from './poker';
import { orderedDeck } from './deck';
import type { PokerState } from './pokerTypes';
const total=(state:PokerState)=>state.stacks.player+state.stacks.fly+state.pot;
describe('heads-up Hold’em',()=>{
 it('posts blinds, deals four unique cards, and starts with the small blind',()=>{
  const {state,effects}=startHand({dealer:'player',deck:orderedDeck()});
  expect(state.stacks).toEqual({player:398,fly:396});expect(state.pot).toBe(6);
  expect(state.holes.fly).toEqual(['2C','4C']);expect(state.holes.player).toEqual(['3C','5C']);
  expect(state.board).toEqual([]);expect(state.kind==='betting'&&state.turn).toBe('player');
  expect(effects[0]).toEqual({kind:'chips',moves:[{from:'player',to:'pot',amountBb:.5},{from:'fly',to:'pot',amountBb:1}]});
 });
 it('keeps the big blind option after a limp and burns before the flop',()=>{
  let state=startHand({deck:orderedDeck()}).state;
  state=play(state,{kind:'call'}).state;
  expect(state.kind==='betting'&&state.turn).toBe('fly');expect(legalActions(state).canCheck).toBe(true);
  state=play(state,{kind:'check'}).state;
  expect(state.board).toEqual(['7C','8C','9C']);expect(state.deck).toHaveLength(44);
  expect(state.kind==='betting'&&state.turn).toBe('fly');expect(state.pot).toBe(8);expect(total(state)).toBe(800);
 });
 it('enforces full minimum raises and preserves the increment after a short all-in',()=>{
  let state=startHand({stacks:{player:788,fly:12}}).state;
  state=play(state,{kind:'raise',to:10}).state;
  expect(()=>play(state,{kind:'raise',to:11})).toThrow();
  state=play(state,{kind:'all-in'}).state;
  expect(state.kind==='betting'&&state.lastFullRaise).toBe(6);
  expect(legalActions(state).raise.kind).toBe('none');
  expect(legalActions(state).toCall).toBe(2);
 });
 it('refunds an unmatched shove before dealing and settling the pot',()=>{
  let state=startHand({stacks:{player:760,fly:40}}).state;
  state=play(state,{kind:'all-in'}).state;
  const transition=play(state,{kind:'call'});
  expect(transition.effects.some(effect=>effect.kind==='chips'&&effect.moves.some(move=>move.from==='pot'&&move.to==='player'&&move.amountBb===180))).toBe(true);
  expect(transition.effects.filter(effect=>effect.kind==='deal').map(effect=>effect.kind==='deal'&&effect.board.length)).toEqual([3,4,5]);
  expect(transition.state.kind).toBe('complete');expect(total(transition.state)).toBe(800);
 });
 it('allows a short opening all-in without reducing the full minimum increment',()=>{
  let state=startHand({stacks:{player:794,fly:6}}).state;
  state=play(state,{kind:'call'}).state;state=play(state,{kind:'check'}).state;
  state=play(state,{kind:'all-in'}).state;
  expect(state.kind==='betting'&&state.lastFullRaise).toBe(4);expect(legalActions(state).call).toBe(2);
  expect(legalActions(state).raise.kind).toBe('none');
  state=play(state,{kind:'call'}).state;expect(state.kind).toBe('complete');expect(total(state)).toBe(800);
 });
 it('runs out a hand when a blind consumes the shorter stack',()=>{
  const transition=startHand({stacks:{player:1,fly:799}});
  expect(transition.state.kind).toBe('complete');expect(transition.state.board).toHaveLength(5);
  expect(total(transition.state)).toBe(800);
  expect(transition.effects.some(effect=>effect.kind==='chips'&&effect.moves.some(move=>move.from==='pot'&&move.to==='fly'&&move.amountBb===.75))).toBe(true);
 });
 it('folds award once and next hands carry balances with an alternating button',()=>{
  const folded=play(startHand({}).state,{kind:'fold'}).state;
  expect(folded.stacks).toEqual({player:398,fly:402});expect(folded.pot).toBe(0);
  expect(()=>play(folded,{kind:'call'})).toThrow();
  const next=nextHand(folded).state;
  expect(next.dealer).toBe('fly');expect(next.handNumber).toBe(2);expect(total(next)).toBe(800);
 });
});

