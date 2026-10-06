import { describe,expect,it } from 'vitest';
import { evaluate,compareHands } from './evaluate';
describe('best five-card poker hand',()=>{
 it.each([
  [['AS','KS','QS','JS','TS','2H','3D'],'Straight flush'],
  [['AC','AD','AH','AS','KH','2C','3D'],'Four of a kind'],
  [['KC','KD','KH','QC','QD','2C','3D'],'Full house'],
  [['AC','JC','9C','6C','2C','KD','QH'],'Flush'],
  [['AC','2D','3H','4S','5C','KD','QH'],'Straight'],
  [['AC','AD','AH','KS','JC','8D','2H'],'Three of a kind'],
  [['AC','AD','KH','KS','JC','8D','2H'],'Two pair'],
  [['AC','AD','KH','QS','JC','8D','2H'],'One pair'],
  [['AC','KD','QH','9S','7C','4D','2H'],'High card'],
 ] as const)('recognizes %s as %s',(cards,name)=>{expect(evaluate(cards).label).toBe(name);});
 it('rejects duplicate cards and ranks a wheel straight flush correctly',()=>{
  expect(()=>evaluate(['AS','AS','KS','QS','JS'])).toThrow();
  expect(evaluate(['AS','2S','3S','4S','5S','KD','QH']).score).toEqual([8,5]);
 });
 it('uses six-high to beat the wheel',()=>{
  expect(compareHands(evaluate(['AC','2D','3H','4S','5C','KD','QH']),evaluate(['2C','3D','4H','5S','6C','KD','QH']))).toBeLessThan(0);
 });
 it('selects the higher trips when two triplets make a full house',()=>{
  expect(evaluate(['AC','AD','AH','KC','KD','KH','2S']).score).toEqual([6,14,13]);
 });
 it('breaks pair ties with every kicker and ignores suits',()=>{
  expect(compareHands(evaluate(['AC','AD','KH','QS','9C','8D','2H']),evaluate(['AH','AS','KD','QC','8H','7D','2C']))).toBeGreaterThan(0);
  expect(compareHands(evaluate(['AC','KD','QH','JS','TC','2D','3H']),evaluate(['AH','KS','QC','JD','TH','4D','5H']))).toBe(0);
 });
});

