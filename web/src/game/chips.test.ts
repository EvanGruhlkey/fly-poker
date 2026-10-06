import { describe, expect, it } from 'vitest';
import { createSession, act } from './demo';
import { planTransfers } from './chips';
describe('chip movement', () => {
 it('moves a call into the pot', () => {
  const before = createSession();
  expect(planTransfers(before, act(before,'call',8.5),'call',8.5)).toEqual([[{from:'player',to:'pot',amountBb:4.25}]]);
 });
 it('moves a raise and the fly matching difference into the pot together', () => {
  const before=createSession();
  expect(planTransfers(before,act(before,'raise',10),'raise',10)).toEqual([[{from:'player',to:'pot',amountBb:10},{from:'fly',to:'pot',amountBb:5.75}]]);
 });
 it('keeps chips still on a check', () => {
  const before=act(createSession(),'call',8.5);
  expect(planTransfers(before,act(before,'call',8.5),'call',8.5)).toEqual([]);
 });
 it('moves the whole pot to the fly on a fold', () => {
  const before=createSession();
  expect(planTransfers(before,act(before,'fold',8.5),'fold',8.5)).toEqual([[{from:'pot',to:'fly',amountBb:8.5}]]);
 });
 it('bets before awarding a final pot and conserves exact chip amounts', () => {
  const before=createSession(); const after=act(before,'raise',94);
  const plan=planTransfers(before,after,'raise',94);
  expect(plan).toEqual([[{from:'player',to:'pot',amountBb:94},{from:'fly',to:'pot',amountBb:89.75}],[{from:'pot',to:'fly',amountBb:192.25}]]);
  const accounts={player:94,fly:97.5,pot:8.5};
  plan.flat().forEach(move=>{accounts[move.from]-=move.amountBb;accounts[move.to]+=move.amountBb;});
  expect(accounts).toEqual({player:0,fly:200,pot:0});
 });
});

