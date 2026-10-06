import { describe, expect, it } from 'vitest';
import { createSession, act } from './demo';
const total = (state: ReturnType<typeof createSession>) => state.table.playerStackBb + state.table.flyStackBb + state.table.potBb;
describe('local table demo', () => {
 it('calls an existing bet and advances to a checked turn', () => {
  const next = act(createSession(), 'call', 8.5);
  expect(next.table.board).toEqual(['AS', '9D', '4C', 'TH']);
  expect(next.table.playerStackBb).toBe(89.75);
  expect(next.table.flyStackBb).toBe(97.5);
  expect(next.table.potBb).toBe(12.75);
  expect(next.toCall).toBe(0);
  expect(total(next)).toBe(200);
 });
 it('ends a folded hand and awards the pot once', () => {
  const next = act(createSession(), 'fold', 8.5);
  expect(next.kind).toBe('finished');
  expect(next.message).toBe('You folded. The fly takes the pot.');
  expect(next.table.flyStackBb).toBe(106);
  expect(next.table.potBb).toBe(0);
  expect(act(next, 'call', 8.5)).toEqual(next);
  expect(total(next)).toBe(200);
 });
 it('caps an all-in raise, reveals all cards, and pays the scripted winner', () => {
  const next = act(createSession(), 'raise', 1000);
  expect(next.table.playerStackBb).toBe(0);
  expect(next.table.flyStackBb).toBe(200);
  expect(next.table.potBb).toBe(0);
  expect(next.table.board).toHaveLength(5);
  expect(next.kind).toBe('finished');
 });
 it('requires a full minimum raise and only charges the fly the call difference', () => {
  const next = act(createSession(), 'raise', 4);
  expect(next.table.playerStackBb).toBe(85.5);
  expect(next.table.flyStackBb).toBe(93.25);
  expect(next.table.potBb).toBe(21.25);
  expect(total(next)).toBe(200);
 });
 it('checks for free on later streets and completes a coherent showdown', () => {
  const turn = act(createSession(), 'call', 8.5);
  const river = act(turn, 'call', 8.5);
  expect(river.table.playerStackBb).toBe(89.75);
  expect(river.table.potBb).toBe(12.75);
  const result = act(river, 'call', 8.5);
  expect(result.table.flyStackBb).toBe(110.25);
  expect(result.message).toContain('pair of aces');
  expect(result.table.potBb).toBe(0);
  expect(total(result)).toBe(200);
 });
});
