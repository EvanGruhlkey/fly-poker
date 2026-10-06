import { describe, expect, it } from 'vitest';
import { createSession, act } from './demo';
describe('local table demo', () => {
  it('advances to the turn and charges both stacks', () => {
    const next = act(createSession(), 'call', 8);
    expect(next.street).toBe(1);
    expect(next.table.board).toEqual(['AS', '9D', '4C', 'TH']);
    expect(next.table.playerStackBb).toBe(90);
    expect(next.table.potBb).toBe(16.5);
  });
  it('ends a folded hand', () => {
    const next = act(createSession(), 'fold', 8);
    expect(next.kind).toBe('finished');
    expect(next.message).toBe('You folded. The fly takes the pot.');
    expect(act(next, 'call', 8)).toEqual(next);
  });
  it('caps a raise at the available stack', () => {
    const next = act(createSession(), 'raise', 1000);
    expect(next.table.playerStackBb).toBe(0);
    expect(next.table.flyStackBb).toBe(3.5);
  });
});
