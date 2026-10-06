import { describe, expect, it } from 'vitest';
import { createDemoTable } from '../game/model';
import { hudMarkup } from './hud';
describe('table interface', () => {
 it('shows readable cards, controls, and truthful demo status', () => {
  const markup = hudMarkup(createDemoTable());
  expect(markup).toContain('Call 4.25 BB');
  expect(markup).toContain('A of spades');
  expect(markup).toContain('K of hearts');
  expect(markup).toContain('connectome backend is not connected');
  expect(markup).toContain('data-phase="player-turn"');
  expect(markup).not.toContain('connectome online');
 });
});
