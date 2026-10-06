import { describe,expect,it } from 'vitest';
import { startHand,play } from '../game/poker';
import { orderedDeck } from '../game/deck';
import { pokerMarkup } from './pokerHud';
import { tableFor,statusFor,type Presentation } from './pokerView';
const presentation=(state:ReturnType<typeof startHand>['state']):Presentation=>({table:tableFor(state),opponentCards:[],winningCards:[],animation:'none',dealingCards:[],winner:'none',message:statusFor(state),busy:false});
describe('real poker interface',()=>{
 it('shows legal blind calls, true cards, and protects the fly hand',()=>{
  const state=startHand({deck:orderedDeck()}).state;const markup=pokerMarkup(state,presentation(state));
  expect(markup).toContain('Call 0.5 BB');expect(markup).toContain('3 of clubs');expect(markup).toContain('5 of clubs');
  expect(markup).not.toContain('/assets/cards/2C.svg');expect(markup).not.toContain('/assets/cards/4C.svg');
  expect(markup).toContain('min="8"');expect(markup).toContain('data-action="all-in"');expect(markup).toContain('The neural backend is not connected');
 });
 it('offers the next hand after a fold without revealing the fly cards',()=>{
  const state=play(startHand({deck:orderedDeck()}).state,{kind:'fold'}).state;const markup=pokerMarkup(state,presentation(state));
  expect(markup).toContain('Deal next hand');expect(markup).not.toContain('/assets/cards/2C.svg');
  expect(statusFor(state)).toContain('You folded');
 });
 it('disables human actions while the fly acts',()=>{
  const state=play(startHand({}).state,{kind:'call'}).state;const markup=pokerMarkup(state,presentation(state));
  expect(markup).toContain('data-action="call" disabled');expect(markup).toContain('The fly is thinking');
 });
});

