import type { Card } from '../game/model';
const SUITS: Record<string,string> = {H:'hearts',D:'diamonds',S:'spades',C:'clubs'};
export function cardMarkup(card: Card): string {
 const rank = card[0] === 'T' ? '10' : card[0];
 return `<span class="playing-card"><img src="/assets/cards/${card}.svg" alt="${rank} of ${SUITS[card[1]]}" width="100" height="140"/></span>`;
}
