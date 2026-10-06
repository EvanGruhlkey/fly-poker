import type { Card } from '../game/model';
const SUITS: Record<string, string> = { H: '♥', D: '♦', C: '♣', S: '♠' };
const POSITIONS: Record<string, readonly [number, number][]> = {
 A: [[50,50]], '2': [[50,25],[50,75]], '3': [[50,24],[50,50],[50,76]],
 '4': [[30,25],[70,25],[30,75],[70,75]],
 '5': [[30,25],[70,25],[50,50],[30,75],[70,75]],
 '6': [[30,24],[70,24],[30,50],[70,50],[30,76],[70,76]],
 '7': [[30,24],[70,24],[50,37],[30,50],[70,50],[30,76],[70,76]],
 '8': [[30,24],[70,24],[50,37],[30,50],[70,50],[50,63],[30,76],[70,76]],
 '9': [[30,22],[70,22],[30,40],[70,40],[50,50],[30,60],[70,60],[30,78],[70,78]],
 T: [[30,22],[70,22],[50,31],[30,40],[70,40],[30,60],[70,60],[50,69],[30,78],[70,78]],
};
export function cardSvg(card: Card): string {
 const rank = card[0] === 'T' ? '10' : card[0];
 const suit = SUITS[card[1]];
 const color = card[1] === 'H' || card[1] === 'D' ? '#ba392e' : '#17221d';
 const pips = POSITIONS[card[0]];
 const face = pips ? pips.map(([x,y]) => `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" font-size="${card[0] === 'A' ? 35 : 19}" ${y > 50 ? `transform="rotate(180 ${x} ${y})"` : ''}>${suit}</text>`).join('')
 : `<rect x="23" y="22" width="54" height="56" rx="2" fill="#eee4cf" stroke="${color}" stroke-width="1"/><path d="M30 65L35 39L43 46L50 30L57 46L65 39L70 65Z" fill="${color}"/><text x="50" y="62" text-anchor="middle" fill="#fff7e6" font-size="21">${rank}</text><text x="50" y="75" text-anchor="middle" font-size="13">${suit}</text>`;
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 140" role="img" aria-label="${rank} of ${{H:'hearts',D:'diamonds',C:'clubs',S:'spades'}[card[1]]}" fill="${color}" font-family="Georgia,serif"><rect x=".5" y=".5" width="99" height="139" rx="7" fill="#fff9e9" stroke="#d5c8a8"/><g transform="translate(0 20) scale(1 1)">${face}</g><g font-size="15" font-weight="bold"><text x="7" y="18">${rank}</text><text x="8" y="32" font-size="13">${suit}</text><g transform="rotate(180 50 70)"><text x="7" y="18">${rank}</text><text x="8" y="32" font-size="13">${suit}</text></g></g></svg>`;
}
export function cardMarkup(card: Card): string { return `<span class="playing-card">${cardSvg(card)}</span>`; }
