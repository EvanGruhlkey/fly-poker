import type { Card } from './model';
export const RANKS=['2','3','4','5','6','7','8','9','T','J','Q','K','A'] as const;
export function orderedDeck():Card[]{
 const deck:Card[]=[];
 for(const suit of ['C','D','H','S'] as const)for(const rank of RANKS){const card:Card=`${rank}${suit}`;deck.push(card);}
 return deck;
}
export function shuffledDeck(random:()=>number=()=>crypto.getRandomValues(new Uint32Array(1))[0]/4294967296):Card[]{
 const deck=orderedDeck();
 for(let i=deck.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[deck[i],deck[j]]=[deck[j],deck[i]];}
 return deck;
}

