import type { Card } from './model';
export interface HandRank { readonly score:readonly number[];readonly label:string;readonly cards:readonly Card[]; }
const NAMES=['High card','One pair','Two pair','Three of a kind','Straight','Flush','Full house','Four of a kind','Straight flush'];
export function compareHands(left:HandRank,right:HandRank):number{
 for(let i=0;i<Math.max(left.score.length,right.score.length);i++){const difference=(left.score[i]??0)-(right.score[i]??0);if(difference)return difference;}
 return 0;
}
function five(cards:readonly Card[]):HandRank{
 const ranks=cards.map(card=>'23456789TJQKA'.indexOf(card[0])+2).sort((a,b)=>b-a);
 const counts=new Map<number,number>();ranks.forEach(rank=>counts.set(rank,(counts.get(rank)??0)+1));
 const groups=[...counts].sort((a,b)=>b[1]-a[1]||b[0]-a[0]);
 const flush=cards.every(card=>card[1]===cards[0][1]);
 const straight=counts.size===5?(ranks[0]-ranks[4]===4?ranks[0]:ranks.join(',')==='14,5,4,3,2'?5:0):0;
 let score:number[];
 if(flush&&straight)score=[8,straight];
 else if(groups[0][1]===4)score=[7,groups[0][0],groups[1][0]];
 else if(groups[0][1]===3&&groups[1][1]===2)score=[6,groups[0][0],groups[1][0]];
 else if(flush)score=[5,...ranks];
 else if(straight)score=[4,straight];
 else if(groups[0][1]===3)score=[3,groups[0][0],...groups.slice(1).map(group=>group[0])];
 else if(groups[0][1]===2&&groups[1][1]===2)score=[2,groups[0][0],groups[1][0],groups[2][0]];
 else if(groups[0][1]===2)score=[1,groups[0][0],...groups.slice(1).map(group=>group[0])];
 else score=[0,...ranks];
 return {score,label:NAMES[score[0]],cards};
}
export function evaluate(cards:readonly Card[]):HandRank{
 if(cards.length<5||cards.length>7||new Set(cards).size!==cards.length)throw new Error('Evaluate five to seven unique cards');
 let best=five(cards.slice(0,5));
 const choose=(start:number,selected:readonly Card[])=>{
  if(selected.length===5){const candidate=five(selected);if(compareHands(candidate,best)>0)best=candidate;return;}
  for(let i=start;i<=cards.length-(5-selected.length);i++)choose(i+1,[...selected,cards[i]]);
 };
 choose(0,[]);return best;
}
