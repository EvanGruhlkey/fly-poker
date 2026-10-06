import type { Card } from './model';
import { evaluate } from './evaluate';
import { legalActions } from './poker';
import type { Betting,LegalActions,PokerAction } from './pokerTypes';
export interface FlyObservation {
 readonly holeCards:readonly [Card,Card];readonly board:readonly Card[];readonly pot:number;
 readonly stack:number;readonly opponentStack:number;readonly committed:number;readonly legal:LegalActions;
}
export function observeFly(state:Betting):FlyObservation{
 if(state.turn!=='fly')throw new Error('The fly is not acting');
 return {holeCards:state.holes.fly,board:state.board,pot:state.pot,stack:state.stacks.fly,opponentStack:state.stacks.player,committed:state.committed.fly,legal:legalActions(state)};
}
function strength(view:FlyObservation):number{
 const ranks=view.holeCards.map(card=>'23456789TJQKA'.indexOf(card[0])+2);
 if(view.board.length<3)return (ranks[0]+ranks[1])/40+(ranks[0]===ranks[1]?.25:0)+(view.holeCards[0][1]===view.holeCards[1][1]?.06:0);
 const score=evaluate([...view.holeCards,...view.board]).score;
 return [ .2,.38,.62,.75,.8,.86,.93,.98,1 ][score[0]]+(score[0]<2?(score[1]??0)/60:0);
}
export function chooseFly(view:FlyObservation,random:()=>number=Math.random):PokerAction{
 const legal=view.legal;const value=strength(view);const roll=random();
 if(legal.toCall&&legal.canFold&&roll<Math.max(.03,legal.call/(view.pot+legal.call)-value*.45))return {kind:'fold'};
 if(legal.raise.kind==='raise'&&((value>.64&&roll<.45)||(roll>.94&&view.stack>legal.toCall*3))){
  const target=view.committed+legal.toCall+Math.max(4,Math.round(view.pot*(value>.85?.9:.5)));
  return {kind:'raise',to:Math.min(legal.raise.maxTo,Math.max(legal.raise.minTo,target))};
 }
 return legal.canCheck?{kind:'check'}:{kind:'call'};
}
