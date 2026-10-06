import type { Card,TableView } from '../game/model';
import type { PokerState } from '../game/pokerTypes';
export type PokerTable=Pick<TableView,'phase'|'handNumber'|'potBb'|'playerStackBb'|'flyStackBb'|'playerCards'|'board'>;
export interface Presentation {
 readonly table:PokerTable;readonly opponentCards:readonly Card[];readonly winningCards:readonly Card[];
 readonly dealingCards:readonly Card[];readonly winner:'player'|'fly'|'split'|'none';readonly message:string;readonly busy:boolean;
}
export function tableFor(state:PokerState):PokerTable{
 return {phase:state.kind==='complete'?'showdown':state.turn==='fly'?'fly-thinking':'player-turn',handNumber:state.handNumber,
  playerStackBb:state.stacks.player/4,flyStackBb:state.stacks.fly/4,potBb:state.pot/4,playerCards:state.holes.player,board:state.board};
}
export function statusFor(state:PokerState):string{
 if(state.kind==='complete'){
  if(!state.stacks.player)return 'The fly wins the match. Play a rematch?';
  if(!state.stacks.fly)return 'You win the match. Play a rematch?';
  if(state.result.reason==='fold')return `${state.result.winner==='player'?'You win':'Fly wins'} ${state.result.awarded/4} BB. Opponent folded.`;
  return `${state.result.winner==='split'?'Split pot':state.result.winner==='player'?'You win':'Fly wins'} · ${state.result.label} · ${state.result.awarded/4} BB.`;
 }
 return state.turn==='fly'?'The fly is thinking…':'Your turn.';
}

