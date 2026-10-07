import type { ChipAccount,ChipTransfer } from '../game/chips';
import type { Card,TableView } from '../game/model';
import type { PokerState } from '../game/pokerTypes';
export interface PokerTable extends TableView {readonly playerWagerBb:number;readonly flyWagerBb:number;}
export interface Presentation {
 readonly table:PokerTable;readonly opponentCards:readonly Card[];readonly winningCards:readonly Card[];
 readonly animation:'none'|'reveal'|'win';readonly dealingCards:readonly Card[];readonly winner:'player'|'fly'|'split'|'none';readonly message:string;readonly busy:boolean;
}
export function tableFor(state:PokerState):PokerTable{
 return {phase:state.kind==='complete'?'showdown':state.turn==='fly'?'fly-thinking':'player-turn',handNumber:state.handNumber,
  playerWagerBb:state.kind==='betting'?state.committed.player/4:0,flyWagerBb:state.kind==='betting'?state.committed.fly/4:0,
  playerStackBb:state.stacks.player/4,flyStackBb:state.stacks.fly/4,potBb:state.pot/4,playerCards:state.holes.player,board:state.board};
}
export function statusFor(state:PokerState):string{
 if(state.kind==='complete'){
  if(!state.stacks.player)return 'The fly wins the match. Play a rematch?';
  if(!state.stacks.fly)return 'You win the match. Play a rematch?';
  if(state.result.reason==='fold')return `${state.result.winner==='player'?'Fly folded. You win':'You folded. Fly wins'} ${state.result.awarded/4} BB.`;
  return `${state.result.winner==='split'?'Split pot':state.result.winner==='player'?'You win':'Fly wins'} · ${state.result.label} · ${state.result.awarded/4} BB.`;
 }
 return state.turn==='fly'?'The fly is thinking…':'Your turn.';
}





export function chipBalances(view:PokerTable):Record<ChipAccount,number>{
 return {player:view.playerStackBb,fly:view.flyStackBb,pot:view.potBb-view.playerWagerBb-view.flyWagerBb,'player-wager':view.playerWagerBb,'fly-wager':view.flyWagerBb};
}
export function moveChips(view:PokerTable,moves:readonly ChipTransfer[]):PokerTable{
 const balances=chipBalances(view);
 for(const move of moves){balances[move.from]-=move.amountBb;balances[move.to]+=move.amountBb;}
 return {...view,playerStackBb:balances.player,flyStackBb:balances.fly,playerWagerBb:balances['player-wager'],flyWagerBb:balances['fly-wager'],potBb:balances.pot+balances['player-wager']+balances['fly-wager']};
}
