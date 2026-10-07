import type {Card} from '../game/model';
import type {Betting,LegalActions,PokerAction,Street} from '../game/pokerTypes';
import {legalActions} from '../game/poker';
import {abstractActions,encodeObservation} from './domain.mjs';
export interface BrainObservation {
 readonly holeCards:readonly [Card,Card];readonly board:readonly Card[];readonly street:Street;
 readonly dealer:boolean;readonly pot:number;readonly stack:number;readonly opponentStack:number;
 readonly committed:number;readonly opponentCommitted:number;readonly legal:LegalActions;
}
export function observeBrain(state:Betting):BrainObservation {
 if(state.turn!=='fly')throw new Error('The fly is not acting');
 return {holeCards:state.holes.fly,board:state.board,street:state.street,dealer:state.dealer==='fly',
  pot:state.pot,stack:state.stacks.fly,opponentStack:state.stacks.player,
  committed:state.committed.fly,opponentCommitted:state.committed.player,legal:legalActions(state)};
}
export function brainInput(view:BrainObservation):{features:Float32Array;mask:readonly boolean[];actions:readonly (PokerAction|null)[]} {
 const actions=abstractActions(view);
 return {features:encodeObservation(view,actions),mask:actions.map(action=>action!==null),actions};
}
