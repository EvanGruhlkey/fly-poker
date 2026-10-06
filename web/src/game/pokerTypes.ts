import type { Card } from './model';
import type { ChipTransfer } from './chips';
import type { HandRank } from './evaluate';
export type Seat='player'|'fly';
export type Street=0|1|2|3;
export type Chips=number;
export type Balances=Readonly<Record<Seat,Chips>>;
export interface Hand {
 readonly handNumber:number;readonly dealer:Seat;readonly stacks:Balances;readonly committed:Balances;
 readonly pot:Chips;readonly board:readonly Card[];readonly holes:Readonly<Record<Seat,readonly [Card,Card]>>;
 readonly deck:readonly Card[];readonly street:Street;readonly history:readonly string[];
}
export interface Betting extends Hand {
 readonly kind:'betting';readonly turn:Seat;readonly currentBet:Chips;readonly lastFullRaise:Chips;
 readonly pending:readonly Seat[];readonly raiseRights:readonly Seat[];
}
export interface Complete extends Hand {
 readonly kind:'complete';readonly result:{readonly winner:Seat|'split';readonly reason:'fold'|'showdown';readonly label:string;readonly cards:readonly Card[];readonly awarded:Chips};
}
export type PokerState=Betting|Complete;
export type PokerAction={readonly kind:'fold'}|{readonly kind:'check'}|{readonly kind:'call'}|{readonly kind:'all-in'}|{readonly kind:'raise';readonly to:Chips};
export interface LegalActions {
 readonly toCall:Chips;readonly call:Chips;readonly canCheck:boolean;readonly canFold:boolean;readonly canAllIn:boolean;
 readonly raise:{readonly kind:'none'}|{readonly kind:'raise';readonly minTo:Chips;readonly maxTo:Chips};
}
export type Effect={readonly kind:'chips';readonly moves:readonly ChipTransfer[]}|{readonly kind:'deal';readonly board:readonly Card[]}|{readonly kind:'reveal';readonly hands:Readonly<Record<Seat,HandRank>>};
export interface Transition {readonly state:PokerState;readonly effects:readonly Effect[];}
export const other=(seat:Seat):Seat=>seat==='player'?'fly':'player';
export const bb=(chips:Chips):string=>`${chips/4}`;
export const streetName=(street:Street):string=>['Preflop','Flop','Turn','River'][street];

