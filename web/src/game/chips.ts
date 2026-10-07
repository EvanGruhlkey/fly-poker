export type ChipAccount='player'|'fly'|'pot'|'player-wager'|'fly-wager';
export const CHIP_ACCOUNTS=['player','fly','pot','player-wager','fly-wager'] satisfies ChipAccount[];
export const wagerAccount=(seat:'player'|'fly'):'player-wager'|'fly-wager'=>seat==='player'?'player-wager':'fly-wager';
export interface ChipTransfer {readonly from:ChipAccount;readonly to:ChipAccount;readonly amountBb:number;}
