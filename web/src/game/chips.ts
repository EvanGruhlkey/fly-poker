export type ChipAccount='player'|'fly'|'pot';
export interface ChipTransfer {readonly from:ChipAccount;readonly to:ChipAccount;readonly amountBb:number;}
