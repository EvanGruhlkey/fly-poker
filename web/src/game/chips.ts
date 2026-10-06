import { contributions, type DemoAction, type DemoSession } from './demo';
export type ChipAccount = 'player' | 'fly' | 'pot';
export interface ChipTransfer { readonly from: ChipAccount; readonly to: ChipAccount; readonly amountBb: number; }
export function planTransfers(before: DemoSession, after: DemoSession, action: DemoAction, amount: number): readonly (readonly ChipTransfer[])[] {
 if (before.kind === 'finished') return [];
 const paid=contributions(before,action,amount);
 const bets: ChipTransfer[]=[];
 if(paid.player>0) bets.push({from:'player',to:'pot',amountBb:paid.player});
 if(paid.fly>0) bets.push({from:'fly',to:'pot',amountBb:paid.fly});
 const stages: ChipTransfer[][]=[];
 if(bets.length) stages.push(bets);
 if(after.kind==='finished') stages.push([{from:'pot',to:'fly',amountBb:before.table.potBb+paid.player+paid.fly}]);
 return stages;
}
