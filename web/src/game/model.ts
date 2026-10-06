export type Card=`${'2'|'3'|'4'|'5'|'6'|'7'|'8'|'9'|'T'|'J'|'Q'|'K'|'A'}${'C'|'D'|'H'|'S'}`;
export type TablePhase='dealing'|'fly-thinking'|'player-turn'|'showdown';
export interface TableView {
 readonly phase:TablePhase;readonly handNumber:number;readonly potBb:number;
 readonly playerStackBb:number;readonly flyStackBb:number;
 readonly playerCards:readonly [Card,Card];readonly board:readonly Card[];
}
export interface SceneQuality {readonly shadows:boolean;readonly pixelRatio:number;}
export function qualityForWidth(width:number):SceneQuality{
 return width<800?{shadows:false,pixelRatio:1}:{shadows:true,pixelRatio:1.5};
}
