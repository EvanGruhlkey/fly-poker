import type {BrainObservation} from './observation';
import type {PokerAction} from '../game/pokerTypes';
export function roundEven(value:number):number;
export function abstractActions(view:BrainObservation):readonly (PokerAction|null)[];
export function encodeObservation(view:BrainObservation,actions?:readonly (PokerAction|null)[]):Float32Array;
