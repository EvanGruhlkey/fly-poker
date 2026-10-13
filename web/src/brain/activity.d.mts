import type {LoadedModel} from './runtime.mjs';
export function sampleConnectome(model:Pick<LoadedModel,'arrays'>,limit?:number):{ids:number[];edges:number[][]};
