export interface BrainMetadata {
 readonly modelHash:string;readonly graphHash:string;readonly neurons:number;readonly connections:number;
 readonly warmSteps:number;readonly updates:number;readonly seed:number;
}
export type BrainStatus={readonly kind:'loading'}|{readonly kind:'ready';readonly metadata:BrainMetadata}|{readonly kind:'error';readonly message:string};
export type BrainRequest={readonly kind:'load';readonly id:number;readonly manifestUrl:string}|{readonly kind:'infer';readonly id:number;readonly features:Float32Array;readonly mask:readonly boolean[]};
export type BrainResponse={readonly kind:'ready';readonly id:number;readonly metadata:BrainMetadata}|{readonly kind:'result';readonly id:number;readonly index:number;readonly value:number;readonly inferenceMs:number}|{readonly kind:'error';readonly id:number;readonly message:string};
export function parseResponse(value:unknown):BrainResponse {
 if(!value||typeof value!=='object'||!('kind' in value)||!('id' in value)||typeof value.id!=='number')throw new Error('Invalid brain worker reply');
 if(value.kind==='error'&&'message' in value&&typeof value.message==='string')return {kind:'error',id:value.id,message:value.message};
 if(value.kind==='result'&&'index' in value&&'value' in value&&'inferenceMs' in value&&
  typeof value.index==='number'&&Number.isInteger(value.index)&&value.index>=0&&value.index<7&&
  typeof value.value==='number'&&Number.isFinite(value.value)&&typeof value.inferenceMs==='number')return {kind:'result',id:value.id,index:value.index,value:value.value,inferenceMs:value.inferenceMs};
 if(value.kind==='ready'&&'metadata' in value&&value.metadata&&typeof value.metadata==='object'){
  const m=value.metadata;
  if('modelHash' in m&&typeof m.modelHash==='string'&&'graphHash' in m&&typeof m.graphHash==='string'&&
   'neurons' in m&&typeof m.neurons==='number'&&'connections' in m&&typeof m.connections==='number'&&
   'warmSteps' in m&&typeof m.warmSteps==='number'&&'updates' in m&&typeof m.updates==='number'&&'seed' in m&&typeof m.seed==='number')
   return {kind:'ready',id:value.id,metadata:{modelHash:m.modelHash,graphHash:m.graphHash,neurons:m.neurons,connections:m.connections,warmSteps:m.warmSteps,updates:m.updates,seed:m.seed}};
 }
 throw new Error('Invalid brain worker reply');
}
