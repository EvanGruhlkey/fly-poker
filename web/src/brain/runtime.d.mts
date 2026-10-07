export interface LoadedModel {
 readonly manifest:{readonly neurons:number;readonly connections:number;readonly sha256:string;readonly graph_sha256:string;
 readonly bytes:number;readonly progress:{readonly warm_steps:number;readonly updates:number};readonly training_config:{readonly seed:number}};
 readonly arrays:Readonly<Record<string,Float32Array|Int32Array>>;
}
export function parseModel(manifest:unknown,buffer:ArrayBuffer):LoadedModel;
export function forward(model:LoadedModel,features:readonly number[]|Float32Array,mask:readonly boolean[]):{logits:number[];value:number};
