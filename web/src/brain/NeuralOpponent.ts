import type {PokerAction} from '../game/pokerTypes';
import {brainInput,type BrainObservation} from './observation';
import {parseResponse,type BrainResponse,type BrainRequest,type BrainMetadata} from './protocol';
interface Pending {readonly resolve:(response:BrainResponse)=>void;readonly reject:(error:Error)=>void;readonly timeout:ReturnType<typeof setTimeout>}
export class NeuralOpponent {
 private worker:Worker|undefined;
 private serial=0;
 private pending=new Map<number,Pending>();
 private ready:Promise<BrainMetadata>|undefined;
 private request(request:BrainRequest):Promise<BrainResponse>{
  return new Promise((resolve,reject)=>{
   const timeout=setTimeout(()=>{this.pending.delete(request.id);reject(new Error('Brain worker timed out. Retry loading the model.'));},60000);
   this.pending.set(request.id,{resolve,reject,timeout});this.worker?.postMessage(request);
  });
 }
 load():Promise<BrainMetadata>{
  if(this.ready)return this.ready;
  this.worker=new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
  this.worker.onmessage=(event:MessageEvent<unknown>)=>{
   try{const response=parseResponse(event.data);const pending=this.pending.get(response.id);if(!pending)return;
    clearTimeout(pending.timeout);this.pending.delete(response.id);
    if(response.kind==='error')pending.reject(new Error(response.message));else pending.resolve(response);
   }catch(error){this.fail(error instanceof Error?error:new Error('Invalid brain worker reply'));}
  };
  this.worker.onerror=()=>this.fail(new Error('Brain worker failed. Retry loading the model.'));
  this.ready=this.request({kind:'load',id:++this.serial,manifestUrl:new URL('brain/manifest.json',document.baseURI).href}).then(response=>{
   if(response.kind!=='ready')throw new Error('Brain did not finish loading');return response.metadata;
  });return this.ready;
 }
 async choose(view:BrainObservation):Promise<{action:PokerAction;inferenceMs:number}>{
  await this.load();const input=brainInput(view);
  const response=await this.request({kind:'infer',id:++this.serial,features:input.features,mask:input.mask});
  if(response.kind!=='result')throw new Error('Brain did not produce an action');
  const action=input.actions[response.index];if(!action)throw new Error('Brain selected an illegal action');
  return {action,inferenceMs:response.inferenceMs};
 }
 private fail(error:Error):void{
  this.pending.forEach(pending=>{clearTimeout(pending.timeout);pending.reject(error);});this.pending.clear();
 }
 destroy():void{this.fail(new Error('Brain request cancelled'));this.worker?.terminate();this.worker=undefined;this.ready=undefined;}
 retry():Promise<BrainMetadata>{this.destroy();return this.load();}
}
