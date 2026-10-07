import {parseModel,forward,type LoadedModel} from './runtime.mjs';
import type {BrainRequest,BrainResponse} from './protocol';
const graphHash='4d63295f5e0d9b9ca7b4cf7802f975c2fc9fad87f29ee451305ccab5de02abe2';
let model:LoadedModel|undefined;
const reply=(message:BrainResponse)=>self.postMessage(message);
self.onmessage=async(event:MessageEvent<BrainRequest>)=>{
 const request=event.data;
 try{
  if(request.kind==='load'){
   const response=await fetch(request.manifestUrl,{cache:'no-cache'});
   if(!response.ok)throw new Error('Trained brain manifest is unavailable. Install the released model assets and retry.');
   const manifest:unknown=await response.json();
   if(!manifest||typeof manifest!=='object'||!('sha256' in manifest)||typeof manifest.sha256!=='string')throw new Error('Invalid brain manifest');
   const binaryUrl=new URL('model.bin',new URL(request.manifestUrl,self.location.href));binaryUrl.searchParams.set('sha256',manifest.sha256);
   const binaryResponse=await fetch(binaryUrl,{cache:'force-cache'});
   if(!binaryResponse.ok)throw new Error('Trained brain weights are unavailable.');
   const buffer=await binaryResponse.arrayBuffer();
   const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
   if(hash!==manifest.sha256)throw new Error('Brain weights failed SHA-256 verification.');
   const loaded=parseModel(manifest,buffer);
   const m=loaded.manifest;
   if(m.graph_sha256!==graphHash||m.neurons!==134209||m.connections!==2700513||
      !m.progress?.warm_steps||!m.progress?.updates||!Number.isInteger(m.training_config?.seed))throw new Error('This is not the trained full-connectome poker model.');
   model=loaded;
   reply({kind:'ready',id:request.id,metadata:{modelHash:hash,graphHash:m.graph_sha256,neurons:m.neurons,connections:m.connections,
    warmSteps:m.progress.warm_steps,updates:m.progress.updates,seed:m.training_config.seed}});
  }else{
   if(!model)throw new Error('Trained brain is not loaded.');
   const start=performance.now();const result=forward(model,request.features,request.mask);
   let index=0;for(let i=1;i<7;i++)if(result.logits[i]>result.logits[index])index=i;
   reply({kind:'result',id:request.id,index,value:result.value,inferenceMs:performance.now()-start});
  }
 }catch(error){reply({kind:'error',id:request.id,message:error instanceof Error?error.message:'Brain inference failed.'});}
};
