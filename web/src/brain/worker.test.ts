import {afterEach,expect,it,vi} from 'vitest';
import type {BrainRequest,BrainResponse} from './protocol';

const {parseModel}=vi.hoisted(()=>({parseModel:vi.fn()}));
vi.mock('./runtime.mjs',()=>({parseModel,forward:vi.fn()}));
afterEach(()=>{vi.unstubAllGlobals();vi.resetModules();vi.clearAllMocks();});

it.each([1024,0])('validates imitation-only training with %i completed steps',async(warmSteps)=>{
 const manifest={sha256:'00',graph_sha256:'4d63295f5e0d9b9ca7b4cf7802f975c2fc9fad87f29ee451305ccab5de02abe2',
  neurons:134209,connections:2700513,observation_version:'cards-context-v3',
  progress:{warm_steps:warmSteps,updates:0},training_config:{seed:43}};
 parseModel.mockReturnValue({manifest});
 const replies:BrainResponse[]=[];
 const worker:{location:{href:string};postMessage:(reply:BrainResponse)=>void;
  onmessage?: (event:{data:BrainRequest})=>Promise<void>}={
   location:{href:'http://localhost/worker.js'},postMessage:reply=>replies.push(reply)};
 vi.stubGlobal('self',worker);
 vi.stubGlobal('crypto',{subtle:{digest:async()=>new Uint8Array([0]).buffer}});
 vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce({ok:true,json:async()=>manifest})
  .mockResolvedValueOnce({ok:true,arrayBuffer:async()=>new ArrayBuffer(0)}));
 await import('./worker');
 await worker.onmessage?.({data:{kind:'load',id:1,manifestUrl:'http://localhost/brain/manifest.json'}});
 expect(replies[0]?.kind).toBe(warmSteps>0?'ready':'error');
});
