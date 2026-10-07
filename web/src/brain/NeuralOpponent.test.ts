import {describe,it,expect,vi,afterEach} from 'vitest';
import {NeuralOpponent} from './NeuralOpponent';
import {observeBrain} from './observation';
import {startHand,play} from '../game/poker';
import type {BrainRequest,BrainResponse} from './protocol';
class TestWorker {
 static current:TestWorker;
 onmessage:((event:{data:BrainResponse})=>void)|undefined;
 onerror:(()=>void)|undefined;
 sent:BrainRequest[]=[];
 terminated=false;
 constructor(){TestWorker.current=this;}
 postMessage(message:BrainRequest){this.sent.push(message);}
 terminate(){this.terminated=true;}
 respond(response:BrainResponse){this.onmessage?.({data:response});}
}
const metadata={modelHash:'a'.repeat(64),graphHash:'b'.repeat(64),neurons:134209,connections:2700513,warmSteps:64,updates:32,seed:17};
afterEach(()=>vi.unstubAllGlobals());
describe('persistent local brain requests',()=>{
 it('loads once, sends only encoded observations, and rejects cancelled inference',async()=>{
  vi.stubGlobal('Worker',TestWorker);vi.stubGlobal('document',{baseURI:'http://localhost/'});
  const brain=new NeuralOpponent();const ready=brain.load();const worker=TestWorker.current;
  worker.respond({kind:'ready',id:1,metadata});await ready;
  const state=play(startHand().state,{kind:'call'}).state;
  if(state.kind!=='betting')throw new Error('Expected betting');
  const choice=brain.choose(observeBrain(state));await Promise.resolve();
  expect(worker.sent).toHaveLength(2);
  expect(worker.sent[1]).toHaveProperty('features');expect(worker.sent[1]).not.toHaveProperty('holes');
  brain.destroy();await expect(choice).rejects.toThrow('cancelled');expect(worker.terminated).toBe(true);
 });
 it('surfaces model errors and retries with a fresh worker',async()=>{
  vi.stubGlobal('Worker',TestWorker);vi.stubGlobal('document',{baseURI:'http://localhost/'});
  const brain=new NeuralOpponent();const load=brain.load();
  TestWorker.current.respond({kind:'error',id:1,message:'Checksum mismatch'});
  await expect(load).rejects.toThrow('Checksum mismatch');
  const old=TestWorker.current;const retry=brain.retry();
  expect(old.terminated).toBe(true);TestWorker.current.respond({kind:'ready',id:2,metadata});
  await expect(retry).resolves.toEqual(metadata);brain.destroy();
 });
});
