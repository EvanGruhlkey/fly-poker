import {it,expect} from 'vitest';
import {sampleAction} from './sampling.mjs';
it('samples stable legal softmax without extra noise',()=>{
 const logits=[1000,999,-Infinity,1000];
 expect(sampleAction(logits,()=>0)).toBe(0);
 expect(sampleAction(logits,()=>.5)).toBe(1);
 expect(sampleAction(logits,()=>.999)).toBe(3);
 const counts=[0,0];let seed=91;
 const rng=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 for(let i=0;i<10000;i++)counts[sampleAction([Math.log(.7),Math.log(.3)],rng)]++;
 expect(counts[0]/10000).toBeCloseTo(.7,1);
 expect(()=>sampleAction([-Infinity,-Infinity])).toThrow();
});
