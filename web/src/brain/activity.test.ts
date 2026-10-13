import {expect,it} from 'vitest';
import {sampleConnectome} from './activity.mjs';

it('samples real directed edges and preserves neuron identities',()=>{
 const model={arrays:{csr_indptr:new Int32Array([0,1,2,3]),csr_indices:new Int32Array([2,0,1])}};
 const sample=sampleConnectome(model,3);
 expect(sample.edges.map(([from,to])=>[sample.ids[from],sample.ids[to]])).toEqual([[2,0],[0,1],[1,2]]);
});
