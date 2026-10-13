export function sampleConnectome(model, limit=1024) {
 const {csr_indptr:ptr,csr_indices:sources}=model.arrays;
 const ids=[];const slots=new Map();const edges=[];
 const slot=id=>{if(!slots.has(id)){slots.set(id,ids.length);ids.push(id);}return slots.get(id);};
 let target=0;
 for(let i=0;i<Math.min(limit,sources.length);i++){
  const edge=Math.floor(i*sources.length/Math.min(limit,sources.length));
  while(ptr[target+1]<=edge)target++;
  edges.push([slot(sources[edge]),slot(target)]);
 }
 return {ids,edges};
}
