export function sampleAction(logits,rng=Math.random) {
 const maximum=Math.max(...logits);
 if(!Number.isFinite(maximum)||logits.some(x=>x!==-Infinity&&!Number.isFinite(x)))throw new Error('Invalid policy logits');
 const weights=logits.map(x=>Math.exp(x-maximum));
 const total=weights.reduce((sum,x)=>sum+x,0);
 const draw=rng();
 if(!Number.isFinite(draw)||draw<0||draw>=1)throw new Error('Invalid policy random draw');
 let remaining=draw*total;
 for(let index=0;index<weights.length;index++){
  remaining-=weights[index];
  if(remaining<0)return index;
 }
 return weights.findLastIndex(x=>x>0);
}
