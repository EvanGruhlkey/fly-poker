const fail = message => { throw new Error(message); };
const integer = value => Number.isSafeInteger(value) && value >= 0;
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);

export function parseModel(manifest, buffer) {
  if (!object(manifest) || manifest.format !== 'fly-poker-brain-v2' ||
      manifest.observation_version !== 'cards-public-v1' || manifest.action_version !== 'holdem-seven-v1' ||
      !object(manifest.model) || manifest.model.approximation !== 'signed-learned-saturating-rate-v2' ||
      !object(manifest.tensors) || manifest.bytes !== buffer.byteLength) fail('Incompatible brain export');
  const {neurons:n, connections:e, inputs:ni, outputs:no} = manifest;
  if (![n,e,ni,no].every(integer) || !n || !e || !ni || !no || n > 200000 || e > 4000000 ||
      ni > n || no > n || manifest.model.steps !== 8 || manifest.model.observation_features !== 118 ||
      manifest.model.action_count !== 7 || !(manifest.model.saturation > 0) ||
      manifest.layer_norm_epsilon !== 1e-5) fail('Invalid brain dimensions');
  const shapes = {csr_indptr:[n+1],csr_indices:[e],input_idx:[ni],output_idx:[no],signs:[e],
    weights:[e],leak:[n],bias:[n],'sensory.weight':[ni,118],'sensory.bias':[ni],
    'motor_norm.weight':[no],'motor_norm.bias':[no],'policy.weight':[7,no],
    'policy.bias':[7],'value.weight':[1,no],'value.bias':[1]};
  const arrays = {};
  let offset = 0;
  for (const [name, shape] of Object.entries(shapes)) {
    const descriptor = manifest.tensors[name];
    const expected = name.endsWith('_idx') || name.startsWith('csr_') || name === 'signs' ? 'int32' : 'float32';
    if (!object(descriptor) || !integer(descriptor.offset) || descriptor.offset !== offset ||
        !integer(descriptor.length) || descriptor.length !== shape.reduce((a,b)=>a*b,1) ||
        descriptor.dtype !== expected || JSON.stringify(descriptor.shape) !== JSON.stringify(shape) ||
        offset + descriptor.length*4 > buffer.byteLength) fail(`Invalid tensor: ${name}`);
    arrays[name] = expected === 'int32' ? new Int32Array(buffer,offset,descriptor.length) : new Float32Array(buffer,offset,descriptor.length);
    if (!arrays[name].every(Number.isFinite)) fail(`Non-finite tensor: ${name}`);
    offset += descriptor.length*4;
  }
  if (offset !== buffer.byteLength) fail('Unexpected brain data');
  const ptr = arrays.csr_indptr;
  if (ptr[0] !== 0 || ptr[n] !== e || !ptr.every((x,i)=>x>=0 && (i===0 || x>=ptr[i-1]))) fail('Invalid CSR offsets');
  for (const name of ['csr_indices','input_idx','output_idx']) {
    if (!arrays[name].every(x=>x>=0 && x<n)) fail(`Invalid neuron indices: ${name}`);
  }
  if (!arrays.weights.every((x,i)=>Math.sign(x)===arrays.signs[i] && Math.abs(arrays.signs[i])===1) ||
      !arrays.leak.every(x=>x>0 && x<1)) fail('Invalid signed weights or neuron leaks');
  return {manifest, arrays};
}

function linear(weights, bias, input) {
  const output = new Float32Array(bias.length);
  for (let row=0;row<output.length;row++) {
    let sum = 0;
    for (let col=0;col<input.length;col++) sum += weights[row*input.length+col]*input[col];
    output[row] = sum+bias[row];
  }
  return output;
}

export function forward(model, features, mask) {
  if (features.length!==118 || !features.every(Number.isFinite) || mask.length!==7 ||
      !mask.every(x=>typeof x==='boolean') || !mask.some(Boolean)) fail('Invalid private observation');
  const {arrays:a, manifest:m} = model;
  const sensory = linear(a['sensory.weight'], a['sensory.bias'], features);
  const injected = new Float32Array(m.neurons);
  a.input_idx.forEach((neuron,i)=>{injected[neuron]=sensory[i];});
  let rates = new Float32Array(m.neurons);
  let next = new Float32Array(m.neurons);
  const sat = m.model.saturation;
  for (let step=0;step<m.model.steps;step++) {
    for (let neuron=0;neuron<m.neurons;neuron++) {
      let sum=0;
      for (let edge=a.csr_indptr[neuron];edge<a.csr_indptr[neuron+1];edge++) sum+=a.weights[edge]*rates[a.csr_indices[edge]];
      const drive=Math.fround(Math.fround(Math.fround(sum)+injected[neuron])+a.bias[neuron]);
      const activation=Math.fround(sat*Math.tanh(Math.max(0,drive)/sat));
      const leak=a.leak[neuron];
      next[neuron]=Math.fround(Math.fround((1-leak)*rates[neuron])+Math.fround(leak*activation));
    }
    [rates,next]=[next,rates];
  }
  const motor=new Float32Array(m.outputs);
  a.output_idx.forEach((neuron,i)=>{motor[i]=rates[neuron];});
  const mean=motor.reduce((sum,x)=>sum+x,0)/motor.length;
  const variance=motor.reduce((sum,x)=>sum+(x-mean)**2,0)/motor.length;
  const scale=1/Math.sqrt(variance+m.layer_norm_epsilon);
  for (let i=0;i<motor.length;i++) motor[i]=(motor[i]-mean)*scale*a['motor_norm.weight'][i]+a['motor_norm.bias'][i];
  const logits=Array.from(linear(a['policy.weight'],a['policy.bias'],motor),(x,i)=>mask[i]?x:-Infinity);
  const value=Math.tanh(linear(a['value.weight'],a['value.bias'],motor)[0]);
  if (!logits.every((x,i)=>!mask[i] || Number.isFinite(x)) || !Number.isFinite(value)) fail('Brain produced non-finite output');
  return {logits,value};
}
