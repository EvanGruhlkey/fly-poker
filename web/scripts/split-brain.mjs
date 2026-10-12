import {readFile,writeFile,unlink} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';

const directory=resolve(process.argv[2]??'dist/brain');
const manifest=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));
const binary=await readFile(join(directory,'model.bin'));
if(createHash('sha256').update(binary).digest('hex')!==manifest.sha256)throw new Error('Brain checksum mismatch');
const size=12*1024*1024;
const parts=[];
for(let offset=0;offset<binary.length;offset+=size){
 const name=`model-${parts.length}.bin`;
 await writeFile(join(directory,name),binary.subarray(offset,offset+size));
 parts.push(name);
}
await writeFile(join(directory,'manifest.json'),JSON.stringify({...manifest,binary_parts:parts},null,2));
await unlink(join(directory,'model.bin'));
console.log(`Packaged ${binary.length} brain bytes in ${parts.length} parts`);
