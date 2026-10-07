import {readFile,writeFile,rename,unlink,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {join} from 'node:path';
const releasedSha='d8caa3f93b8a126b1b5b421ab04f695d8f7681bf603337b8139c23caf487bcda';
const releasedModel='https://github.com/EvanGruhlkey/fly-poker/releases/download/brain-v2-seed17/model.bin';
const digest=data=>createHash('sha256').update(data).digest('hex');
export async function fetchBrain(directory, url=releasedModel, pinnedHash) {
 const manifest=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));
 if(manifest.format!=='fly-poker-brain-v2'||!Number.isInteger(manifest.bytes)||manifest.bytes<=0||manifest.bytes>40000000||
    typeof manifest.sha256!=='string'||!/^[a-f0-9]{64}$/.test(manifest.sha256))throw new Error('Invalid pinned brain manifest');
 if(pinnedHash&&manifest.sha256!==pinnedHash)throw new Error('Manifest differs from the pinned published model');
 const target=join(directory,'model.bin');
 const previous=await readFile(target).catch(error=>{if(error.code==='ENOENT')return undefined;throw error;});
 if(previous&&previous.length===manifest.bytes&&digest(previous)===manifest.sha256)return {downloaded:false,sha256:manifest.sha256};
 const response=await fetch(url);
 if(!response.ok)throw new Error(`Model download failed: HTTP ${response.status}`);
 const data=Buffer.from(await response.arrayBuffer());
 if(data.length!==manifest.bytes||digest(data)!==manifest.sha256)throw new Error('Model download failed size/SHA-256 verification');
 await mkdir(directory,{recursive:true});
 const temporary=target+`.${process.pid}.partial`;
 try{await writeFile(temporary,data);await rename(temporary,target);}finally{await unlink(temporary).catch(error=>{if(error.code!=='ENOENT')throw error;});}
 return {downloaded:true,sha256:manifest.sha256};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const directory=fileURLToPath(new URL('../public/brain/',import.meta.url));
 fetchBrain(directory,releasedModel,releasedSha).then(result=>console.log(`Trained brain verified: ${result.sha256}`)).catch(error=>{console.error(error.message);process.exitCode=1;});
}
