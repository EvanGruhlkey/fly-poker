import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {BrainActivity} from '../brain/protocol';

export class ConnectomePanel {
 private readonly element=document.createElement('aside');
 private readonly canvas=document.createElement('canvas');
 private readonly scene=new THREE.Scene();
 private readonly camera=new THREE.PerspectiveCamera(38,1.5,.01,20);
 private readonly abort=new AbortController();
 private renderer:THREE.WebGLRenderer|undefined;
 private controls:OrbitControls|undefined;
 private geometry:THREE.BufferGeometry|undefined;
 private material:THREE.ShaderMaterial|undefined;
 private points:THREE.Points|undefined;
 private activity:THREE.BufferAttribute|undefined;
 private frame:BrainActivity|undefined;
 private animation:number|undefined;
 private groups=new Uint8Array(134209);
 private busy=false;
 private clock=0;
 private last=0;
 private frames=0;
 private measured=0;
 isOpen=false;
 constructor(){
  this.element.className='connectome-panel';this.element.hidden=true;
  this.element.setAttribute('aria-label','Fly connectome activity');
  this.canvas.setAttribute('role','img');this.canvas.setAttribute('aria-label','3D FlyWire connectome. Drag to rotate, scroll to zoom.');
  this.element.append(this.canvas);document.body.append(this.element);
  this.camera.position.set(0,0,3.7);
  void this.load();
 }
 private async load():Promise<void>{
  try{
   const response=await fetch(new URL('connectome/anatomy.bin',document.baseURI),{signal:this.abort.signal});
   if(!response.ok)throw new Error('Connectome positions unavailable');
   const buffer=await response.arrayBuffer();if(this.abort.signal.aborted)return;
   if(buffer.byteLength!==134209*6*4)throw new Error('Invalid connectome positions');
   const raw=new Float32Array(buffer);if(!raw.every(Number.isFinite))throw new Error('Invalid neuron positions');
   const positions=new Float32Array(134209*3);const colors=new Float32Array(134209*3);
   for(let i=0;i<134209;i++){positions.set(raw.subarray(i*6,i*6+3),i*3);colors.set(raw.subarray(i*6+3,i*6+6),i*3);}
   const groups=new Map<string,number>();for(let i=0;i<134209;i++){const key=colors.subarray(i*3,i*3+3).join(',');if(!groups.has(key))groups.set(key,groups.size);this.groups[i]=groups.get(key)??0;}
   this.geometry=new THREE.BufferGeometry();
   this.geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
   this.geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
   this.activity=new THREE.BufferAttribute(new Float32Array(134209),1).setUsage(THREE.DynamicDrawUsage);
   this.geometry.setAttribute('activity',this.activity);
   this.material=new THREE.ShaderMaterial({transparent:true,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending,
    vertexShader:'attribute vec3 color; attribute float activity; varying vec3 c; varying float a; void main(){c=color; a=activity; vec4 p=modelViewMatrix*vec4(position,1.); gl_Position=projectionMatrix*p; gl_PointSize=(1.8+activity*12.)/max(.8,-p.z*.4);}',
    fragmentShader:'varying vec3 c; varying float a; void main(){float r=length(gl_PointCoord-.5)*2.; if(r>1.)discard; float glow=exp(-r*r*4.); gl_FragColor=vec4(mix(c,vec3(1.,.96,.72),a*.65),glow*(.16+a*.84));}'});
   this.points=new THREE.Points(this.geometry,this.material);this.scene.add(this.points);
   this.renderer=new THREE.WebGLRenderer({canvas:this.canvas,antialias:false,alpha:false,powerPreference:'high-performance'});
   this.renderer.setClearColor(0x040408);this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
   this.controls=new OrbitControls(this.camera,this.canvas);this.controls.enablePan=false;this.controls.enableDamping=true;this.controls.minDistance=2;this.controls.maxDistance=6;
   if(this.frame)this.receive(this.frame);
   if(this.isOpen)this.start();
  }catch(error){if(!this.abort.signal.aborted)this.canvas.setAttribute('aria-label',error instanceof Error?error.message:'3D connectome unavailable');}
 }
 toggle(button:HTMLButtonElement):void{
  this.isOpen=!this.isOpen;this.element.hidden=!this.isOpen;button.setAttribute('aria-expanded',String(this.isOpen));
  if(this.isOpen)this.start();else this.stop();
 }
 thinking():void{this.busy=true;}
 finished():void{this.busy=false;}
 reset():void{this.busy=false;this.frame=undefined;if(this.activity){this.activity.array.fill(0);this.activity.needsUpdate=true;}}
 receive(frame:BrainActivity):void{
  this.frame=frame;
  if(this.activity){
   const peaks=new Float32Array(16);
   frame.ids.forEach((id,i)=>{const group=this.groups[id]??0;peaks[group]=Math.max(peaks[group]??0,frame.rates[i]??0);});
   frame.ids.forEach((id,i)=>{const rate=frame.rates[i]??0;const peak=peaks[this.groups[id]??0]??0;
    this.activity?.setX(id,rate>0&&peak>0?Math.log1p(100*rate/peak)/Math.log(101):0);
   });this.activity.needsUpdate=true;
  }
  this.canvas.setAttribute('aria-label',`134,209 neurons in 3D, ${frame.ids.length} live neuron activations, brightness relative within each neuron group, processing step ${frame.step+1}. Drag to rotate.`);
 }
 private start():void{
  if(this.animation!==undefined||!this.renderer)return;
  this.last=performance.now();this.measured=this.last;this.frames=0;
  const tick=(now:number)=>{
   const elapsed=Math.min(now-this.last,100);this.last=now;
   const width=this.element.clientWidth;const height=Math.round(width*2/3);
   if(this.canvas.clientHeight!==height||this.canvas.width!==Math.round(width*this.renderer!.getPixelRatio())){this.renderer!.setSize(width,height,false);this.canvas.style.height=`${height}px`;this.camera.aspect=width/height;this.camera.updateProjectionMatrix();}
   if(this.points&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches){this.clock+=elapsed*.00025;this.points.rotation.y=.28*Math.sin(this.clock);this.points.rotation.x=-.06+.06*Math.sin(this.clock*.6);}
   if(this.activity&&!this.busy){const decay=Math.exp(-elapsed/450);for(const id of this.frame?.ids??[])this.activity.setX(id,this.activity.getX(id)*decay);this.activity.needsUpdate=true;}
   this.controls?.update();this.renderer!.render(this.scene,this.camera);
   this.frames++;if(now-this.measured>=1000){this.canvas.dataset.fps=String(Math.round(this.frames*1000/(now-this.measured)));this.frames=0;this.measured=now;}
   this.animation=requestAnimationFrame(tick);
  };this.animation=requestAnimationFrame(tick);
 }
 private stop():void{if(this.animation!==undefined)cancelAnimationFrame(this.animation);this.animation=undefined;}
 destroy():void{this.stop();this.abort.abort();this.controls?.dispose();this.geometry?.dispose();this.material?.dispose();this.renderer?.dispose();this.element.remove();}
}
