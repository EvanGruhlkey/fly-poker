import * as THREE from 'three';
import type { TableView } from '../game/model';
import type { ChipAccount, ChipTransfer } from '../game/chips';
const POSITIONS: Record<ChipAccount, THREE.Vector3> = {
 player:new THREE.Vector3(1.1,1.82,1.5), fly:new THREE.Vector3(1.1,1.82,-1.45), pot:new THREE.Vector3(0,1.82,.1),
};
const DENOMINATIONS=[25,5,1,.25];
export class ChipStacks {
 readonly #root=new THREE.Group();
 readonly #stacks: Record<ChipAccount,THREE.Group>={player:new THREE.Group(),fly:new THREE.Group(),pot:new THREE.Group()};
 readonly #geometry=new THREE.CylinderGeometry(.145,.145,.045,24);
 readonly #stripeGeometry=new THREE.BoxGeometry(.04,.043,.016);
 readonly #materials=[0x181b18,0x477153,0x476a8b,0xe2d9c2].map(color=>new THREE.MeshStandardMaterial({color,roughness:.7}));
 readonly #ivory=new THREE.MeshStandardMaterial({color:0xf5edda,roughness:.8});
 #balances: Record<ChipAccount,number>={player:0,fly:0,pot:0};
 #frame=0;
 #generation=0;
 #resolve: ((complete:boolean)=>void)|undefined;
 constructor(scene:THREE.Scene,view:TableView){
  scene.add(this.#root);
  for(const account of ['player','fly','pot'] satisfies ChipAccount[]){this.#stacks[account].position.copy(POSITIONS[account]);this.#root.add(this.#stacks[account]);}
  this.update(view);
 }
 private pile(amount:number): THREE.Group {
  const group=new THREE.Group(); let remaining=Math.round(amount*4)/4;
  DENOMINATIONS.forEach((value,index)=>{
   const count=Math.floor(remaining/value);remaining-=count*value;
   for(let i=0;i<count;i++){
    const chip=new THREE.Group();
    const body=new THREE.Mesh(this.#geometry,this.#materials[index]);body.castShadow=true;chip.add(body);
    for(let edge=0;edge<6;edge++){
     const angle=edge*Math.PI/3;const stripe=new THREE.Mesh(this.#stripeGeometry,this.#ivory);
     stripe.position.set(Math.cos(angle)*.141,0,Math.sin(angle)*.141);stripe.rotation.y=-angle+Math.PI/2;chip.add(stripe);
    }
    chip.position.set((index-1.5)*.33,i*.046,0);group.add(chip);
   }
  });return group;
 }
 private rebuild(account:ChipAccount):void{const stack=this.#stacks[account];stack.clear();stack.add(this.pile(this.#balances[account]));}
 update(view:TableView):void{
  this.cancel();this.#balances={player:view.playerStackBb,fly:view.flyStackBb,pot:view.potBb};
  for(const account of ['player','fly','pot'] satisfies ChipAccount[])this.rebuild(account);
 }
 cancel():void{
  this.#generation++;cancelAnimationFrame(this.#frame);this.#resolve?.(false);this.#resolve=undefined;
  this.#root.children.filter(child=>!Object.values(this.#stacks).some(stack=>stack===child)).forEach(child=>this.#root.remove(child));
 }
 async animate(stages:readonly (readonly ChipTransfer[])[]):Promise<boolean>{
  const generation=this.#generation;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  for(const stage of stages){
   if(generation!==this.#generation)return false;
   if(reduced){stage.forEach(move=>{this.#balances[move.from]-=move.amountBb;this.#balances[move.to]+=move.amountBb;});for(const account of ['player','fly','pot'] satisfies ChipAccount[])this.rebuild(account);continue;}
   const complete=await this.move(stage);if(!complete)return false;
  }return generation===this.#generation;
 }
 private move(transfers:readonly ChipTransfer[]):Promise<boolean>{
  const flying=transfers.map(move=>{
   this.#balances[move.from]-=move.amountBb;this.rebuild(move.from);
   const group=this.pile(move.amountBb);group.position.copy(POSITIONS[move.from]);this.#root.add(group);return {move,group};
  });
  return new Promise(resolve=>{
   this.#resolve=resolve;const start=performance.now();
   const frame=(now:number)=>{
    const progress=Math.min(1,(now-start)/1100);const eased=progress*progress*(3-2*progress);
    flying.forEach(({move,group})=>{group.position.lerpVectors(POSITIONS[move.from],POSITIONS[move.to],eased);group.position.y+=Math.sin(progress*Math.PI)*.7;group.rotation.y=Math.sin(progress*Math.PI)*.18;});
    if(progress<1){this.#frame=requestAnimationFrame(frame);return;}
    flying.forEach(({move,group})=>{this.#root.remove(group);this.#balances[move.to]+=move.amountBb;this.rebuild(move.to);});
    this.#resolve=undefined;resolve(true);
   };this.#frame=requestAnimationFrame(frame);
  });
 }
 destroy():void{this.cancel();this.#root.removeFromParent();this.#geometry.dispose();this.#stripeGeometry.dispose();this.#materials.forEach(material=>material.dispose());this.#ivory.dispose();}
}

