export function roundEven(value) {
  const floor=Math.floor(value);
  return value-floor===.5 ? floor+(floor%2) : Math.round(value);
}
export function abstractActions(view) {
  const legal=view.legal;
  const actions=Array(7).fill(null);
  if (legal.canFold) actions[0]={kind:'fold'};
  actions[1]=legal.canCheck?{kind:'check'}:{kind:'call'};
  if (legal.raise.kind==='raise') {
    const matched=view.committed+legal.call;
    const targets=[legal.raise.minTo, matched+Math.max(4,roundEven((view.pot+legal.call)*.5)),
      matched+Math.max(4,view.pot+legal.call),matched+Math.max(4,2*(view.pot+legal.call)),legal.raise.maxTo];
    const seen=new Set();
    targets.forEach((amount,index)=>{
      const to=Math.min(legal.raise.maxTo,Math.max(legal.raise.minTo,amount));
      if (!seen.has(to)) {actions[index+2]={kind:'raise',to};seen.add(to);}
    });
  }
  return actions;
}
export function encodeObservation(view,actions=abstractActions(view),version='cards-public-v1') {
  const count={'cards-public-v1':118,'cards-history-v2':250,'cards-context-v3':267}[version];
  if(!count)throw new Error('Unsupported observation version');
  const features=new Float32Array(count);
  const cardIndex=card=>'23456789TJQKA'.indexOf(card[0])*4+'cdhs'.indexOf(card[1].toLowerCase());
  view.holeCards.forEach(card=>{features[cardIndex(card)]=1;});
  view.board.forEach(card=>{features[52+cardIndex(card)]=1;});
  features[104+view.street]=1;
  const raises=actions.filter(action=>action?.kind==='raise').map(action=>action.to);
  features.set([Number(view.dealer),view.pot/800,view.legal.call/400,view.stack/400,
    view.opponentStack/400,view.committed/400,view.opponentCommitted/400,view.board.length/5,
    raises.length?Math.min(...raises)/400:0,raises.length?Math.max(...raises)/400:0],108);
  if(count>=250) (view.publicActions??[]).slice(-12).forEach((event,index)=>{
    const offset=118+index*11;
    features[offset+(event.actor==='fly'?0:1)]=1;features[offset+2+event.street]=1;
    features[offset+6+['fold','check','call','raise'].indexOf(event.kind)]=1;features[offset+10]=event.paid/800;
  });
  if(count===267) {
    const effective=Math.min(view.stack,view.opponentStack);
    features.set([effective/400,effective/Math.max(1,effective+view.pot),
      view.legal.call/Math.max(1,view.pot+view.legal.call)],250);
    features.set(actions.map(action=>action?.kind==='raise'?action.to/400:action?.kind==='call'?view.legal.call/400:0),253);
    features.set(actions.map(action=>Number(action!==null)),260);
  }
  return features;
}
