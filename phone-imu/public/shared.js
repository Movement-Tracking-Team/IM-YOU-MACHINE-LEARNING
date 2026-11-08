export const version=1;
export const clock=()=>performance.now()/1000;
export function timestampOrder(timestamp,previous){
  return timestamp===previous?'duplicate':timestamp<previous?'out_of_order':'increasing';
}
export function pairingToken(){
  const t=new URLSearchParams(location.hash.slice(1)).get('token');
  if(t){localStorage.setItem('imu-token',t);history.replaceState(null,'',location.pathname);}
  return t||localStorage.getItem('imu-token')||'';
}
export function normalizeTimestamp(raw,handler=performance.now(),origin=performance.timeOrigin){
  if(!Number.isFinite(raw)||raw<=0)throw new Error('Motion event has no usable timestamp');
  // Older WebKit used epoch milliseconds. Preserve raw; normalize explicitly to this page's performance clock.
  const legacy=raw>1e12;
  const timestamp=legacy?raw-origin:raw;
  if(Math.abs(timestamp-handler)>60000)throw new Error('Event and performance clocks disagree; reload and resynchronize');
  return {timestamp:timestamp/1000,kind:legacy?'epoch_normalized':'performance_relative'};
}
export function motionSample(event,model){
  const a=event.accelerationIncludingGravity,r=event.rotationRate;
  if(!a||!r||![a.x,a.y,a.z,r.alpha,r.beta,r.gamma].every(Number.isFinite))return null;
  const handler=performance.now(),t=normalizeTimestamp(event.timeStamp,handler);
  return {original_device_timestamp_ms:event.timeStamp,phone_timestamp:t.timestamp,handler_timestamp:handler/1000,
    timestamp_kind:t.kind,estimated_computer_timestamp:model?t.timestamp-model.offset:0,sync_id:model?.id||0,
    ax:a.x,ay:a.y,az:a.z,gx:r.beta,gy:r.gamma,gz:r.alpha,
    rotation_alpha:r.alpha,rotation_beta:r.beta,rotation_gamma:r.gamma,
    browser_interval_ms:Number.isFinite(event.interval)?event.interval:null};
}
export function plot(canvas,history,keys){
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);if(history.length<2)return;
  const scale=Math.max(1,...history.flatMap(s=>keys.map(k=>Math.abs(s[k]))));
  ctx.strokeStyle='#d9e3e9';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  keys.forEach((key,j)=>{ctx.strokeStyle=['#008678','#cf6326','#5756c7'][j];ctx.lineWidth=1.6;ctx.beginPath();
    history.forEach((s,i)=>{const x=i/(history.length-1)*w,y=h/2-s[key]/scale*(h/2-8);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
}
export async function download(url,token,name){
  const r=await fetch(url,{headers:{Authorization:`Bearer ${token}`}});if(!r.ok)throw new Error(await r.text());
  const u=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
}
