import {pairingToken,clock,motionSample,plot,timestampOrder} from './shared.js';
import {DurableBuffer} from './buffer.js';
const $=id=>document.getElementById(id),token=pairingToken();
const clientId=localStorage.getItem('imu-client')||crypto.randomUUID();localStorage.setItem('imu-client',clientId);
const bootId=crypto.randomUUID();let device=localStorage.getItem('imu-device')||'phone_1';$('device').value=device;
if(!['phone_1','phone_2'].includes(device)){device='phone_1';$('device').value=device;}
let ws,model=null,session=null,seq=0,batch=[],preview=[],history=[],lastSensor=-Infinity,times=[],wakeLock;
let enabled=false,welcomed=false,backoff=500,error='',writing=Promise.resolve(),pumping=false,lastPreview=0;
let buffer,storageOK=false;const pending=new Map(),inflight=new Map();
let ends=JSON.parse(localStorage.getItem('imu-ends')||'{}');
let eventIndex=0,timestampWarnings=0,warning='';
function showError(e){error=e instanceof Error?e.message:String(e);$('error').textContent=error;}
function showWarning(message){warning=message;$('warning').textContent=message;}
function send(m){if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(m));}
function saveEnds(){localStorage.setItem('imu-ends',JSON.stringify(ends));}
function key(p){return `${p.session_id}:${p.packet_sequence_number}`;}
const initialization=(async()=>{
  try{
    buffer=await new DurableBuffer().open();
    for(const {packet} of await buffer.all())pending.set(key(packet),packet);
    const interrupted=JSON.parse(localStorage.getItem('imu-active')||'null');
    if(interrupted){
      ends[interrupted.session_id]=interrupted.last_sequence;saveEnds();localStorage.removeItem('imu-active');
      showError('The page was reloaded during recording. Saved batches will replay; acquisition has a gap. Stop this session on the computer.');
    }
    storageOK=true;
  }catch(e){showError('Local recovery storage unavailable: '+e.message);}
})();

async function wake(){
  try{if('wakeLock'in navigator&&document.visibilityState==='visible')wakeLock=await navigator.wakeLock.request('screen');}
  catch{ $('sensor').textContent='Screen wake lock unavailable. Set Auto-Lock to Never during this test.'; }
}
function connect(){
  ws=new WebSocket(`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/ws`);
  ws.onopen=()=>send({type:'hello',protocol_version:1,role:'phone',token,device_id:device,client_id:clientId,boot_id:bootId});
  ws.onmessage=event=>{
    const p2=clock(),m=JSON.parse(event.data);
    if(m.type==='sync_probe'){send({type:'sync_reply',id:m.id,p2,p3:clock()});return;}
    if(m.type==='sync_model'){model=m;status();return;}
    if(m.type==='welcome'){
      welcomed=true;backoff=500;$('connection').textContent='Connected to computer';
      if(session && m.session?.session_id===session.session_id && m.session.stop!==null)session=m.session;
      if(session && m.session?.session_id!==session.session_id){showError('Receiver restarted or session changed. Local packets retained for recovery.');finish();}
      pump();status();return;
    }
    if(m.type==='prepare'){
      if(session||!storageOK||!model||clock()-lastSensor>1.2||document.hidden){showError('Cannot arm: sensors, visibility, storage or clock not ready');return;}
      session=m.session;seq=0;batch=[];
      localStorage.setItem('imu-active',JSON.stringify({session_id:session.session_id,last_sequence:-1}));
      send({type:'armed',session_id:session.session_id});$('stop').disabled=false;wake();return;
    }
    if(m.type==='stop'){
      if(session?.session_id===m.session.session_id)session=m.session;
      else if(!Object.hasOwn(ends,m.session.session_id)) {ends[m.session.session_id]=-1;saveEnds();}
      return;
    }
    if(m.type==='ack'){
      const k=`${m.session_id}:${m.sequence}`;inflight.delete(k);
      buffer.remove(m.session_id,m.sequence).then(()=>{pending.delete(k);pump();}).catch(e=>showError('Could not clear acknowledged buffer: '+e.message));return;
    }
    if(m.type==='end_ack'){delete ends[m.session_id];saveEnds();return;}
    if(m.type==='error')showError(m.message);
  };
  ws.onclose=e=>{
    welcomed=false;inflight.clear();$('connection').textContent='Disconnected — buffering locally';
    if(e.code===1008){showError(e.reason+'. Select an unused phone number and reload.');return;}
    setTimeout(connect,backoff);backoff=Math.min(5000,backoff*1.5);
  };
  ws.onerror=()=>{$('connection').textContent='Connection failed. Check Wi-Fi, certificate trust and receiver.';};
}
function status(){
  send({type:'status',sensors:enabled&&clock()-lastSensor<1.2,visible:!document.hidden,
    sync_id:model?.id,error,warning,buffered:pending.size});
}
function queueBatch(){
  if(!batch.length||!session)return;
  const p={type:'samples',protocol_version:1,device_id:device,device_label:device.replace('_',' '),
    client_id:clientId,boot_id:bootId,session_id:session.session_id,packet_sequence_number:seq++,
    accel_units:'m/s^2',gyro_units:'deg/s',samples:batch};batch=[];
  // Memory copy survives a storage error; download recovery exposes it. Never evict unacknowledged data.
  pending.set(key(p),p);
  writing=writing.then(async()=>{
    await buffer.put(p);
    const active=JSON.parse(localStorage.getItem('imu-active')||'null');
    if(active?.session_id===p.session_id)localStorage.setItem('imu-active',JSON.stringify({session_id:p.session_id,last_sequence:p.packet_sequence_number}));
  }).then(()=>pump()).catch(e=>{
    storageOK=false;showError('Buffer write failed. Recording halted; download recovery packets: '+e.message);
    if(session){ends[session.session_id]=seq-1;session=null;saveEnds();send({type:'stop_request'});}
  });
}
async function pump(){
  if(pumping||!welcomed||ws?.readyState!==WebSocket.OPEN||!storageOK)return;
  pumping=true;
  try{
    const records=await buffer.all();records.sort((a,b)=>a.packet.session_id.localeCompare(b.packet.session_id)||a.packet.packet_sequence_number-b.packet.packet_sequence_number);
    for(const {packet:p} of records){
      if(ws.bufferedAmount>128*1024)break;
      const k=key(p),sent=inflight.get(k);
      if(sent!==undefined&&clock()-sent<2)continue;
      if(sent===undefined&&inflight.size>=4)break;
      if(p.device_id!==device)continue;
      send(p);inflight.set(k,clock());
    }
    for(const [id,last] of Object.entries(ends)){
      if(!records.some(r=>r.packet.session_id===id))send({type:'ended',session_id:id,last_sequence:last});
    }
  }catch(e){showError(e);}finally{pumping=false;}
}
function finish(){
  if(!session)return;
  queueBatch();ends[session.session_id]=seq-1;saveEnds();session=null;localStorage.removeItem('imu-active');
  $('stop').disabled=true;writing.then(()=>pump());
}
function motion(event){
  try{
    const s=motionSample(event,model);if(!s)return;
    s.event_index=eventIndex++;
    s.timestamp_order=timestampOrder(s.phone_timestamp,times.at(-1));
    if(s.timestamp_order!=='increasing'){
      timestampWarnings++;
      showWarning(`${timestampWarnings} repeated or out-of-order browser timestamps. Recording continues; original timestamps are preserved and flagged in the CSV.`);
    }
    lastSensor=clock();times.push(s.phone_timestamp);times=times.slice(-180);
    history.push(s);history=history.slice(-150);preview.push(s);preview=preview.slice(-10);
    if(session&&model&&storageOK){
      if(session.stop!==null&&s.estimated_computer_timestamp>=session.stop){finish();return;}
      batch.push(s);if(batch.length>=10)queueBatch();
    }
  }catch(e){showError(e);send({type:'stop_request'});}
}
$('enable').onclick=async()=>{
  try{
    // Invoke permission directly inside the user gesture, before awaiting storage or networking.
    if(!isSecureContext)throw new Error('Trust the local HTTPS certificate first, then open this page in Safari.');
    if(!window.DeviceMotionEvent)throw new Error('DeviceMotionEvent is unavailable in this browser');
    const permission=typeof DeviceMotionEvent.requestPermission==='function'?DeviceMotionEvent.requestPermission():Promise.resolve('granted');
    if(await permission!=='granted')throw new Error('Motion permission denied. Enable Motion & Orientation Access in Safari settings and retry.');
    await initialization;if(!storageOK)throw new Error('Local buffering is unavailable. Use regular Safari, with free storage.');
    error='';$('error').textContent='';enabled=true;device=$('device').value;localStorage.setItem('imu-device',device);
    $('device').disabled=true;$('enable').disabled=true;
    window.addEventListener('devicemotion',motion);await wake();
    navigator.storage?.persist?.().catch(()=>{});connect();
    setTimeout(()=>{if(clock()-lastSensor>2)showWarning('Waiting for complete acceleration + rotation events. Check Safari motion permission and keep this page visible.');},6000);
  }catch(e){showError(e);}
};
$('stop').onclick=()=>send({type:'stop_request'});
$('recover').onclick=async()=>{
  await initialization;const records=new Map();
  for(const {packet} of await buffer.all())records.set(key(packet),packet);
  for(const [k,p] of pending)records.set(k,p);
  const data={exported_at:new Date().toISOString(),client_id:clientId,ends,packets:[...records.values()]};
  const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=u;a.download=`${device}-recovery.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
};
document.addEventListener('visibilitychange',()=>{
  if(document.hidden&&session)showWarning('Safari was hidden during recording; sensor delivery may have stopped. This session may contain a gap.');
  if(!document.hidden&&enabled)wake();status();
});
window.addEventListener('beforeunload',e=>{if(session||pending.size){e.preventDefault();e.returnValue='';}});
setInterval(()=>{
  if(session&&session.stop!==null&&model&&clock()-model.offset>=session.stop+0.2)finish();
  queueBatch();
  if(welcomed&&preview.length&&clock()-lastPreview>=0.08){send({type:'preview',samples:preview});preview=[];lastPreview=clock();}
},80);
setInterval(()=>{status();pump();
  $('sensor').textContent=clock()-lastSensor<1.2?'Accelerometer + gyroscope streaming':'Waiting for fresh sensor events';
  $('sync').textContent=model?`Clock offset ${(model.offset*1000).toFixed(2)} ms · estimated uncertainty ±${(model.uncertainty*1000).toFixed(2)} ms`:'Clock synchronizing…';
  const hz=times.length>1?(times.length-1)/(times.at(-1)-times[0]):0;
  $('rate').textContent=`Actual event rate: ${hz.toFixed(1)} Hz · target 100 Hz (browser controlled)`;
  $('buffer').textContent=`Unacknowledged packets: ${pending.size}`;
  $('recording').textContent=session?(model&&clock()-model.offset<session.start?'Scheduled start…':'Recording '+session.session_id.slice(0,20)):'Waiting for the computer';
  plot($('plot'),history,['ax','ay','az']);
},500);
