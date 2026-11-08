import {pairingToken,plot,download} from './shared.js';
const $=id=>document.getElementById(id),token=pairingToken(),histories=new Map();let ws,current;
const labels=['Phone 1','Phone 2'];
let sessionSignature='',lastSession=null;
function renderExports(){
  const s=current?.sessions?.find(s=>s.session_id===$('session-list').value);
  $('export-status').textContent=s?.complete?'Both phone buffers acknowledged. Session saved — download now or start another recording.':s?'This session is recording or waiting for buffered data.':'Each Start creates a separate recording. Stop saves it; then you can start again.';
  for(const b of $('exports').children)b.disabled=!s?.complete;
}
for(let i=0;i<labels.length;i++){
  const id=`phone_${i+1}`,card=document.createElement('section');card.className='card';
  card.innerHTML=`<h2>${labels[i]} <span id="${id}-badge" class="badge">Disconnected</span></h2><p class="status-line" id="${id}-status">Waiting for phone</p>
    <div class="metrics">${[['hz','Actual Hz'],['offset','Offset / ms'],['uncertainty','Uncertainty / ms'],['packets','Packets / s'],['missing','Missing packets'],['jitter','Interval SD / ms']].map(([k,l])=>`<div class="metric"><small>${l}</small><strong id="${id}-${k}">—</strong></div>`).join('')}</div>
    <canvas id="${id}-accel" width="600" height="140" aria-label="${labels[i]} acceleration"></canvas><p class="legend">Acceleration · X / Y / Z · m/s²</p>
    <canvas id="${id}-gyro" width="600" height="140" aria-label="${labels[i]} rotation rate"></canvas><p class="legend">Gyroscope · X / Y / Z · deg/s</p>`;
  $('devices').append(card);histories.set(id,[]);
}
function connect(){
  ws=new WebSocket(`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/ws`);
  ws.onopen=()=>{ws.send(JSON.stringify({type:'hello',role:'dashboard',protocol_version:1,token}));$('connection').textContent='Receiver connected';};
  ws.onclose=e=>{$('connection').textContent='Receiver disconnected';$('start').disabled=true;$('stop').disabled=true;
    if(e.code===1008){$('error').textContent='Pairing required. Open the full dashboard link from the receiver console or setup page.';}else setTimeout(connect,1000);};
  ws.onmessage=e=>{
    const state=JSON.parse(e.data);if(state.type==='error'){$('error').textContent=state.message;return;}if(state.type!=='state')return;
    current=state;$('start').disabled=!state.can_start;$('stop').disabled=!state.session||state.session.stop!==null;
    const reasons=state.start_reasons||[];
    $('start-help').textContent=state.can_start?'Ready to record. Press Start; recording begins after a 3-second countdown.':reasons.join('\n');
    $('start-help').style.whiteSpace='pre-line';
    $('start').title=reasons.join('\n');$('start').textContent=state.session?.stop!==null&&state.session?'Start new session':'Start recording';
    if(state.error)$('error').textContent=state.error;
    const s=state.session;$('session').textContent=s?.session_id||'No recording yet';
    $('duration').textContent=s?(state.host_time<s.start&&s.stop===null?`Starting in ${(s.start-state.host_time).toFixed(1)} s`:
      `${s.complete?'Saved':s.stop!==null?'Saving':s.state} · ${Math.max(0,(s.stop??state.host_time)-s.start).toFixed(1)} s${s.stop!==null&&!s.complete?' · waiting for final phone packets':''}`):'Not recording — live plots are a preview';
    for(const d of state.devices){
      const badge=$(d.id+'-badge');badge.textContent=d.ready?'Ready':d.connected?'Connected':'Disconnected';badge.className='badge '+(d.ready?'ready':d.connected?'warn':'');
      $(d.id+'-status').textContent=d.not_ready_reason||d.warning||`${d.sensors?'Accelerometer + gyroscope available':'Waiting for sensors'} · ${d.streaming?'streaming':'no fresh data'} · ${d.buffered} buffered`;
      const fields={hz:d.stats.hz.toFixed(1),offset:d.model?(d.model.offset*1000).toFixed(2):'—',uncertainty:d.model?'±'+(d.model.uncertainty*1000).toFixed(2):'—',packets:d.packet_rate.toFixed(1),missing:d.missing,jitter:d.stats.jitter_ms.toFixed(2)};
      for(const [k,v] of Object.entries(fields))$(d.id+'-'+k).textContent=v;
      let history=histories.get(d.id);for(const sample of d.recent){if(!history.length||sample.phone_timestamp>history.at(-1).phone_timestamp)history.push(sample);}history=history.slice(-180);histories.set(d.id,history);
      plot($(d.id+'-accel'),history,['ax','ay','az']);plot($(d.id+'-gyro'),history,['gx','gy','gz']);
    }
    const signature=JSON.stringify(state.sessions);
    if(signature!==sessionSignature){
      const selected=$('session-list').value;
      $('session-list').replaceChildren();
      for(const [i,recording] of (state.sessions||[]).entries()){
        const option=document.createElement('option');option.value=recording.session_id;
        option.textContent=`Session ${i+1} — ${recording.complete?'Saved':recording.stop===null?'Recording':'Saving'} — ${recording.session_id}`;
        $('session-list').append(option);
      }
      $('session-list').disabled=!(state.sessions||[]).length;
      if(lastSession!==s?.session_id)$('session-list').value=s?.session_id||'';
      else if(selected)$('session-list').value=selected;
      lastSession=s?.session_id;sessionSignature=signature;
    }
    renderExports();
  };
}
$('session-list').onchange=renderExports;
$('start').onclick=()=>{$('error').textContent='';$('start').disabled=true;$('start-help').textContent='Preparing recording…';ws.send(JSON.stringify({type:'start'}));};
$('stop').onclick=()=>ws.send(JSON.stringify({type:'stop'}));
for(const name of ['combined.csv','phone_1.csv','phone_2.csv','metadata.json']){
  const b=document.createElement('button');b.textContent=name;b.className='secondary';b.disabled=true;
  b.onclick=async()=>{try{const id=$('session-list').value;await download(`/export/${id}/${name}`,token,`${id}_${name}`);}catch(e){$('error').textContent=e.message;}};$('exports').append(b);
}
fetch('/api/links',{headers:{Authorization:`Bearer ${token}`}}).then(async r=>{if(!r.ok)throw new Error('Use the paired dashboard link from the console');return r.json();}).then(links=>{
  $('setup').href=links.setupURL;$('setup').textContent=links.cloud?'Open phone connection page':links.setupURL;
  if(links.cloud)$('setup').previousElementSibling.textContent='Open this private setup link in Safari on each iPhone. Wi-Fi or cellular both work; bookmark it for next time.';
}).catch(e=>{$('error').textContent=e.message;});connect();
