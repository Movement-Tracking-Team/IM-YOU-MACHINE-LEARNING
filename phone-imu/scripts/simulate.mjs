import {readFile,writeFile} from 'node:fs/promises';
import https from 'node:https';
import {WebSocket} from 'ws';
import {SimulatedPhone,sleep,until} from './simulated-phone.mjs';
const config=JSON.parse(await readFile(new URL('../.local/config.json',import.meta.url),'utf8'));
const ca=await readFile(new URL('../.local/ca.pem',import.meta.url));
const url=`wss://localhost:${config.httpsPort}/ws`;
const phones=[1,2].map((n)=>new SimulatedPhone(url,config.token,`phone_${n}`,{ca,offset:n*57,drift:(n-2)*0.00003}).start());
const dash=new WebSocket(url,{ca});let state;
dash.on('open',()=>dash.send(JSON.stringify({type:'hello',role:'dashboard',protocol_version:1,token:config.token})));
dash.on('message',raw=>{const m=JSON.parse(raw);if(m.type==='state')state=m;else if(m.type==='error')console.error(m.message);});
try{
  await until(()=>state?.can_start,12000);
  dash.send(JSON.stringify({type:'start'}));await until(()=>state?.session?.stop===null);
  await until(()=>state.host_time>state.session.start+1,7000);
  console.log('Two simulated phones recording; interrupting Phone 2 for 1.2 s.');
  await phones[1].disconnect(1200);
  await sleep(1800);await until(()=>phones[1].pending.size===0);
  dash.send(JSON.stringify({type:'stop'}));await until(()=>state?.session?.complete,15000);
  const session=state.session.session_id;
  const csv=await new Promise((resolve,reject)=>{
    https.get(`https://localhost:${config.httpsPort}/export/${session}/combined.csv`,{ca,headers:{Authorization:`Bearer ${config.token}`}},res=>{
      let body='';res.on('data',d=>body+=d);res.on('end',()=>res.statusCode===200?resolve(body):reject(new Error(body)));
    }).on('error',reject);
  });
  const rows=csv.trim().split('\n').length-1;
  const report={simulated:true,session,rows,duration_s:state.session.stop-state.session.start,
    devices:state.devices.map(d=>({id:d.id,hz:d.stats.hz,interval_sd_ms:d.stats.jitter_ms,max_interval_ms:d.stats.max_interval_ms,offset_ms:d.model.offset*1000,uncertainty_ms:d.model.uncertainty*1000,
      missing:d.missing,buffered:d.buffered})),errors:phones.flatMap(p=>p.errors)};
  await writeFile(new URL('../.local/simulation-result.json',import.meta.url),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  if(report.errors.length||report.devices.some(d=>d.missing||d.buffered||d.hz<95||d.hz>105)||rows<600)throw new Error('Simulation validation failed');
}finally{phones.forEach(p=>p.close());dash.terminate();}
