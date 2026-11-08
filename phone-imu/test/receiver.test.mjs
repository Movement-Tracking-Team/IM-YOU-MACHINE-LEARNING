import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {WebSocket} from 'ws';
import {Receiver} from '../lib/receiver.mjs';
import {exportRecording} from '../lib/storage.mjs';
import {rateStats,SequenceTracker} from '../lib/protocol.mjs';
import {SimulatedPhone,sleep,until} from '../scripts/simulated-phone.mjs';
test('two clients: readiness, shared window, disconnect replay, dedupe, CSV export', {timeout:20000},async()=>{
  const root=await mkdtemp(path.join(tmpdir(),'imu-test-')),server=http.createServer();
  const r=new Receiver(server,{token:'test-token',root,probeInterval:5,syncInterval:800,startDelay:0.8});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const url=`ws://127.0.0.1:${server.address().port}/ws`;
  const phones=[1,2].map(n=>new SimulatedPhone(url,'test-token',`phone_${n}`,{offset:n*25}).start());
  try{
    await assert.rejects(()=>r.start(),/Both phones/);await until(()=>r.snapshot().can_start);
    const metadata=await r.start();await sleep(1250);
    for(const d of r.devices.values())assert.ok(rateStats(d.times).hz>95&&rateStats(d.times).hz<105);
    const duplicate=phones[0].lastPacket;
    phones[0].send(duplicate);await sleep(80);
    assert.ok(r.session.trackers.get('phone_1').duplicates>=1);
    await phones[1].disconnect(500);await sleep(1000);
    await r.stop();await until(()=>r.complete());
    assert.deepEqual(phones.flatMap(p=>p.errors),[]);
    for(const t of r.session.trackers.values())assert.equal(t.missing,0);
    await r.session.chain;await r.session.metadataChain;const result=await exportRecording(r.session.dir);assert.ok(result.rows>200);
    const csv=await readFile(path.join(r.session.dir,'combined.csv'),'utf8');const lines=csv.trim().split('\n'),headers=lines.shift().split(',');
    const rows=lines.map(line=>Object.fromEntries(line.split(',').map((v,i)=>[headers[i],v])));
    assert.equal(new Set(rows.map(row=>row.device_id)).size,2);
    let previous=-Infinity;for(const row of rows){
      const t=+row.common_timestamp;assert.ok(t>=metadata.start&&t<metadata.stop);assert.ok(t>=previous);previous=t;
      assert.ok(Math.abs(+row.phone_timestamp-+row.offset_s-t)<1e-10);
      assert.equal(row.accel_units,'m/s^2');assert.equal(row.gyro_units,'deg/s');
    }
    assert.equal(new Set(rows.map(x=>`${x.device_id}:${x.packet_sequence_number}:${x.phone_timestamp}`)).size,rows.length);
    // Same connections can record repeatedly; earlier sessions stay separately downloadable.
    const original=r.session,originalCSV=csv;
    for(let n=0;n<2;n++){
      await until(()=>r.snapshot().can_start);const next=await r.start();
      assert.notEqual(next.session_id,metadata.session_id);await sleep(1100);
      await r.stop();await until(()=>r.complete());
      await r.session.metadataChain;const exported=await exportRecording(r.session.dir);assert.ok(exported.rows>20);
      assert.ok(r.session.trackers.get('phone_1').next>0);
    }
    assert.equal(r.snapshot().sessions.length,3);assert.ok(r.snapshot().sessions.every(s=>s.complete));
    assert.equal(await readFile(path.join(original.dir,'combined.csv'),'utf8'),originalCSV);
  }finally{phones.forEach(p=>p.close());await r.close();await new Promise(resolve=>server.close(resolve));await rm(root,{recursive:true,force:true});}
});
test('start blockers identify the missing phone and a saving session',()=>{
  const server=http.createServer(),r=new Receiver(server,{token:'secret',root:tmpdir()});
  assert.match(r.snapshot().start_reasons.join(' '),/Phone 2: disconnected/);
  assert.equal(r.snapshot().can_start,false);
  return r.close();
});
test('reconnect replay racing a pending disk write stores the sequence once',async()=>{
  const server=http.createServer(),r=new Receiver(server,{token:'secret',root:tmpdir()});
  const tracker=new SequenceTracker(),writes=[];
  const s={metadata:{devices:[{device_id:'phone_1',client_id:'client'}]},trackers:new Map([['phone_1',tracker]]),
    append:async p=>{await sleep(15);writes.push(p);}};
  r.sessions.set('session_test',s);r.models.set('phone_1:boot:1',{offset:10,uncertainty:0.001});
  const d={id:'phone_1',client_id:'client',ws:null,times:[],packetTimes:[]};
  const sample={phone_timestamp:12,original_device_timestamp_ms:12000,handler_timestamp:12.001,
    estimated_computer_timestamp:2,sync_id:1,ax:1,ay:2,az:3,gx:4,gy:5,gz:6,rotation_alpha:6,rotation_beta:4,rotation_gamma:5};
  const p={type:'samples',protocol_version:1,session_id:'session_test',device_id:'phone_1',boot_id:'boot',
    packet_sequence_number:0,accel_units:'m/s^2',gyro_units:'deg/s',samples:[sample]};
  try{await Promise.all([r.accept(d,p,3),r.accept(d,p,3.1)]);assert.equal(writes.length,1);assert.equal(tracker.duplicates,1);}
  finally{r.sessions.clear();await r.close();}
});
test('overlapping preview and persisted samples do not undercount sampling frequency',()=>{
  const d={times:[],packetTimes:[],recent:[]};
  const sample=t=>({phone_timestamp:t,ax:1,ay:2,az:3,gx:4,gy:5,gz:6});
  Receiver.prototype.observe(d,[0,0.01,0.02,0.03].map(sample),1,false);
  Receiver.prototype.observe(d,[0.02,0.03,0.04,0.05].map(sample),2,true);
  Receiver.prototype.observe(d,[0.03,0.04].map(sample),3,true);
  assert.equal(d.times.length,6);assert.ok(Math.abs(rateStats(d.times).hz-100)<1e-9);
});
test('a dropped connection before the arm deadline cancels the shared window', {timeout:10000},async()=>{
  const root=await mkdtemp(path.join(tmpdir(),'imu-cancel-')),server=http.createServer();
  const r=new Receiver(server,{token:'test-token',root,probeInterval:5,startDelay:0.8});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const phones=[1,2].map(n=>new SimulatedPhone(`ws://127.0.0.1:${server.address().port}/ws`,'test-token',`phone_${n}`).start());
  try{
    await until(()=>r.snapshot().can_start);await r.start();phones[1].close();await sleep(900);
    assert.equal(r.session.metadata.stop,r.session.metadata.start);assert.match(r.session.metadata.state,/cancelled/);
    assert.equal(r.complete(),false);
  }finally{phones.forEach(p=>p.close());await r.close();await new Promise(resolve=>server.close(resolve));await rm(root,{recursive:true,force:true});}
});
test('registration rejects an unauthorized client',async()=>{
  const server=http.createServer(),r=new Receiver(server,{token:'secret',root:tmpdir()});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const ws=new WebSocket(`ws://127.0.0.1:${server.address().port}/ws`);
  ws.on('open',()=>ws.send(JSON.stringify({type:'hello',role:'dashboard',token:'wrong',protocol_version:1})));
  const code=await new Promise(resolve=>ws.on('close',resolve));assert.equal(code,1008);
  await r.close();await new Promise(resolve=>server.close(resolve));
});
