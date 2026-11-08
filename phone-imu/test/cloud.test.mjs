import test from 'node:test';
import assert from 'node:assert/strict';
import {once} from 'node:events';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {WebSocket} from 'ws';
import {SimulatedPhone,sleep,until} from '../scripts/simulated-phone.mjs';
import {cloudConfig} from '../lib/cloud-config.mjs';

test('cloud startup requires a strong pairing secret and explicit recording storage',()=>{
  assert.throws(()=>cloudConfig({}),/Set IMU_TOKEN/);
  assert.throws(()=>cloudConfig({IMU_TOKEN:'short'}),/Set IMU_TOKEN/);
  assert.throws(()=>cloudConfig({IMU_TOKEN:'t'.repeat(48)}),/Set IMU_RECORDINGS/);
});

test('hosted server pairs privately, synchronizes two phones and exports authenticated CSV', {timeout:25000},async()=>{
  const root=await mkdtemp(path.join(tmpdir(),'imu-cloud-test-')),token='cloud-test-secret-'.repeat(3);
  const env={IMU_CLOUD:'1',PORT:'0',IMU_TOKEN:token,IMU_RECORDINGS:root},previous={};
  for(const [key,value] of Object.entries(env)){previous[key]=process.env[key];process.env[key]=value;}
  let app,dash;const phones=[];
  try{
    app=await import('../server.mjs');
    const port=app.server.address().port,base=`http://127.0.0.1:${port}`,url=`ws://127.0.0.1:${port}/ws`;
    assert.deepEqual(await (await fetch(base+'/health')).json(),{ok:true});
    for(const route of ['/','/setup','/dashboard','/phone','/connect.js']){
      const response=await fetch(base+route);assert.equal(response.status,200);
      const body=await response.text();assert.ok(!body.includes(token));assert.ok(!body.includes('BEGIN CERTIFICATE'));
    }
    assert.equal((await fetch(base+'/imu-ca.cer')).status,404);
    assert.equal((await fetch(base+'/api/links')).status,401);
    assert.equal((await fetch(base+'/export/session_123/combined.csv')).status,401);
    const links=await (await fetch(base+'/api/links',{headers:{Authorization:`Bearer ${token}`}})).json();
    assert.equal(links.cloud,true);assert.equal(links.phoneURL,`/phone#token=${encodeURIComponent(token)}`);
    const unauthorized=new WebSocket(url);
    await once(unauthorized,'open');unauthorized.send(JSON.stringify({type:'hello',role:'dashboard',protocol_version:1,token:'wrong'}));
    assert.equal((await once(unauthorized,'close'))[0],1008);
    let state;dash=new WebSocket(url);dash.on('message',raw=>{const m=JSON.parse(raw);if(m.type==='state')state=m;});
    await once(dash,'open');dash.send(JSON.stringify({type:'hello',role:'dashboard',protocol_version:1,token}));
    phones.push(...[1,2].map(n=>new SimulatedPhone(url,token,`phone_${n}`,{offset:n*25}).start()));
    await until(()=>state?.can_start,10000);
    dash.send(JSON.stringify({type:'start'}));
    await until(()=>state?.session&&state.host_time>state.session.start+0.5,7000);
    await phones[1].disconnect(350);await sleep(600);
    dash.send(JSON.stringify({type:'stop'}));await until(()=>state?.session?.complete,7000);
    const result=await fetch(`${base}/export/${state.session.session_id}/combined.csv`,{headers:{Authorization:`Bearer ${token}`}});
    assert.equal(result.status,200);const csv=await result.text();
    assert.match(csv,/phone_1/);assert.match(csv,/phone_2/);assert.ok(csv.trim().split('\n').length>100);
    assert.deepEqual(phones.flatMap(p=>p.errors),[]);
  }finally{
    phones.forEach(p=>p.close());dash?.terminate();await app?.shutdown();
    for(const [key,value] of Object.entries(previous)){if(value===undefined)delete process.env[key];else process.env[key]=value;}
    const resolved=path.resolve(root),parent=path.resolve(tmpdir());
    assert.equal(path.dirname(resolved),parent);assert.ok(path.basename(resolved).startsWith('imu-cloud-test-'));
    await rm(resolved,{recursive:true,force:true});
  }
});
