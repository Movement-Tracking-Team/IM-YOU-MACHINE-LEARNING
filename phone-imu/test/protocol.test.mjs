import test from 'node:test';
import assert from 'node:assert/strict';
import {estimateClock,SequenceTracker,rateStats,validatePacket} from '../lib/protocol.mjs';
import {normalizeTimestamp,motionSample,timestampOrder} from '../public/shared.js';
test('NTP sign and low RTT preference reject delayed exchanges',()=>{
  const exchanges=Array.from({length:25},(_,i)=>{
    const h1=100+i,delay=i<20?0.002:0.2,offset=37;
    return {h1,p2:h1+delay+offset,p3:h1+delay+offset+0.001,h4:h1+delay+0.001+0.002};
  });
  const m=estimateClock(exchanges);assert.ok(Math.abs(m.offset-37)<1e-9);assert.ok(m.uncertainty>=0.0019);
  assert.ok(Math.abs(150-m.offset-113)<1e-10);
});
test('sequence gaps, out of order repair and duplicate replay',()=>{
  const s=new SequenceTracker();s.add(2);assert.equal(s.missing,2);s.add(0);assert.equal(s.missing,1);
  s.add(1);assert.equal(s.missing,0);assert.equal(s.next,3);assert.equal(s.add(1),false);assert.equal(s.duplicates,1);
});
test('actual sensor rate and interval jitter derive from timestamps',()=>{
  const r=rateStats([1,1.01,1.02,1.03]);assert.ok(Math.abs(r.hz-100)<1e-9);assert.ok(r.jitter_ms<1e-9);
  assert.ok(rateStats([1,1.01,1.04]).max_interval_ms>29);
});
test('event timestamp normalization preserves a monotonic timeline',()=>{
  assert.equal(normalizeTimestamp(1250,1260,1700000000000).timestamp,1.25);
  assert.equal(normalizeTimestamp(1700000001250,1260,1700000000000).timestamp,1.25);
  assert.throws(()=>normalizeTimestamp(0));assert.throws(()=>normalizeTimestamp(10,100000));
});
test('browser raw units and beta/gamma/alpha axis mapping are retained',()=>{
  const s=motionSample({timeStamp:performance.now(),accelerationIncludingGravity:{x:1,y:2,z:9.81},rotationRate:{alpha:10,beta:20,gamma:30},interval:16.67},{id:2,offset:1});
  assert.deepEqual([s.ax,s.ay,s.az,s.gx,s.gy,s.gz],[1,2,9.81,20,30,10]);assert.equal(s.sync_id,2);
  assert.equal(motionSample({accelerationIncludingGravity:null},null),null);
});
test('malformed packet is rejected before touching storage',()=>{
  assert.throws(()=>validatePacket({}));
  assert.throws(()=>validatePacket({type:'samples',protocol_version:1,device_id:'../escape',session_id:'s',packet_sequence_number:0,samples:[]}));
});
test('duplicate and out-of-order browser event timestamps are flagged and preserved',()=>{
  assert.equal(timestampOrder(2,2),'duplicate');assert.equal(timestampOrder(1,2),'out_of_order');
  assert.equal(timestampOrder(3,2),'increasing');
  const samples=[2,2,1.99].map((t,i)=>({phone_timestamp:t,original_device_timestamp_ms:t*1000,
    handler_timestamp:2.1,estimated_computer_timestamp:t-1,sync_id:1,event_index:i,
    timestamp_order:timestampOrder(t,i?[2,2,1.99][i-1]:undefined),ax:0,ay:0,az:9.81,gx:0,gy:0,gz:0,
    rotation_alpha:0,rotation_beta:0,rotation_gamma:0}));
  validatePacket({type:'samples',protocol_version:1,device_id:'phone_1',session_id:'test',packet_sequence_number:0,
    accel_units:'m/s^2',gyro_units:'deg/s',samples});
  assert.deepEqual(samples.map(s=>s.phone_timestamp),[2,2,1.99]);
});
