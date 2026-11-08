import http from 'node:http';
import https from 'node:https';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {WebSocket} from 'ws';
const c=JSON.parse(await readFile(new URL('../.local/config.json',import.meta.url),'utf8'));
const ca=await readFile(new URL('../.local/ca.pem',import.meta.url));
async function get(url,secure=false){
  return new Promise((resolve,reject)=>{
    (secure?https:http).get(url,secure?{ca}:{},r=>{let body='';r.on('data',d=>body+=d);r.on('end',()=>resolve({status:r.statusCode,body}));}).on('error',reject);
  });
}
const page=await get(`http://127.0.0.1:${c.setupPort}/dashboard`);
assert.equal(page.status,200);assert.match(page.body,/Two-phone recording/);
console.log('Loopback HTTP dashboard: 200');
await new Promise((resolve,reject)=>{
  const ws=new WebSocket(`ws://127.0.0.1:${c.setupPort}/ws`),timer=setTimeout(()=>{ws.terminate();reject(new Error('Dashboard timed out'));},5000);
  ws.on('open',()=>ws.send(JSON.stringify({type:'hello',role:'dashboard',protocol_version:1,token:c.token})));
  ws.on('message',raw=>{const s=JSON.parse(raw);assert.equal(s.type,'state');clearTimeout(timer);console.log('Loopback WebSocket: live state',s.devices.map(d=>({id:d.id,connected:d.connected,ready:d.ready})));ws.close();resolve();});
  ws.on('error',reject);
});
await new Promise((resolve,reject)=>{
  const ws=new WebSocket(`ws://${c.ip}:${c.setupPort}/ws`);
  ws.on('unexpected-response',(_request,response)=>{assert.equal(response.statusCode,403);response.resume();ws.terminate();console.log('Plain LAN WebSocket: correctly blocked (403)');resolve();});
  ws.on('error',()=>{});ws.on('open',()=>{ws.close();reject(new Error('Unexpected plaintext LAN access'));});
});
const health=await get(`https://${c.ip}:${c.httpsPort}/health`,true);assert.equal(health.status,200);
console.log('CA-verified hotspot HTTPS health: 200');
