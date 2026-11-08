import https from 'node:https';
import http from 'node:http';
import {readFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Worker} from 'node:worker_threads';
import {WebSocketServer} from 'ws';
import {Receiver} from './lib/receiver.mjs';
import {cloudConfig} from './lib/cloud-config.mjs';
const root=path.dirname(fileURLToPath(import.meta.url));
const cloud=process.env.IMU_CLOUD==='1';
const config=cloud?cloudConfig(process.env):JSON.parse(await readFile(process.env.IMU_CONFIG||path.join(root,'.local/config.json'),'utf8'));
const recordings=process.env.IMU_RECORDINGS||path.join(root,'recordings');await mkdir(recordings,{recursive:true});
const phoneURL=cloud?`/phone#token=${encodeURIComponent(config.token)}`:`https://${config.ip}:${config.httpsPort}/phone#token=${config.token}`;
const dashURL=cloud?'/dashboard':`http://localhost:${config.setupPort}/dashboard#token=${config.token}`;
const setupURL=cloud?`/setup#token=${encodeURIComponent(config.token)}`:`http://${config.ip}:${config.setupPort}`;
const setup=`<!doctype html><html lang="en"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Connect your iPhones</title>
<style>body{font:18px system-ui;max-width:720px;margin:40px auto;padding:20px;line-height:1.6;color:#172a38;background:#f2f6f9}a{color:#005fc5}code{overflow-wrap:anywhere}li{margin:16px 0}img{width:200px}</style>
<h1>Connect two iPhones</h1><p>Keep the computer and phones on the same Wi-Fi. Complete these steps once on each iPhone in Safari.</p>
<ol><li><a href="/imu.mobileconfig">Download the local HTTPS certificate profile</a>. Tap Allow.</li>
<li>Open Settings → General → VPN &amp; Device Management → Three Phone IMU HTTPS → Install. Enter your passcode if requested.</li>
<li>Open Settings → General → About → Certificate Trust Settings. Enable full trust for <b>Three Phone IMU Local CA</b>.</li>
<li>Return here and <a href="${phoneURL}">open the phone recorder</a>. Select a different phone number on each device, tap Enable sensors, and Allow motion access.</li></ol>
<p><a href="${dashURL}">Computer dashboard (open on this computer)</a> · <a href="/imu-ca.cer">CA certificate</a></p>
<p>Open on every phone: <a href="${setupURL}">${setupURL}</a></p><p>Certificate SHA-256: <code>${config.fingerprint}</code></p>
<p>This profile only installs this computer’s local development CA. Compare its fingerprint with the receiver console before trusting it. Remove it in Settings when testing is finished. A browser certificate warning bypass alone may not enable motion sensors.</p>
<p>If this page does not load on the phones, the Wi-Fi may isolate devices. Try a private router or computer hotspot and rerun setup with its IPv4 address.</p></html>`;
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.csv':'text/csv; charset=utf-8','.json':'application/json'};
const staticFiles=new Set(['index.html','phone.html','connect.html','connect.js','app.css','dashboard.js','phone.js','shared.js','buffer.js']);
function send(res,status,type,body){res.writeHead(status,{'Content-Type':type,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'});res.end(body);}
const activeExports=new Map();
async function handler(req,res){
  try {
    const url=new URL(req.url,'https://localhost');
    if(req.method!=='GET'){send(res,405,'text/plain','GET only');return;}
    if(url.pathname==='/health'){send(res,200,'application/json',JSON.stringify(cloud?{ok:true}:{ok:true,server_id:receiver.serverId,phones:receiver.snapshot().devices.map(d=>({id:d.id,connected:d.connected}))}));return;}
    if(url.pathname==='/setup'&&!cloud){send(res,200,'text/html',setup);return;}
    if(url.pathname==='/api/links'){
      if(req.headers.authorization!==`Bearer ${config.token}`){send(res,401,'text/plain','Pairing required');return;}
      send(res,200,'application/json',JSON.stringify({phoneURL,setupURL,cloud}));return;
    }
    const match=url.pathname.match(/^\/export\/(session_[a-f0-9-]+)\/(combined\.csv|phone_[123]\.csv|metadata\.json)$/);
    if(match){
      if(req.headers.authorization!==`Bearer ${config.token}`){send(res,401,'text/plain','Pairing required');return;}
      const s=receiver.sessions.get(match[1]);
      if(!s || !receiver.complete(s)){send(res,409,'text/plain','Stop and wait for both phone buffers to drain before exporting');return;}
      await s.chain;await s.syncWrites;await s.metadataChain;
      if(!activeExports.has(s.dir))activeExports.set(s.dir,new Promise((resolve,reject)=>{
        const w=new Worker(new URL('./lib/export-worker.mjs',import.meta.url),{workerData:s.dir});
        w.once('message',m=>m.error?reject(new Error(m.error)):resolve(m));w.once('error',reject);
        w.once('exit',code=>{if(code)reject(new Error('Export worker exited '+code));});
      }).finally(()=>activeExports.delete(s.dir)));
      await activeExports.get(s.dir);
      res.setHeader('Content-Disposition',`attachment; filename="${match[1]}_${match[2]}"`);
      send(res,200,mime[path.extname(match[2])],await readFile(path.join(s.dir,match[2])));return;
    }
    const name=cloud&&['/','/setup'].includes(url.pathname)?'connect.html':['/','/dashboard'].includes(url.pathname)?'index.html':url.pathname==='/phone'?'phone.html':url.pathname.slice(1);
    if(!staticFiles.has(name)){send(res,404,'text/plain','Not found');return;}
    res.setHeader('Permissions-Policy','accelerometer=(self), gyroscope=(self)');
    res.setHeader('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    send(res,200,mime[path.extname(name)],await readFile(path.join(root,'public',name)));
  }catch(e){console.error(e);if(!res.headersSent)send(res,500,'text/plain',e.message);else res.end();}
}
// Cloud hosting terminates HTTPS at its proxy; a single Node process owns all recordings.
const server=cloud?http.createServer(handler):https.createServer({key:await readFile(path.join(root,'.local/server-key.pem')),cert:await readFile(path.join(root,'.local/server.pem'))},handler);
const receiver=new Receiver(server,{token:config.token,root:recordings});
const isLoopback=req=>['127.0.0.1','::1','::ffff:127.0.0.1'].includes(req.socket.remoteAddress);
const bootstrap=http.createServer(async(req,res)=>{
  try{
    if(req.url==='/imu.mobileconfig')send(res,200,'application/x-apple-aspen-config',await readFile(path.join(root,'.local/imu.mobileconfig')));
    else if(req.url==='/imu-ca.cer')send(res,200,'application/pkix-cert',await readFile(path.join(root,'.local/imu-ca.cer')));
    else if(isLoopback(req)&&!['/','/setup'].includes(new URL(req.url,'http://localhost').pathname))await handler(req,res);
    else send(res,200,'text/html; charset=utf-8',setup);
  }catch{send(res,500,'text/plain','Certificate unavailable; rerun npm run setup');}
});
// Only the computer's loopback interface can use the certificate-free dashboard WebSocket.
// LAN phone acquisition remains exclusively WSS.
const localHub=new WebSocketServer({noServer:true,maxPayload:256*1024,perMessageDeflate:false});
bootstrap.on('upgrade',(req,socket,head)=>{
  if(!isLoopback(req)||req.url!=='/ws'){socket.end('HTTP/1.1 403 Forbidden\r\n\r\n');return;}
  localHub.handleUpgrade(req,socket,head,ws=>receiver.connection(ws));
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(cloud?Number(process.env.PORT||10000):config.httpsPort,'0.0.0.0',resolve);});
if(!cloud)await new Promise((resolve,reject)=>{bootstrap.once('error',reject);bootstrap.listen(config.setupPort,'0.0.0.0',resolve);});
if(cloud)console.log(`Two Phone IMU cloud receiver listening on port ${server.address().port}`);
else console.log(`Two Phone IMU is running\nPhone setup: ${setupURL}\nPhone recorder: ${phoneURL}\nDashboard: ${dashURL}\nData: ${recordings}\nCA SHA-256: ${config.fingerprint}`);
export {server};
export async function shutdown(){
  await receiver.stop('server shutdown');await receiver.close();localHub.close();
  await Promise.all([server,bootstrap].filter(s=>s.listening).map(s=>new Promise(resolve=>s.close(resolve))));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  process.on('SIGINT',()=>shutdown().then(()=>process.exit()));process.on('SIGTERM',()=>shutdown().then(()=>process.exit()));
}
