import {randomUUID} from 'node:crypto';
import {mkdir, appendFile} from 'node:fs/promises';
import path from 'node:path';
import {WebSocketServer, WebSocket} from 'ws';
import {VERSION, DEVICES, now, finite, estimateClock, SequenceTracker, rateStats, validatePacket} from './protocol.mjs';
import {Recording} from './storage.mjs';

export class Receiver {
  constructor(server, {token, root, syncInterval=15000, probeInterval=35, startDelay=3}) {
    Object.assign(this,{token,root,syncInterval,probeInterval,startDelay});
    this.serverId=randomUUID(); this.devices=new Map(); this.viewers=new Set(); this.subscribers=new Set(); this.models=new Map();this.sockets=new Set();
    this.sessions=new Map(); this.acceptChains=new Map(); this.syncSerial=0; this.timers=new Set(); this.session=null; this.error=null;
    this.wss=new WebSocketServer({server,path:'/ws',maxPayload:256*1024,perMessageDeflate:false});
    this.wss.on('connection', ws=>this.connection(ws));
    this.tick=setInterval(()=>this.broadcast(),250);
    this.heartbeat=setInterval(()=>{
      for(const ws of this.sockets){
        if(ws.alive===false){ws.terminate();continue;}
        ws.alive=false;ws.ping();
      }
    },5000);
  }
  send(ws, message) {
    if (ws?.readyState===WebSocket.OPEN && ws.bufferedAmount < 1024*1024) ws.send(JSON.stringify(message));
  }
  later(fn,ms) { const timer=setTimeout(()=>{this.timers.delete(timer);fn();},ms);this.timers.add(timer);return timer; }
  connection(ws) {
    this.sockets.add(ws);
    ws.alive=true;ws.on('pong',()=>{ws.alive=true;});
    const deadline=setTimeout(()=>ws.close(1008,'Registration timeout'),5000);
    let d=null, role=null, pending=Promise.resolve();
    ws.on('message', raw=>{
      const received=now(); // H4: capture before parsing or disk queues.
      let m;
      try { m=JSON.parse(raw); } catch { ws.close(1008,'Invalid JSON');return; }
      if (!role) {
        if (m.type!=='hello' || m.token!==this.token || m.protocol_version!==VERSION) {ws.close(1008,'Unauthorized/version mismatch');return;}
        clearTimeout(deadline);
        if (m.role==='dashboard') {role=m.role;this.viewers.add(ws);this.send(ws,this.snapshot());return;}
        // Live inference needs every sample. The dashboard snapshot carries only the last 10
        // per device every 250 ms, which drops more than half of a 100 Hz stream.
        if (m.role==='subscriber') {
          role=m.role;this.subscribers.add(ws);
          this.send(ws,{type:'subscribed',server_id:this.serverId,units:{acceleration:'m/s^2',rotation:'deg/s'},
            devices:DEVICES,session:this.session?.metadata ?? null});
          return;
        }
        if (m.role!=='phone' || !DEVICES.includes(m.device_id) || !/^[a-zA-Z0-9_-]{8,80}$/.test(m.client_id) ||
            !/^[a-zA-Z0-9_-]{8,80}$/.test(m.boot_id)) {ws.close(1008,'Invalid device identity');return;}
        d=this.devices.get(m.device_id);
        if (d?.ws?.readyState===WebSocket.OPEN || (d && d.client_id!==m.client_id && this.session && !this.complete())) {
          ws.close(1008,'Phone slot already occupied'); return;
        }
        d={...d,id:m.device_id,client_id:m.client_id,boot_id:m.boot_id,source:m.source==='simulated'?'simulated':'browser',ws,connected:true,ready:false,health:null,
          statusAt:0,latestAt:0,times:[],packetTimes:[],recent:[],model:null,pending:new Map(),exchanges:[]};
        role='phone';this.devices.set(d.id,d);
        this.send(ws,{type:'welcome',server_id:this.serverId,session:this.session?.metadata ?? null});
        this.sync(d);
        return;
      }
      if (role==='dashboard') {
        if(m.type==='start') this.start().catch(e=>this.send(ws,{type:'error',message:e.message}));
        if(m.type==='stop') this.stop().catch(e=>this.send(ws,{type:'error',message:e.message}));
        return;
      }
      if (role==='subscriber') return; // Read-only: subscribers never drive the session.
      if (m.type==='sync_reply') {this.syncReply(d,m,received);return;}
      if (m.type==='status') {
        d.health={sensors:m.sensors===true,visible:m.visible===true,error:String(m.error||'').slice(0,250),warning:String(m.warning||'').slice(0,300),buffered:Number(m.buffered)||0};
        d.statusAt=received;d.ready=d.health.sensors&&d.health.visible&&!d.health.error&&!!d.model&&m.sync_id===d.model.id;
        return;
      }
      if (m.type==='preview') {
        if(Array.isArray(m.samples) && m.samples.length<=20) this.observe(d,m.samples,received,false);
        return;
      }
      if (m.type==='armed' && m.session_id===this.session?.metadata.session_id) {
        if(d.ready) this.session.armed.add(d.id); return;
      }
      if(m.type==='stop_request') {this.stop().catch(e=>this.send(ws,{type:'error',message:e.message}));return;}
      // Serialize disk acceptance for this connection. Socket reads/sync remain unblocked.
      if(m.type==='samples' || m.type==='ended') {
        d.queued=(d.queued||0)+1;
        if(d.queued>256) {ws.close(1008,'Receiver backpressure: reconnect to replay');return;}
        pending=pending.then(async()=>{
          if(m.type==='samples') await this.accept(d,m,received);
          else await this.ended(d,m);
        }).catch(e=>{this.error=e.message;this.send(ws,{type:'error',message:e.message});})
          .finally(()=>{d.queued--;});
      }
    });
    ws.on('error',()=>{});
    ws.on('close',()=>{clearTimeout(deadline);this.sockets.delete(ws);this.viewers.delete(ws);this.subscribers.delete(ws);if(d && this.devices.get(d.id)===d){d.connected=false;d.ready=false;}});
  }
  sync(d) {
    if(!d.connected || this.devices.get(d.id)!==d)return;
    d.exchanges=[];d.pending.clear();let n=0;
    const probe=()=>{
      if(!d.connected)return;
      if(n++===25) {
        this.later(()=>this.finishSync(d),350);
        this.later(()=>this.sync(d),this.syncInterval); return;
      }
      const id=randomUUID(),h1=now();d.pending.set(id,h1);
      this.send(d.ws,{type:'sync_probe',id,h1});this.later(probe,this.probeInterval);
    };probe();
  }
  syncReply(d,m,h4) {
    const h1=d.pending.get(m.id); if(h1===undefined || !finite(m.p2)||!finite(m.p3))return;
    d.pending.delete(m.id);d.exchanges.push({h1,p2:m.p2,p3:m.p3,h4});
  }
  finishSync(d) {
    if(!d.connected)return;
    try {
      const model={...estimateClock(d.exchanges),id:++this.syncSerial,host_time:now(),device_id:d.id,boot_id:d.boot_id};
      d.model=model;d.ready=false;this.models.set(`${d.id}:${d.boot_id}:${model.id}`,model);
      this.send(d.ws,{type:'sync_model',...model});
      if(this.session) this.logSync(model,d.exchanges);
    }catch {d.ready=false;}
  }
  logSync(model,exchanges) {
    const s=this.session;
    s.syncWrites=s.syncWrites.then(()=>appendFile(path.join(s.dir,'sync.jsonl'),JSON.stringify({model,exchanges})+'\n'))
      .catch(e=>{this.error='Sync journal: '+e.message;});
  }
  readinessReason(d) {
    if(!d?.connected)return 'disconnected — open the recorder and tap Enable sensors & connect';
    if(d.health?.error)return d.health.error;
    if(!d.health?.visible)return 'keep the recorder visible in Safari';
    if(now()-d.statusAt>=2.5)return 'waiting for a fresh phone status';
    if(!d.health?.sensors)return 'waiting for accelerometer and gyroscope permission/data';
    if(!d.model||now()-d.model.host_time>=40||!d.ready)return 'synchronizing clock';
    if(d.model.uncertainty>=0.05)return 'clock uncertainty too high — check the Wi-Fi connection';
    return null;
  }
  ready(d) {return this.readinessReason(d)===null;}
  startReasons() {
    if(this.starting)return ['Preparing the next recording'];
    if(this.session?.metadata.stop===null)return ['A session is already running — press Stop & save first'];
    if(this.session&&!this.complete())return ['Saving the previous session — waiting for '+DEVICES.filter(id=>!this.session.ends.has(id)||this.session.trackers.get(id).next!==this.session.ends.get(id)+1).map(id=>id.replace('phone_','Phone ')).join(', ')];
    return DEVICES.flatMap(id=>{const reason=this.readinessReason(this.devices.get(id));return reason?[`${id.replace('phone_','Phone ')}: ${reason}`]:[];});
  }
  async start() {
    if(this.starting) throw new Error('Start is already in progress');
    if(this.session && !this.complete())throw new Error('Previous session is recording or still waiting for buffered packets');
    if(!DEVICES.every(id=>this.ready(this.devices.get(id))))throw new Error('Both phones must be ready. '+this.startReasons().join('; '));
    this.starting=true;
    try {
      const metadata={protocol_version:VERSION,session_id:'session_'+randomUUID(),server_id:this.serverId,
        start:now()+this.startDelay,stop:null,state:'scheduled',requested_hz:100,
        timestamp_source:'Browser event.timeStamp (event delivery, not independent hardware timestamps)',
        units:{acceleration:'m/s^2',rotation:'deg/s'},axis_mapping:'gx=beta, gy=gamma, gz=alpha',
        devices:DEVICES.map(id=>({device_id:id,client_id:this.devices.get(id).client_id,source:this.devices.get(id).source}))};
      const s=await new Recording(this.root,metadata).init();
      s.armed=new Set();s.trackers=new Map(DEVICES.map(id=>[id,new SequenceTracker()]));s.ends=new Map();s.syncWrites=Promise.resolve();
      this.session=s;this.sessions.set(metadata.session_id,s);this.error=null;
      this.publish({type:'session_started',session:metadata});
      for(const d of this.devices.values()) {this.logSync(d.model,d.exchanges);this.send(d.ws,{type:'prepare',session:metadata});}
      this.later(()=>{
        if(s!==this.session || s.metadata.stop!==null)return;
        if(s.armed.size!==DEVICES.length || !DEVICES.every(id=>this.ready(this.devices.get(id)))) {
          this.stop('cancelled: phones did not all arm before boundary',metadata.start).catch(e=>{this.error=e.message;});
        }else {s.metadata.state='recording';s.saveMetadata().catch(e=>{this.error=e.message;});}
      },Math.max(0,(metadata.start-now()-0.25)*1000));
      return metadata;
    }finally{this.starting=false;}
  }
  async stop(reason='stopped',boundary=now()+0.25) {
    const s=this.session;if(!s || s.metadata.stop!==null)return;
    s.metadata.stop=Math.max(s.metadata.start,boundary);s.metadata.state=reason;
    await s.saveMetadata();
    this.publish({type:'session_stopped',session_id:s.metadata.session_id,reason});
    for(const d of this.devices.values())this.send(d.ws,{type:'stop',session:s.metadata});
  }
  complete(s=this.session) {
    return !!s && s.metadata.stop!==null&&DEVICES.every(id=>s.ends.has(id)&&s.trackers.get(id).next===s.ends.get(id)+1);
  }
  async ended(d,m) {
    const s=this.sessions.get(m.session_id);
    if(!s || !Number.isSafeInteger(m.last_sequence) || m.last_sequence < -1 ||
      s.metadata.devices.find(x=>x.device_id===d.id)?.client_id!==d.client_id)throw new Error('Invalid end marker');
    if(s.ends.has(d.id)&&s.ends.get(d.id)!==m.last_sequence)throw new Error('Conflicting end marker');
    s.ends.set(d.id,m.last_sequence);
    await s.chain;
    if(s.trackers.get(d.id).next===m.last_sequence+1) {
      s.metadata.delivery=Object.fromEntries(DEVICES.map(id=>[id,{next_sequence:s.trackers.get(id).next,last_sequence:s.ends.get(id)??null}]));
      await s.saveMetadata();this.send(d.ws,{type:'end_ack',session_id:m.session_id});
    }
  }
  observe(d,samples,received,packet=true) {
    const good=samples.filter(s=>finite(s.phone_timestamp)&&['ax','ay','az','gx','gy','gz'].every(k=>finite(s[k])));
    if(!good.length)return;
    const times=good.map(s=>s.phone_timestamp);
    // Preview and persisted batches overlap; merge timestamps rather than dropping partially overlapping batches.
    d.times=[...new Set([...d.times,...times])].sort((a,b)=>a-b).slice(-300);
    d.latestAt=received;d.recent=good.slice(-10);
    if(packet)d.packetTimes.push(received);
    d.packetTimes=d.packetTimes.filter(t=>received-t<5);
  }
  accept(d,p,received) {
    // A reconnect may replay while the old socket still has a pending disk write.
    // Serialize dedupe + write + ACK across sockets, not only within one connection.
    const key=`${p.session_id}:${d.id}`;
    const job=(this.acceptChains.get(key)||Promise.resolve()).then(()=>this.acceptLocked(d,p,received));
    this.acceptChains.set(key,job.catch(()=>{}));
    return job;
  }
  async acceptLocked(d,p,received) {
    validatePacket(p);
    if(p.device_id!==d.id)throw new Error('Device mismatch');
    const s=this.sessions.get(p.session_id);
    if(!s || s.metadata.devices.find(x=>x.device_id===d.id)?.client_id!==d.client_id)
      throw new Error('Unknown session: retained on phone; download local recovery data');
    const tracker=s.trackers.get(d.id);
    if(p.packet_sequence_number>tracker.next+10000)throw new Error('Sequence outside recovery window');
    if(tracker.has(p.packet_sequence_number)) {tracker.duplicates++;this.send(d.ws,{type:'ack',session_id:p.session_id,sequence:p.packet_sequence_number});return;}
    const samples=p.samples.map(sample=>{
      const model=this.models.get(`${d.id}:${p.boot_id}:${sample.sync_id}`);
      if(!model)throw new Error('Unknown clock model: packet retained on phone');
      return {...sample,common_timestamp:sample.phone_timestamp-model.offset,offset_s:model.offset,sync_uncertainty_s:model.uncertainty};
    });
    await s.append({...p,samples,received_timestamp:received});
    tracker.add(p.packet_sequence_number);this.observe(d,samples,received);
    this.publish({type:'samples',session_id:p.session_id,device_id:d.id,
      accel_units:p.accel_units,gyro_units:p.gyro_units,
      samples:samples.map(({common_timestamp,ax,ay,az,gx,gy,gz})=>({common_timestamp,ax,ay,az,gx,gy,gz}))});
    this.send(d.ws,{type:'ack',session_id:p.session_id,sequence:p.packet_sequence_number});
  }
  publish(message) {for(const ws of this.subscribers)this.send(ws,message);}
  snapshot() {
    const t=now();
    return {type:'state',server_id:this.serverId,host_time:t,error:this.error,
      session:this.session?{...this.session.metadata,complete:this.complete()}:null,
      can_start:this.startReasons().length===0,start_reasons:this.startReasons(),
      sessions:[...this.sessions.values()].map(s=>({session_id:s.metadata.session_id,start:s.metadata.start,stop:s.metadata.stop,state:s.metadata.state,complete:this.complete(s)})),
      devices:DEVICES.map(id=>{
        const d=this.devices.get(id), tracker=this.session?.trackers.get(id);
        return {id,connected:!!d?.connected,ready:d?this.ready(d):false,sensors:!!d?.health?.sensors,
          streaming:!!d?.connected&&t-(d?.latestAt||0)<1,model:d?.model??null,
          stats:rateStats(d?.times||[]),packet_rate:(d?.packetTimes||[]).filter(x=>t-x<5).length/5,
          missing:tracker?.missing||0,duplicates:tracker?.duplicates||0,buffered:d?.health?.buffered||0,
          error:d?.health?.error||null,warning:d?.health?.warning||null,not_ready_reason:this.readinessReason(d),recent:d?.recent||[]};
      })};
  }
  broadcast() {const s=this.snapshot();for(const ws of this.viewers)this.send(ws,s);}
  async close() {
    clearInterval(this.tick);clearInterval(this.heartbeat);for(const t of this.timers)clearTimeout(t);
    for(const ws of this.sockets)ws.terminate();
    await new Promise(resolve=>this.wss.close(resolve));
    await Promise.all(this.acceptChains.values());
    for(const s of this.sessions.values()){await s.syncWrites;await s.close();}
  }
}
