import {WebSocket} from 'ws';
import {randomUUID} from 'node:crypto';
import {now} from '../lib/protocol.mjs';
export const sleep=ms=>new Promise(r=>setTimeout(r,ms));
export async function until(predicate,timeout=10000){
  const start=Date.now();while(!predicate()){if(Date.now()-start>timeout)throw new Error('Timed out waiting for condition');await sleep(20);}
}
export class SimulatedPhone {
  constructor(url,token,id,{offset=20,rate=100,ca,drift=0}={}){
    Object.assign(this,{url,token,id,offset,rate,ca,drift});this.clientId=randomUUID();this.bootId=randomUUID();
    this.pending=new Map();this.seq=0;this.session=null;this.ends=new Map();this.running=true;this.errors=[];
  }
  phoneNow(){return now()*(1+this.drift)+this.offset;}
  send(m){if(this.ws?.readyState===WebSocket.OPEN)this.ws.send(JSON.stringify(m));}
  connect(){
    this.ws=new WebSocket(this.url,this.ca?{ca:this.ca}:{});
    this.ws.on('open',()=>this.send({type:'hello',role:'phone',protocol_version:1,token:this.token,
      device_id:this.id,client_id:this.clientId,boot_id:this.bootId,source:'simulated'}));
    this.ws.on('message',raw=>{
      const p2=this.phoneNow(),m=JSON.parse(raw);
      if(m.type==='welcome'){
        this.connected=true;
        if(this.session?.session_id===m.session?.session_id)this.session=m.session;
        this.replay();
      }
      if(m.type==='sync_probe')this.send({type:'sync_reply',id:m.id,p2,p3:this.phoneNow()});
      if(m.type==='sync_model'){this.model=m;this.status();}
      if(m.type==='prepare'){this.session=m.session;this.seq=0;this.send({type:'armed',session_id:m.session.session_id});}
      if(m.type==='stop'&&this.session?.session_id===m.session.session_id)this.session=m.session;
      if(m.type==='ack')this.pending.delete(`${m.session_id}:${m.sequence}`);
      if(m.type==='end_ack')this.ends.delete(m.session_id);
      if(m.type==='error')this.errors.push(m.message);
    });
    this.ws.on('close',()=>{this.connected=false;});this.ws.on('error',()=>{});
    return this;
  }
  status(){this.send({type:'status',sensors:true,visible:true,sync_id:this.model?.id,buffered:this.pending.size});}
  replay(){for(const p of this.pending.values())this.send(p);}
  start(){
    this.connect();this.next=now();
    this.timer=setInterval(()=>{
      if(!this.model){this.next=now();return;}
      const samples=[];
      while(this.next<=now()&&samples.length<10){
        const t=this.next*(1+this.drift)+this.offset,host=t-this.model.offset;
        samples.push({phone_timestamp:t,original_device_timestamp_ms:t*1000,handler_timestamp:this.phoneNow(),
          estimated_computer_timestamp:host,sync_id:this.model.id,ax:Math.sin(this.next*8),ay:0.1,az:9.81,
          gx:2,gy:3,gz:4,rotation_alpha:4,rotation_beta:2,rotation_gamma:3,browser_interval_ms:1000/this.rate});
        this.next+=1/this.rate;
      }
      if(!samples.length)return;
      if(this.session){
        const accepted=samples.filter(s=>this.session.stop===null||s.estimated_computer_timestamp<this.session.stop);
        if(accepted.length){const p={type:'samples',protocol_version:1,session_id:this.session.session_id,device_id:this.id,
          device_label:this.id,boot_id:this.bootId,packet_sequence_number:this.seq++,accel_units:'m/s^2',gyro_units:'deg/s',samples:accepted};
          this.pending.set(`${p.session_id}:${p.packet_sequence_number}`,p);this.lastPacket=p;this.send(p);}
        if(this.session.stop!==null&&this.phoneNow()-this.model.offset>=this.session.stop+0.1){this.ends.set(this.session.session_id,this.seq-1);this.session=null;}
      }else this.send({type:'preview',samples});
    },80);
    this.heartbeat=setInterval(()=>{this.status();
      for(const [id,last] of this.ends)if(![...this.pending.values()].some(p=>p.session_id===id))this.send({type:'ended',session_id:id,last_sequence:last});
    },200);
    this.retry=setInterval(()=>this.replay(),2000);return this;
  }
  async disconnect(ms=1000){this.ws.terminate();await sleep(ms);if(this.running)this.connect();}
  close(){this.running=false;clearInterval(this.timer);clearInterval(this.heartbeat);clearInterval(this.retry);this.ws?.terminate();}
}
