export class DurableBuffer {
  async open(){
    this.db=await new Promise((resolve,reject)=>{
      const r=indexedDB.open('three-phone-imu-v1',1);
      r.onupgradeneeded=()=>r.result.createObjectStore('packets',{keyPath:'key'});
      r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
    });return this;
  }
  async transaction(mode,action){
    return new Promise((resolve,reject)=>{
      const t=this.db.transaction('packets',mode),r=action(t.objectStore('packets'));
      t.oncomplete=()=>resolve(r?.result);t.onerror=()=>reject(t.error);t.onabort=()=>reject(t.error||new Error('Buffer transaction aborted'));
    });
  }
  put(packet){return this.transaction('readwrite',s=>s.put({key:`${packet.session_id}:${packet.packet_sequence_number}`,packet}));}
  remove(session,sequence){return this.transaction('readwrite',s=>s.delete(`${session}:${sequence}`));}
  all(){return this.transaction('readonly',s=>s.getAll());}
  count(){return this.transaction('readonly',s=>s.count());}
}
