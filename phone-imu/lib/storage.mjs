import {mkdir, open, readFile, writeFile, rename} from 'node:fs/promises';
import path from 'node:path';
export async function atomicJSON(file, data) {
  await writeFile(file+'.tmp', JSON.stringify(data, null, 2));
  await rename(file+'.tmp',file);
}
export class Recording {
  constructor(root, metadata) {
    this.metadata = metadata;
    this.dir = path.join(root, metadata.session_id);
    this.chain = Promise.resolve();
    this.metadataChain = Promise.resolve();
  }
  async init() {
    await mkdir(this.dir, {recursive:true});
    this.file = await open(path.join(this.dir, 'packets.jsonl'), 'a');
    await this.saveMetadata();
    return this;
  }
  saveMetadata() {
    const snapshot=structuredClone(this.metadata);
    this.metadataChain=this.metadataChain.then(()=>atomicJSON(path.join(this.dir, 'metadata.json'),snapshot));
    return this.metadataChain;
  }
  append(packet) {
    const job = this.chain.then(async () => {
      await this.file.write(JSON.stringify(packet)+'\n');
      // ACK only after the OS has flushed the journal to disk.
      await this.file.datasync();
    });
    this.chain = job;
    return job;
  }
  async close() { await this.chain; await this.metadataChain; await this.file.close(); }
}

export const COLUMNS = ['device_id','session_id','common_timestamp','session_time_s','original_device_timestamp_ms',
  'phone_timestamp','handler_timestamp','estimated_computer_timestamp','ax','ay','az','gx','gy','gz',
  'rotation_alpha','rotation_beta','rotation_gamma','packet_sequence_number','accel_units','gyro_units',
  'sync_id','offset_s','sync_uncertainty_s','packet_received_timestamp','browser_interval_ms','event_index','timestamp_order'];

// Run in a worker: export never sorts a large recording on the receiver's event loop.
export async function exportRecording(dir) {
  const metadata = JSON.parse(await readFile(path.join(dir,'metadata.json'),'utf8'));
  if (metadata.stop == null) throw new Error('Stop recording before exporting');
  const rows = [];
  const lines = (await readFile(path.join(dir,'packets.jsonl'),'utf8')).split('\n').filter(Boolean);
  for (const line of lines) {
    const packet = JSON.parse(line);
    for (const s of packet.samples) {
      const t = s.common_timestamp;
      if (t < metadata.start || t >= metadata.stop) continue;
      rows.push({...s, device_id:packet.device_id, session_id:packet.session_id, session_time_s:t-metadata.start,
        packet_sequence_number:packet.packet_sequence_number, accel_units:packet.accel_units,
        gyro_units:packet.gyro_units, packet_received_timestamp:packet.received_timestamp});
    }
  }
  rows.sort((a,b)=>a.common_timestamp-b.common_timestamp || a.device_id.localeCompare(b.device_id));
  const csv = values => COLUMNS.join(',')+'\n'+values.map(r=>COLUMNS.map(k=>r[k] ?? '').join(',')).join('\n')+'\n';
  await writeFile(path.join(dir,'combined.csv'), csv(rows));
  for (const {device_id:id} of metadata.devices) await writeFile(path.join(dir,id+'.csv'), csv(rows.filter(r=>r.device_id===id)));
  return {rows:rows.length, session_id:metadata.session_id};
}
