export const VERSION = 1;
export const DEVICES = ['phone_1', 'phone_2'];
export const now = () => performance.now() / 1000;
export const finite = v => typeof v === 'number' && Number.isFinite(v);
export function estimateClock(exchanges) {
  const valid = exchanges.map(({h1, p2, p3, h4}) => ({
    offset: ((p2 - h1) + (p3 - h4)) / 2,
    rtt: (h4 - h1) - (p3 - p2), h1, p2, p3, h4
  })).filter(x => finite(x.offset) && finite(x.rtt) && x.rtt >= 0 && x.rtt < 1 && x.p3 >= x.p2);
  if (valid.length < 10) throw new Error('Insufficient valid clock exchanges');
  const best = valid.sort((a,b) => a.rtt - b.rtt).slice(0, 5);
  const offsets = best.map(x => x.offset).sort((a,b) => a-b);
  const offset = offsets[Math.floor(offsets.length / 2)];
  return {offset, uncertainty: best[0].rtt / 2 + Math.max(...offsets.map(x => Math.abs(x-offset))),
    min_rtt: best[0].rtt, exchanges: valid.length};
}

export function validatePacket(p) {
  if (p.protocol_version !== VERSION || p.type !== 'samples' || !DEVICES.includes(p.device_id) ||
      typeof p.session_id !== 'string' || !Number.isSafeInteger(p.packet_sequence_number) || p.packet_sequence_number < 0 ||
      !Array.isArray(p.samples) || p.samples.length < 1 || p.samples.length > 100 ||
      p.accel_units !== 'm/s^2' || p.gyro_units !== 'deg/s') throw new Error('Invalid packet envelope');
  for (const s of p.samples) {
    if (!['phone_timestamp','original_device_timestamp_ms','handler_timestamp','estimated_computer_timestamp',
      'ax','ay','az','gx','gy','gz','rotation_alpha','rotation_beta','rotation_gamma'].every(k => finite(s[k])) ||
      !Number.isSafeInteger(s.sync_id)) throw new Error('Invalid sample');
    // Browser timestamps may repeat or arrive out of order. Preserve them rather than fabricating times.
    if(s.event_index!==undefined&&(!Number.isSafeInteger(s.event_index)||s.event_index<0))throw new Error('Invalid event index');
    if(s.timestamp_order!==undefined&&!['increasing','duplicate','out_of_order'].includes(s.timestamp_order))throw new Error('Invalid timestamp flag');
  }
}

export class SequenceTracker {
  constructor() { this.next = 0; this.later = new Set(); this.duplicates = 0; }
  has(n) { return n < this.next || this.later.has(n); }
  add(n) {
    if (this.has(n)) { this.duplicates++; return false; }
    this.later.add(n);
    while (this.later.delete(this.next)) this.next++;
    return true;
  }
  get missing() { return this.later.size ? Math.max(...this.later) - this.next + 1 - this.later.size : 0; }
}

export function rateStats(times) {
  const dt = times.slice(1).map((t,i) => t - times[i]).filter(t => t > 0);
  if (!dt.length) return {hz: 0, jitter_ms: 0, max_interval_ms: 0};
  const mean = dt.reduce((a,b)=>a+b,0)/dt.length;
  return {hz: 1/mean, jitter_ms: Math.sqrt(dt.reduce((s,x)=>s+(x-mean)**2,0)/dt.length)*1000,
    max_interval_ms: Math.max(...dt)*1000};
}
