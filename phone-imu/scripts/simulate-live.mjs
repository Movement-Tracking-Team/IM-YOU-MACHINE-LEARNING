// Drive two simulated phones through a recording so the live API has something to chew on.
// Usage: node e2e_simulate.mjs [seconds]
import {readFile} from 'node:fs/promises';
import {WebSocket} from 'ws';
import {SimulatedPhone, sleep, until} from './simulated-phone.mjs';

const seconds = Number(process.argv[2] || 14);
const config = JSON.parse(await readFile(new URL('../.local/config.json', import.meta.url), 'utf8'));
const ca = await readFile(new URL('../.local/ca.pem', import.meta.url));
const url = `wss://localhost:${config.httpsPort}/ws`;

const phones = [1, 2].map(n => new SimulatedPhone(url, config.token, `phone_${n}`, {ca, offset: n * 57}).start());
const dash = new WebSocket(url, {ca});
let state;
dash.on('open', () => dash.send(JSON.stringify({type: 'hello', role: 'dashboard', protocol_version: 1, token: config.token})));
dash.on('message', raw => {const m = JSON.parse(raw); if (m.type === 'state') state = m; else if (m.type === 'error') console.error(m.message);});

try {
  await until(() => state?.can_start, 20000);
  dash.send(JSON.stringify({type: 'start'}));
  await until(() => state?.session?.stop === null);
  console.log(`recording ${seconds}s as ${state.session.session_id}`);
  await sleep(seconds * 1000);
  dash.send(JSON.stringify({type: 'stop'}));
  await until(() => state?.session?.complete, 20000);
  console.log('stopped cleanly');
} finally {
  phones.forEach(p => p.close());
  dash.terminate();
}
