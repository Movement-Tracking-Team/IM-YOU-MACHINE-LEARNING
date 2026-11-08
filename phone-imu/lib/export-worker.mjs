import {parentPort, workerData} from 'node:worker_threads';
import {exportRecording} from './storage.mjs';
try { parentPort.postMessage(await exportRecording(workerData)); }
catch (e) { parentPort.postMessage({error:e.message}); }
