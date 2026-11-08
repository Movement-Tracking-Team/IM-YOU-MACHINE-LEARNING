import path from 'node:path';
import {exportRecording} from '../lib/storage.mjs';
if(!process.argv[2])throw new Error('Usage: node scripts/export.mjs recordings/session_UUID');
console.log(await exportRecording(path.resolve(process.argv[2])));
