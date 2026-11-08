import {mkdir,readFile,writeFile,access} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {networkInterfaces} from 'node:os';
import {randomBytes,randomUUID,X509Certificate} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(fileURLToPath(new URL('..',import.meta.url)));
process.chdir(root);
await mkdir('.local',{recursive:true});
const addresses=Object.values(networkInterfaces()).flat().filter(x=>x.family==='IPv4'&&!x.internal).map(x=>x.address);
const ip=process.argv[2]||addresses[0];
if(!ip || !/^\d{1,3}(\.\d{1,3}){3}$/.test(ip))throw new Error('Pass the computer Wi-Fi IPv4 address: npm run setup -- 192.168.1.20');
const openssl=process.env.OPENSSL || 'openssl';
const run=args=>execFileSync(openssl,args,{stdio:'pipe'});
if(!process.argv.includes('--existing-certs')) {
await writeFile('.local/openssl.cnf','[req]\ndistinguished_name=dn\n[dn]\n');
try {await access('.local/ca.pem');} catch {
  run(['req','-config','.local/openssl.cnf','-x509','-newkey','rsa:2048','-nodes','-keyout','.local/ca-key.pem','-out','.local/ca.pem',
    '-sha256','-days','365','-subj','/CN=Three Phone IMU Local CA',
    '-addext','basicConstraints=critical,CA:TRUE,pathlen:0','-addext','keyUsage=critical,keyCertSign,cRLSign']);
}
await writeFile('.local/server.ext',`subjectAltName=IP:${ip},IP:127.0.0.1,DNS:localhost\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n`);
run(['req','-config','.local/openssl.cnf','-new','-newkey','rsa:2048','-nodes','-keyout','.local/server-key.pem','-out','.local/server.csr','-subj','/CN=Three Phone IMU']);
run(['x509','-req','-in','.local/server.csr','-CA','.local/ca.pem','-CAkey','.local/ca-key.pem','-CAcreateserial',
  '-out','.local/server.pem','-days','90','-sha256','-extfile','.local/server.ext']);
}
const ca=new X509Certificate(await readFile('.local/ca.pem'));
let existing={};try{existing=JSON.parse(await readFile('.local/config.json','utf8'));}catch{}
const config={ip,httpsPort:8443,setupPort:8080,token:existing.token||randomBytes(24).toString('hex'),fingerprint:ca.fingerprint256};
await writeFile('.local/config.json',JSON.stringify(config,null,2));
const profile=`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>PayloadContent</key><array><dict><key>PayloadCertificateFileName</key><string>imu-local-ca.cer</string>
<key>PayloadContent</key><data>${ca.raw.toString('base64')}</data><key>PayloadDescription</key><string>Trust the HTTPS receiver on your own computer.</string>
<key>PayloadDisplayName</key><string>Three Phone IMU Local CA</string><key>PayloadIdentifier</key><string>local.imu.root</string>
<key>PayloadType</key><string>com.apple.security.root</string><key>PayloadUUID</key><string>${randomUUID()}</string><key>PayloadVersion</key><integer>1</integer></dict></array>
<key>PayloadDisplayName</key><string>Three Phone IMU HTTPS</string><key>PayloadDescription</key><string>Local development certificate only. No device management or data collection.</string>
<key>PayloadIdentifier</key><string>local.imu.profile</string><key>PayloadRemovalDisallowed</key><false/>
<key>PayloadType</key><string>Configuration</string><key>PayloadUUID</key><string>${randomUUID()}</string><key>PayloadVersion</key><integer>1</integer>
</dict></plist>`;
await writeFile('.local/imu.mobileconfig',profile);
await writeFile('.local/imu-ca.cer',ca.raw);
console.log(`HTTPS ready for ${ip}.\nPhone setup: http://${ip}:8080\nDashboard: http://localhost:8080/dashboard#token=${config.token}\nCA SHA-256: ${config.fingerprint}`);
