# Two-phone IMU prototype

For a permanent HTTPS link usable over separate Wi-Fi or cellular networks, see [cloud deployment](DEPLOY.md). The prepared Render configuration uses one receiver and a persistent recording disk.

Local Safari acquisition, a Node.js HTTPS/WebSocket receiver, and a computer dashboard. All new code, dependencies, credentials and recordings are contained in this folder. No native app, Xcode, Phyphox, cloud service, Git commit or push is required.

The current configuration records **Phone 1 and Phone 2 only**, as requested. Phone 3 is no longer shown or required. Existing certificate names and browser storage identifiers retain their original names so installed trust and buffered data remain usable.

### Start, stop, and repeat

Refresh the dashboard and both phone recorder pages after installing this update; tap **Enable sensors & connect** on each phone again. When both cards show **Ready**, the dashboard says **Ready to record** and enables Start. Press **Start recording**, wait for the three-second countdown, and press **Stop & save** whenever you want to end the session. After the final packets are acknowledged, it shows **Saved** and enables **Start new session**. Repeat without refreshing. Each recording has its own ID and directory. Use **Saved sessions** to download any earlier session created since this receiver started. If Start is unavailable, the text under it names the missing phone, sensor/clock condition, or pending data transfer.

Repeated or out-of-order browser event timestamps are preserved, with `event_index` and `timestamp_order` CSV fields, and reported as a warning. They no longer permanently disable readiness or force a recording to stop. Inspect these flags before downstream resampling. True timestamp-domain incompatibility or storage failures still block acquisition.

## Architecture inspected

The parent repository is a Python/PyTorch BioMAT reproduction and movement classifier. It has `biomat/classify_phone.py` for offline CSV inference, but no running frontend, mobile project, React Native, Expo, or WebSocket receiver to extend. The sibling upstream checkout includes an unrelated Streamlit model demonstration. This isolated JavaScript prototype collects datasets for that Python workflow without changing its training code.

The user's final browser requirements supersede the original Core Motion proposal. Browser motion events expose a shared event timestamp, not separate native accelerometer/gyroscope hardware timestamps. This distinction is retained in metadata. We do not claim raw Core Motion fidelity or a guaranteed 100 Hz rate.

## Run on this computer

From this folder, Node.js 22+ and OpenSSL are required. The backend serves the frontend itself; there is only one process to run.

```powershell
npm.cmd ci
./scripts/setup.ps1 -Address 192.168.137.1
npm.cmd start
```

If npm has been configured offline but you have network access, use `npm.cmd ci --offline=false`. In this managed workspace dependencies were installed from the existing npm cache. The lockfile pins the installed packages. No global npm configuration was changed.

For macOS/Linux, use `npm run setup -- YOUR_WIFI_IPV4` instead of the PowerShell setup script. Use `ipconfig` on Windows to find the Wi-Fi IPv4 address. If that address changes, stop the receiver, rerun setup with the new address and restart. Setup keeps the existing CA and pairing token; only the server certificate is renewed. Do not regenerate/delete the CA unless you intend to reinstall trust on all phones.

The console prints the paired dashboard URL and phone URL. The laptop dashboard is served at `http://localhost:8080/dashboard` (use the printed paired link). Its plain WebSocket is restricted to loopback connections, so the laptop does not need to trust the local HTTPS certificate just to monitor recordings. Phones still use HTTPS/WSS. Current setup page:

**http://192.168.137.1:8080** (Windows Mobile Hotspot)

Open that page on this computer to find the paired dashboard link. It is also the single setup link to open on every iPhone. Over the LAN, HTTP only serves certificate setup; phone sensor collection uses HTTPS/WSS on port 8443. The laptop's loopback dashboard is the only plain-HTTP monitoring exception. The local pairing token is in `.local/config.json`; links carry it in their fragment, which the frontend immediately removes and stores locally.

## Connect two iPhones

1. Put this computer and all phones on the same Wi-Fi network. Open the setup link above **in Safari**, not an embedded social-app browser.
2. Download the certificate profile and tap Allow. In Settings → General → VPN & Device Management, install **Three Phone IMU HTTPS**.
3. In Settings → General → About → Certificate Trust Settings, enable full trust for **Three Phone IMU Local CA**. Compare the certificate SHA-256 fingerprint displayed on the setup page with the receiver console. The profile contains only this local CA, with no management enrollment. Remove it when finished testing.
4. Return to the setup page and tap **open the phone recorder**. Select **Phone 1** or **Phone 2**, one different number per phone. Tap **Enable sensors & connect**, then allow Motion & Orientation access. If iOS asks for local-network access, allow it.
5. Keep Safari visible and the screen unlocked. The app requests a screen wake lock where supported. If unavailable, temporarily set Auto-Lock to Never for the experiment. Motion permission must be requested by this tap; it cannot be granted remotely by the computer.
6. On the computer dashboard, wait until both cards show Ready and both sensor streams are available. Press **Start recording**. Acquisition is already warm; recording begins on the common timestamp three seconds later.
7. Press **Stop** on the computer (or Stop all phones on a phone). Leave pages open until all buffers drain. Download `combined.csv`, two phone CSVs, and `metadata.json` from the dashboard.

A warning-page bypass is not a substitute for certificate trust: Safari may still reject motion access. If motion was denied, check Safari's website Motion & Orientation permission, then reload and enable sensors again. On desktop browsers, absent motion events are expected.

### Local network troubleshooting

The initial university Wi-Fi address (`172.28.26.232`) did not load on the user's iPhone. Windows already had an inbound TCP allowance for the actual Node executable. The user enabled Windows Mobile Hotspot and connected the phone; the receiver certificate and setup links were updated to **192.168.137.1**. Keep the hotspot enabled and connect both phones to it. If the setup page does not load on a different network, rerun setup with its IPv4 address. Disable client/AP isolation only on a router you control if enabled.

Windows Firewall may require an inbound rule for TCP 8080 and 8443. On a trusted **Private** network, an administrator can run:

```powershell
New-NetFirewallRule -DisplayName 'Three Phone IMU local prototype' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080,8443 -Profile Private -RemoteAddress LocalSubnet
```

This command was **not** run automatically; no global firewall or system trust settings were changed. `scripts/allow-local-network.ps1 -Address YOUR_IPV4` also provides an administrator-only rule restricted to the actual Node executable, current local IP, these two ports and LocalSubnet peers if a future network requires it. Do not disable Windows Firewall. The prototype's bootstrap page exposes its local pairing link to people on the same network, so use a trusted test network and do not port-forward it to the Internet.

## Acquisition and synchronization

- `DeviceMotionEvent.accelerationIncludingGravity` is preserved in **m/s²**. `rotationRate` is preserved in **degrees/second**, including the original alpha/beta/gamma fields. Axis mapping: `gx = beta`, `gy = gamma`, `gz = alpha`. No smoothing, gravity removal, normalization or resampling is applied.
- A 100 Hz target is recorded as intent. DeviceMotionEvent has no cross-browser frequency-setting API; Safari may deliver around the display refresh rate or lower. Actual rate, interval standard deviation and maximum intervals derive from delivered event timestamps. The browser provides one timestamp for the event containing both sensors; independent native sensor timestamps are unavailable.
- `event.timeStamp` is retained as `original_device_timestamp_ms`. Modern relative timestamps are converted from milliseconds to seconds without changing their origin. Legacy epoch timestamps, if encountered, are explicitly normalized using `performance.timeOrigin`, preserving the original. An incompatible timestamp domain raises an error. `handler_timestamp` is retained separately.
- The host and phone use `performance.now()/1000`. The host sends H1, the phone captures P2 before parsing and P3 before sending, and the host captures H4 before parsing. Twenty-five probes run initially, then every 15 seconds. The median offset of the five lowest-RTT valid exchanges is used: `((P2-H1)+(P3-H4))/2`. Computer time is phone time **minus** this offset.
- Each clock model has an immutable ID. Every sample names its model. Both the phone estimate and the receiver's independently calculated conversion are saved. Later resynchronization does not apply a new offset retroactively to buffered packets. All exchanges and models are recorded in `sync.jsonl`.
- Estimated uncertainty is minimum RTT/2 plus spread of the selected offsets. This describes the clock-exchange estimate, **not measured physical sensor alignment**. Asymmetric Wi-Fi delay, browser scheduling, sensor filtering and delivery delay remain unmeasured. Periodic resync uses piecewise offsets; no assumption of perfect drift compensation is made.
- START requires fresh sensor status and a recent model on both phones. A start is scheduled three seconds ahead; phones must acknowledge that they are armed before the deadline or it is cancelled. Pre-start measurements stay in the raw journal; CSVs include only `[start, stop)`. STOP schedules a shared boundary 250 ms ahead. A disconnected phone learns that boundary on reconnect and its late samples are filtered consistently.
- Batches contain up to ten samples, flushed every 80 ms. This caps batching latency even at lower browser rates. Preview plots update separately from recording persistence. Transmission is asynchronous with up to four unacknowledged batches in flight.
- IndexedDB persists each batch before transmission. Per-session sequence numbers, receiver deduplication and acknowledgements after journal `datasync()` support temporary connection loss. Lost ACKs cause harmless replay. A disk/storage/protocol error is visible and packets remain available for recovery. There is no silent buffer eviction.

## Dataset files

Every session has an ignored directory under `recordings/session_UUID/`:

| File | Contents |
| --- | --- |
| `packets.jsonl` | Original batches plus reception timestamps, conversion models and common sample times; authoritative journal, including pre-roll and any post-window samples |
| `sync.jsonl` | Clock models and H1/P2/P3/H4 exchanges |
| `metadata.json` | Shared start/stop, protocol, clock epoch identifier, participant identities, units and delivery status |
| `phone_1.csv`, `phone_2.csv` | Each phone's raw axes within the common window, sorted by common time |
| `combined.csv` | Both phones' rows merged and sorted by common time; no interpolation or fabricated simultaneous samples |

CSV export runs in a worker thread after recording and delivery complete, keeping sorting off the receiving event loop. Columns include device/session IDs, `common_timestamp`, `session_time_s`, original device timestamp, normalized phone timestamp, handler timestamp, phone estimate, all six axes, original rotation fields, packet sequence, units, sync model and uncertainty, packet receive time, and browser-reported interval.

```python
import pandas as pd
df = pd.read_csv('recordings/session_UUID/combined.csv')
phone1 = df[df.device_id == 'phone_1'].sort_values('common_timestamp')
print(phone1.phone_timestamp.diff().describe())
```

Existing stopped recordings can be re-exported after a receiver restart with:

```powershell
node scripts/export.mjs recordings/session_UUID
```

The existing Python classifier expects a calibrated 100 Hz-or-higher dataset. Safari recordings below its minimum rate are **not directly compatible**. The classifier's unit, mounting-axis, gravity and calibration checks remain intact. Do not bypass them or interpolate a low-rate recording and claim it was sampled at 100 Hz.

## Tests

```powershell
npm.cmd test
# Start the receiver first; keep physical phones disconnected during simulation:
npm.cmd run simulate
# Optional, on a machine where Playwright is allowed to launch Chromium:
node scripts/browser-check.mjs
```

The simulator opens two authenticated WSS clients with different clock offsets and drift, starts a session, disconnects Phone 2 for 1.2 seconds while generating data, replays its buffer, stops, and downloads the merged CSV. Its report is `.local/simulation-result.json`. These clients are clearly recorded as `source: simulated` in session metadata. Do not confuse simulation with physical-iPhone validation.

See [VALIDATION.md](VALIDATION.md) for checks actually executed and the practical phone test. This prototype cannot install the certificate or grant permissions on a remote iPhone automatically.

## Prototype boundaries and recovery

- Do not background Safari, lock the phone, navigate away or reload during recording. Visibility changes are reported. No browser can guarantee continuous collection while iOS suspends the page. A page crash can lose the partial batch not yet persisted (up to about 80 ms); phone flash failure or OS eviction can lose more.
- Ordinary Wi-Fi disconnects with the page alive are recovered automatically. A receiver restart creates a new monotonic epoch and does not resume an active session or accept old packets against new clock models. Stored packets remain on the phone; use **Download local recovery packets**. An interrupted session requires manual recovery/inspection, not a success label. Download recovery data before clearing site storage.
- A page reload interrupts acquisition. Its saved batches are replayed, but the gap is reported; stop and repeat the experiment. Do not refresh just to change phone numbers during a session.
- A stopped session is complete only when both phones supply their final sequence and all sequences have been stored. Trailing losses are therefore detected even if no later packet arrives. Leave the receiver and phones running during draining. An unrecoverable phone blocks the next session; save the journal/recovery data and restart the receiver for a new experiment.
- Recordings and browser buffers consume local disk. Storage failures stop capture visibly. Large exports currently load that session into worker memory; keep prototype sessions modest.
- Protect `.local/ca-key.pem` and the pairing configuration. `.local/` and recordings are ignored by this folder's `.gitignore`. They are under your OneDrive-backed workspace and may still be synchronized by OneDrive; Git ignore does not control that desktop service.

## Implementation files

`server.mjs` hosts HTTPS, setup and export routes. `lib/receiver.mjs` handles WebSockets, synchronization, readiness and session boundaries. `lib/protocol.mjs` validates packets and computes rate/clock statistics. `lib/storage.mjs` and `lib/export-worker.mjs` handle durable journals and exports. `public/phone.js`, `buffer.js` and `shared.js` implement Safari acquisition and replay; `public/dashboard.js` controls the computer interface. `scripts/` contains certificate setup, offline export, simulated clients and browser validation. `test/` contains the automated tests.

API references: [W3C Device Orientation and Motion](https://www.w3.org/TR/orientation-event/), [WebKit motion permission discussion](https://bugs.webkit.org/show_bug.cgi?id=201676), [Apple: trust manually installed certificate profiles](https://support.apple.com/en-us/102390).
