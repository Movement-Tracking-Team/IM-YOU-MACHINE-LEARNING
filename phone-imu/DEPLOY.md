# Permanent phone recording link

Deploy this app as one Node.js web service with a persistent disk and managed HTTPS. Render is the prepared target. Phones and the dashboard may use different networks; no local CA profile or hotspot is needed. Sensor timestamps remain device timestamps. Internet latency/asymmetry can increase clock uncertainty; validate synchronization using the actual networks before an experiment.

## Free Render deployment for testing

No purchased domain or persistent disk is required. Use `phone-imu/render-free.yaml` as the Blueprint path in the parent repository. This selects the Free plan and temporary recording storage. For a standalone repository containing only this app, remove `rootDir: phone-imu`; the prepared `phone-recorder-free` package already does this and names the free Blueprint `render.yaml`.

1. Upload the contents of the prepared `phone-recorder-free` folder to a new GitHub repository (keep `package.json` and `render.yaml` at the repository root).
2. Sign in at https://dashboard.render.com, choose **New > Blueprint**, connect GitHub, and select that repository.
3. Review the Blueprint: it must show a **Free** web service with **no disk**. Deploy it. The pairing key is generated automatically.
4. Open the service's `https://...onrender.com` link. Copy `IMU_TOKEN` from the service's Environment settings and enter it once on each device. Bookmark the link.

Download CSVs immediately after every recording. Free services can restart at any time and sleep after 15 minutes without inbound traffic; server files are deleted on sleep, restart, or redeployment. An active recording is interrupted by a restart. Waking the service can take about a minute. The URL remains the same. Free usage quotas apply; without a payment method, exceeding allowances can suspend service rather than incur overage charges. See https://render.com/docs/free.

## Paid Render deployment with retained recording files

1. Connect the repository to Render and create a Blueprint using `phone-imu/render.yaml`. The file assumes the repository root is `Movement-Tracking-Machine-Learning`. If publishing only the `phone-imu` directory as its own repository, remove `rootDir: phone-imu` and use `render.yaml` as the Blueprint path.
2. Review the paid Starter service and 1 GB disk charges before creating the service. Do not substitute ephemeral storage for the recording disk.
3. Once deployed, open the service's permanent HTTPS URL. Retrieve the generated `IMU_TOKEN` from the service's environment settings and enter it on the connection page once per device. Treat it as a password: it grants access to recording controls and downloads. Do not commit it or paste it in public messages.
4. Bookmark the connection page on each device. Open the dashboard on the computer and the phone recorder in Safari on each iPhone. Choose different phone numbers and enable sensors. The dashboard's private setup link can also pair a new phone.

Deploy only source files: exclude `.local`, `recordings`, `node_modules`, test outputs, and any keys/configuration files. Existing local recordings are not uploaded automatically. The persistent disk stores new hosted recordings.

## Runtime settings

- `IMU_CLOUD=1`: use the hosting provider's HTTPS proxy and one HTTP/WebSocket backend port.
- `PORT`: supplied by the host; defaults to 10000.
- `IMU_TOKEN`: random secret, at least 32 characters. Cloud startup fails without it.
- `IMU_RECORDINGS`: persistent disk directory. Required in cloud mode.

Keep exactly one instance. Session coordination uses process memory. A restart or redeployment interrupts an active recording; finish and download before deploying. Files survive on the disk, but the dashboard lists only sessions from the current server process. For an older stopped session use `node scripts/export.mjs /var/data/recordings/session_UUID` in the hosting shell and retrieve its files. Interrupted sessions require manual recovery as described in README.md. Disk backups are separate from this app.

Vercel Functions support WebSockets, but connections can expire at the function duration limit and different clients can land on different instances. Hosting this receiver there requires external shared session coordination and durable storage; uploading the present app alone is insufficient.

The default `npm start` without `IMU_CLOUD=1` retains local HTTPS/hotspot mode.
