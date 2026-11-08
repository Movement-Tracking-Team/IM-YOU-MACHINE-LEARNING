# Live joint-angle and activity API

Runs the 100 Hz `canonical100_v1` models on a sliding window while a recording is
in progress, and pushes one result every half second to any connected client.

## Running it

Three processes, in this order:

```bash
cd phone-imu && npm start          # receiver: phones connect here
cd live-api  && python server.py   # this service, on :8000
cd frontend  && npx expo start     # the app
```

The service finds the receiver through `phone-imu/.local/config.json`. Override
with `IMU_RECEIVER_URL` and `IMU_TOKEN`, and point `BIOMAT_EXPORT` at a different
checkpoint folder if needed.

Predictions only flow **while a session is recording**, because clock-corrected
timestamps only exist then. Press Start on the receiver dashboard first.

## Endpoints

| Route | What it gives you |
| --- | --- |
| `GET /health` | receiver connected, recording state, observed sample rate |
| `GET /contract` | window size, cadence, class names, checkpoint SHA-256, caveats |
| `WS /live` | `hello`, then `status` on change and `window` every 0.5 s |

A `window` message carries the 2.5 s it describes, whether it passed the motion
gate, the activity, per-class scores, and 250 samples of hip, knee and ankle
angle in degrees.

## What the numbers mean

`activity` is restricted to walking, stairs up and stairs down. The checkpoint
also has ramp up and ramp down, but no ramp was ever walked on a phone, and on
our own eight labelled trials those two classes absorbed most of the errors:
window agreement was 59.3% with them and 91.8% without. The full five scores are
still in `scores` if you want them.

`usable: false` means the acquisition gate rejected the window — under 75% of
samples above 0.25 rad/s, or an unstable rotation axis. It is **not** a trained
standing class. On our recorded trials the gate rejected 54% of windows, so
expect the activity readout to go quiet often.

Joint angles are unvalidated on phone data. There is no motion capture to compare
against, so treat them as a trace, not a measurement.

## Latency

Each result covers the preceding 2.5 s and arrives every 0.5 s. That delay is
inherent: the low-pass filter and the canonicalization are both two-sided, so
they need the whole window plus a short margin. Inference itself is 26 ms median
and 62 ms at the 95th percentile on CPU, against a 500 ms budget.

## Checking it still matches the offline path

`replay.py` feeds a recorded trial through the streaming engine and compares it
against `predict.py` running on the same recording:

```bash
python replay.py --activity stair_ascent --trial 1
```

Across seven trials the gate decisions and class predictions are identical and
the largest joint-angle difference is 0.005°. The streaming engine emits one
fewer window per trial, because the last one has no room for its filter margin.

`e2e_listen.py` connects to `/live` and prints what arrives;
`phone-imu/scripts/simulate-live.mjs <seconds>` drives two simulated phones
through a recording so you can exercise the chain without hardware. The
simulated signal is too gentle to pass the motion gate, which is expected.
