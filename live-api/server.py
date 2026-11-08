"""Live joint-angle and activity API.

Subscribes to the phone-IMU receiver, runs the 100 Hz canonical100_v1 models on a
sliding window, and pushes one result every half second to any connected client.
"""
import asyncio
import contextlib
import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine import CLASSES, EXPORT_DIR, HOP, HZ, LENGTH, PERFORMED, TARGETS, LiveEngine

LOG = logging.getLogger("live-api")
PHONE_IMU = Path(__file__).resolve().parents[1] / "phone-imu"
RECEIVER_CONFIG = Path(os.environ.get("IMU_CONFIG", PHONE_IMU / ".local" / "config.json"))
PROTOCOL_VERSION = 1
RECONNECT_SECONDS = 2.0


def receiver_target():
    """Where the Node receiver is, and the token to register with.

    Browser phones on `/phone` do not need this. Missing config means we wait
    for those sockets instead of refusing to start.
    """
    url, token = os.environ.get("IMU_RECEIVER_URL"), os.environ.get("IMU_TOKEN")
    if url and token:
        return url, token
    if not RECEIVER_CONFIG.exists():
        return None
    config = json.loads(RECEIVER_CONFIG.read_text())
    # The loopback bootstrap port serves /ws without the local certificate.
    return url or f"ws://127.0.0.1:{config['setupPort']}/ws", token or config["token"]


def normalize_samples(samples):
    """Accept either the receiver's `common_timestamp` or the app's `t`."""
    out = []
    for sample in samples:
        if "common_timestamp" not in sample and "t" in sample:
            sample = {**sample, "common_timestamp": sample["t"]}
        out.append(sample)
    return out


def checkpoint_digest():
    sha = hashlib.sha256()
    for name in sorted(["model.pt", "classifier_model.pt"]):
        sha.update((EXPORT_DIR / name).read_bytes())
    return sha.hexdigest()


class Hub:
    """One engine, many viewers."""

    def __init__(self):
        self.engine = LiveEngine()
        self.clients: set[WebSocket] = set()
        self.receiver_connected = False
        # Browser phones streaming straight to /phone, counted so a disconnect is reported.
        self.phones = 0
        self.session = None
        self.windows_sent = 0
        self.last_window = None
        self.lock = asyncio.Lock()

    def status(self):
        return {"type": "status", "receiver_connected": self.receiver_connected or self.phones > 0,
                "phones": self.phones,
                "session_id": (self.session or {}).get("session_id"),
                "recording": self.session is not None,
                "observed_hz": self.engine.rates(),
                "windows_sent": self.windows_sent}

    def contract(self):
        return {"type": "hello", "sample_rate_hz": HZ, "window_samples": LENGTH,
                "hop_samples": HOP, "seconds_per_window": LENGTH / HZ,
                "seconds_between_windows": HOP / HZ,
                "classes": CLASSES, "reported_classes": PERFORMED, "targets": TARGETS,
                "angle_units": "degrees",
                "checkpoint_sha256": checkpoint_digest(),
                "caveats": [
                    "Each result describes the preceding 2.5 s, not the present instant.",
                    "Windows failing the motion gate carry no activity; the gate is not a trained standing class.",
                    "Joint angles are unvalidated on phone data: no motion capture reference exists.",
                ]}

    async def broadcast(self, message):
        for client in list(self.clients):
            try:
                await client.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                self.clients.discard(client)

    async def handle(self, message):
        kind = message.get("type")
        if kind == "session_started":
            async with self.lock:
                self.engine.reset()
            self.session = message.get("session")
            await self.broadcast(self.status())
        elif kind == "session_stopped":
            self.session = None
            await self.broadcast(self.status())
        elif kind == "samples":
            await self.ingest(message)

    async def ingest(self, message):
        if self.session is None:
            return
        async with self.lock:
            self.engine.add_samples(message["device_id"], normalize_samples(message.get("samples", [])),
                                    message.get("accel_units", "m/s^2"),
                                    message.get("gyro_units", "deg/s"))
            # 26 ms median, 62 ms p95: off the event loop so sample intake never stalls.
            windows = await asyncio.to_thread(self.engine.drain)
        for window in windows:
            self.last_window = window
            self.windows_sent += 1
            await self.broadcast({"type": "window", **asdict(window)})


async def pump(hub: Hub):
    """Stay subscribed to the receiver, reconnecting for as long as the service runs."""
    target = receiver_target()
    if target is None:
        LOG.info("no phone-imu receiver; browser phones connect on /phone")
        return
    url, token = target
    hello = json.dumps({"type": "hello", "role": "subscriber",
                        "token": token, "protocol_version": PROTOCOL_VERSION})
    while True:
        try:
            async with websockets.connect(url, max_size=2 ** 20) as socket:
                await socket.send(hello)
                hub.receiver_connected = True
                await hub.broadcast(hub.status())
                LOG.info("subscribed to receiver at %s", url)
                async for raw in socket:
                    await hub.handle(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("receiver connection lost (%s); retrying", exc)
        finally:
            hub.receiver_connected = False
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(RECONNECT_SECONDS)


hub = Hub()
app = FastAPI(title="Movement Tracking live API")
# The Expo client is served from a different origin during development.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    app.state.pump = asyncio.create_task(pump(hub))


@app.on_event("shutdown")
async def shutdown():
    app.state.pump.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await app.state.pump


@app.get("/health")
async def health():
    return {"ok": True, **hub.status()}


@app.get("/contract")
async def contract():
    return hub.contract()


@app.websocket("/phone")
async def phone(socket: WebSocket):
    """Safari phones stream samples here. The Vercel app calls this machine."""
    await socket.accept()
    hub.phones += 1
    await hub.broadcast(hub.status())
    try:
        while True:
            message = await socket.receive_json()
            kind = message.get("type")
            if kind == "session_start":
                async with hub.lock:
                    hub.engine.reset()
                hub.session = {"session_id": message.get("session_id")}
                await hub.broadcast(hub.status())
            elif kind == "session_stop":
                hub.session = None
                await hub.broadcast(hub.status())
            elif kind == "samples":
                await hub.ingest(message)
    except WebSocketDisconnect:
        pass
    finally:
        hub.phones -= 1
        await hub.broadcast(hub.status())


@app.websocket("/live")
async def live(socket: WebSocket):
    await socket.accept()
    hub.clients.add(socket)
    try:
        await socket.send_json(hub.contract())
        await socket.send_json(hub.status())
        if hub.last_window is not None:
            await socket.send_json({"type": "window", **asdict(hub.last_window)})
        while True:
            await socket.receive_text()  # Clients send nothing; this detects the close.
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(socket)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
