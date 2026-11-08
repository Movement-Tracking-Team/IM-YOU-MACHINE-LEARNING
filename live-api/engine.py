"""Streaming version of the offline canonical100_v1 pipeline.

`predict.py` in the export folder reads whole CSV files. This does the same work
incrementally so predictions can come out while a session is still running. The
preprocessing must match the offline path exactly or the checkpoints are being
fed something they were not trained on, which is the mistake that made Judy's
dashboard look broken.
"""
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULT_EXPORT = (Path(__file__).resolve().parents[2]
                  / "exports" / "biomat_shank_thigh_100hz")
EXPORT_DIR = Path(os.environ.get("BIOMAT_EXPORT", DEFAULT_EXPORT)).resolve()
# The export ships preprocessing/model/classifier as a self-contained unit that
# imports its own siblings, so it has to be importable as a top-level package.
sys.path.insert(0, str(EXPORT_DIR))

from predict import infer, load_models  # noqa: E402
from preprocessing import CLASSES, HZ, LENGTH, SITES, TARGETS, canonicalize  # noqa: E402

HOP = 50
#  The app offers these; the model's two ramp classes were never performed on a
#  phone and absorbed most of the errors on our own labelled trials.
PERFORMED = ["walking", "stair_ascent", "stair_descent"]
PERFORMED_INDEX = [CLASSES.index(name) for name in PERFORMED]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
# phone_1 is the right thigh and phone_2 the right shank; the model wants shank first.
DEVICE_SITE = {"phone_2": 0, "phone_1": 1}
SOS = butter(4, 20, fs=HZ, output="sos")
# sosfiltfilt is two-sided. Giving each window this much context on both sides
# keeps the live filter close to the offline one, at the cost of HOP/HZ seconds.
MARGIN = 50


@dataclass
class Window:
    index: int
    start_s: float
    end_s: float
    usable: bool
    reason: str | None
    activity: str | None
    """Best of the three performed activities, even when the gate rejected the window."""
    activity_restricted: str
    activity_unrestricted: str
    confidence: float
    scores: dict[str, float]
    angles: dict[str, list[float]]
    summary: dict[str, dict[str, float]]


@dataclass
class _Stream:
    """Timestamps and six channels for one phone, oldest samples dropped."""
    times: deque = field(default_factory=lambda: deque(maxlen=4000))
    values: deque = field(default_factory=lambda: deque(maxlen=4000))

    def add(self, timestamp, row):
        # Browser timestamps can repeat or arrive late; interpolation needs them sorted.
        if self.times and timestamp <= self.times[-1]:
            return
        self.times.append(timestamp)
        self.values.append(row)

    def covers(self, start, stop):
        return bool(self.times) and self.times[0] <= start and self.times[-1] >= stop

    def resample(self, grid):
        times = np.fromiter(self.times, dtype=np.float64, count=len(self.times))
        values = np.array(self.values, dtype=np.float64)
        return np.column_stack([np.interp(grid, times, values[:, j]) for j in range(6)])

    def rate_hz(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


class LiveEngine:
    """Turns two live phone streams into one prediction every HOP samples."""

    def __init__(self, models=None, margin=MARGIN):
        self.models = models if models is not None else load_models()
        self.margin = margin
        self.reset()

    def reset(self):
        self.streams = {device: _Stream() for device in DEVICE_SITE}
        self.origin = None
        self.next_start = 0
        self.emitted = 0

    def add_samples(self, device_id, samples, accel_units="m/s^2", gyro_units="deg/s"):
        """Feed one packet. Units are converted to the m/s^2 and rad/s the model expects."""
        if device_id not in self.streams:
            return
        accel_gain = 9.80665 if accel_units == "g" else 1.0
        gyro_gain = np.pi / 180.0 if gyro_units == "deg/s" else 1.0
        stream = self.streams[device_id]
        for sample in samples:
            row = [sample["ax"] * accel_gain, sample["ay"] * accel_gain, sample["az"] * accel_gain,
                   sample["gx"] * gyro_gain, sample["gy"] * gyro_gain, sample["gz"] * gyro_gain]
            if not all(np.isfinite(row)) or not np.isfinite(sample["common_timestamp"]):
                continue
            stream.add(float(sample["common_timestamp"]), row)

    def rates(self):
        return {device: round(stream.rate_hz(), 2) for device, stream in self.streams.items()}

    def _overlap(self):
        if not all(stream.times for stream in self.streams.values()):
            return None
        return (max(s.times[0] for s in self.streams.values()),
                min(s.times[-1] for s in self.streams.values()))

    def drain(self):
        """Every window whose data has fully arrived since the last call."""
        return [window for window in iter(self._next_window, None)]

    def _next_window(self):
        overlap = self._overlap()
        if overlap is None:
            return None
        first, last = overlap
        if self.origin is None:
            self.origin = first
        # Both the window and its filter margin must already be in the buffers.
        need_from = self.origin + (self.next_start - self.margin) / HZ
        need_to = self.origin + (self.next_start + LENGTH - 1 + self.margin) / HZ
        if max(need_from, self.origin) < first or need_to > last:
            return None

        left = min(self.margin, self.next_start)
        grid = self.origin + np.arange(self.next_start - left,
                                       self.next_start + LENGTH + self.margin) / HZ
        if not all(s.covers(grid[0], grid[-1]) for s in self.streams.values()):
            return None

        columns = [None, None]
        for device, stream in self.streams.items():
            columns[DEVICE_SITE[device]] = stream.resample(grid)
        padded = sosfiltfilt(SOS, np.concatenate(columns, axis=1), axis=0)
        raw = padded[left:left + LENGTH][None]

        window = self._score(raw, self.next_start)
        self.next_start += HOP
        self.emitted += 1
        return window

    def _score(self, raw, start_index):
        start_s = start_index / HZ
        base = dict(index=self.emitted, start_s=round(start_s, 3),
                    end_s=round(start_s + (LENGTH - 1) / HZ, 3))
        try:
            canonical, quality = canonicalize(raw)
        except ValueError as exc:
            return Window(**base, usable=False, reason=str(exc), activity=None,
                          activity_restricted="", activity_unrestricted="",
                          confidence=0.0, scores={}, angles={}, summary={})

        reason = self._gate(raw, quality)
        output = infer(self.models, canonical)
        scores = output["classifier"][0]
        angles = output["regressor"][0]

        restricted = PERFORMED[int(np.argmax(scores[PERFORMED_INDEX]))]
        return Window(
            **base,
            usable=reason is None,
            reason=reason,
            activity=restricted if reason is None else None,
            activity_restricted=restricted,
            activity_unrestricted=CLASSES[int(np.argmax(scores))],
            confidence=round(float(scores[PERFORMED_INDEX].max() / scores[PERFORMED_INDEX].sum()), 4),
            scores={name: round(float(value), 5) for name, value in zip(CLASSES, scores)},
            angles={name: np.round(angles[:, j], 2).tolist() for j, name in enumerate(TARGETS)},
            summary={name: {"min": round(float(angles[:, j].min()), 2),
                            "max": round(float(angles[:, j].max()), 2),
                            "mean": round(float(angles[:, j].mean()), 2)}
                     for j, name in enumerate(TARGETS)},
        )

    @staticmethod
    def _gate(raw, quality):
        """The unchanged acquisition gate. It is not a trained 'standing' class."""
        for site in range(2):
            gyro = raw[:, :, site * 6 + 3:site * 6 + 6]
            if (np.linalg.norm(gyro, axis=2) > 0.25).mean() < 0.75:
                return f"not enough movement at the {SITES[site]}"
            if quality[site]["principal_axis_fraction"][0] < 0.6:
                return f"rotation axis unstable at the {SITES[site]}"
        return None
