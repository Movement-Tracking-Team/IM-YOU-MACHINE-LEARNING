"""Replay a recorded trial through LiveEngine and check it matches the offline path.

The streaming engine filters each window with a bounded margin; the offline path
filters the whole recording at once. This measures what that costs, because a
live prediction that disagrees with the published offline one is not usable.
"""
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine import DEVICE_SITE, HOP, LENGTH, LiveEngine, PERFORMED, PERFORMED_INDEX

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                      / "exports" / "biomat_shank_thigh_100hz"))

from predict import infer  # noqa: E402
from preprocessing import CLASSES, HZ, TARGETS, prepare_phone  # noqa: E402

DATASET = Path(r"C:\Users\irhma\OneDrive\Desktop\Motion Tracking\Our Dataset")
# Sensor Logger on iOS with unit standardisation off: raw accel in g, raw gyro in rad/s.
ACCEL, GYRO = "AccelerometerUncalibrated", "GyroscopeUncalibrated"


def read_zip(path):
    with zipfile.ZipFile(path) as archive:
        frames = {name: pd.read_csv(archive.open(f"{name}.csv")) for name in (ACCEL, GYRO)}
    accel, gyro = frames[ACCEL], frames[GYRO]
    seconds = accel.time.astype("int64").to_numpy() / 1e9
    gyro_seconds = gyro.time.astype("int64").to_numpy() / 1e9
    out = pd.DataFrame({"common_timestamp": seconds})
    for axis in "xyz":
        out[f"a{axis}"] = accel[axis].to_numpy()
        out[f"g{axis}"] = np.interp(seconds, gyro_seconds, gyro[axis].to_numpy())
    out["accel_units"] = "g"
    out["gyro_units"] = "rad/s"
    return out.sort_values("common_timestamp").drop_duplicates("common_timestamp")


def find_pair(activity, trial):
    """Locate one trial's two archives. Several were saved with typos in the name."""
    prefix = activity[:5].lower()
    candidates = [p for p in DATASET.glob("*.zip")
                  if p.stem.lower().startswith(prefix) and f"_{trial}-" in p.name]
    if activity.startswith("stair"):
        direction = "ascen" if "ascent" in activity else "desc"
        candidates = [p for p in candidates if direction in p.stem.lower()]

    def pick(marker):
        found = [p for p in candidates if marker in p.stem.lower()]
        if not found:
            raise SystemExit(f'No {marker} archive for {activity} trial {trial}; '
                             f'candidates were {[p.name for p in candidates]}')
        # One stair-ascent shank archive is misnamed phone_1; prefer the longest name,
        # which is the duplicate-suffixed copy the audit resolved by device identity.
        return max(found, key=lambda p: len(p.name))

    thigh = pick("phone_1")
    shank = next((p for p in candidates if "phone_2" in p.stem.lower()
                  or "phoje_2" in p.stem.lower()), None)
    if shank is None:
        raise SystemExit(f'No shank archive for {activity} trial {trial}')
    return thigh, shank


def build_session(thigh_zip, shank_zip, folder):
    read_zip(thigh_zip).to_csv(folder / "phone_1.csv", index=False)
    read_zip(shank_zip).to_csv(folder / "phone_2.csv", index=False)
    return folder


def stream_through_engine(engine, folder, packet=10):
    """Feed both CSVs in timestamp order, in small packets, like the receiver does."""
    tables = {device: pd.read_csv(folder / f"{device}.csv") for device in DEVICE_SITE}
    queues = {device: table.to_dict("records") for device, table in tables.items()}
    cursors = {device: 0 for device in queues}
    windows = []
    while any(cursors[d] < len(queues[d]) for d in queues):
        device = min((d for d in queues if cursors[d] < len(queues[d])),
                     key=lambda d: queues[d][cursors[d]]["common_timestamp"])
        chunk = queues[device][cursors[device]:cursors[device] + packet]
        cursors[device] += len(chunk)
        engine.add_samples(device, chunk, accel_units="g", gyro_units="rad/s")
        windows.extend(engine.drain())
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", default="stair_ascent")
    parser.add_argument("--trial", default="1")
    args = parser.parse_args()

    thigh, shank = find_pair(args.activity, args.trial)
    print(f"thigh: {thigh.name}\nshank: {shank.name}\n")

    folder = Path(tempfile.mkdtemp(prefix="replay_"))
    try:
        build_session(thigh, shank, folder)
        engine = LiveEngine()

        canonical, starts, timeline, moving, info = prepare_phone(folder)
        offline = infer(engine.models, canonical)
        print(f"offline: {len(canonical)} windows, {int(moving.sum())} pass the gate, "
              f"rates {info['rates_hz']}")

        live = stream_through_engine(engine, folder)
        print(f"live:    {len(live)} windows, {sum(w.usable for w in live)} pass the gate\n")

        pairs = min(len(live), len(canonical))
        gate_same = sum(live[i].usable == bool(moving[i]) for i in range(pairs))
        offline_restricted = [PERFORMED[int(np.argmax(offline["classifier"][i][PERFORMED_INDEX]))]
                              for i in range(pairs)]
        activity_same = sum(live[i].activity_unrestricted == CLASSES[int(np.argmax(offline["classifier"][i]))]
                            for i in range(pairs))
        restricted_same = sum(live[i].activity_restricted == offline_restricted[i]
                              for i in range(pairs))

        angle_gap = max(
            max(abs(np.array(live[i].angles[name]) - offline["regressor"][i][:, j]).max()
                for j, name in enumerate(TARGETS))
            for i in range(pairs))
        score_gap = max(max(abs(live[i].scores[name] - offline["classifier"][i][k])
                            for k, name in enumerate(CLASSES))
                        for i in range(pairs))

        print(f"compared {pairs} windows")
        print(f"  gate decision identical      {gate_same}/{pairs}")
        print(f"  5-class argmax identical     {activity_same}/{pairs}")
        print(f"  3-class argmax identical     {restricted_same}/{pairs}")
        print(f"  largest joint-angle gap      {angle_gap:.3f} degrees")
        print(f"  largest class-score gap      {score_gap:.5f}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
