"""Connect to /live like the app would and report what arrives."""
import argparse
import asyncio
import json
from collections import Counter

import websockets


async def main(url, seconds):
    counts = Counter()
    windows = []
    async with websockets.connect(url) as socket:
        try:
            async with asyncio.timeout(seconds):
                async for raw in socket:
                    message = json.loads(raw)
                    counts[message["type"]] += 1
                    if message["type"] == "hello":
                        print(f"contract: {message['seconds_per_window']}s window every "
                              f"{message['seconds_between_windows']}s, "
                              f"reporting {message['reported_classes']}")
                    elif message["type"] == "status":
                        print(f"status: receiver={message['receiver_connected']} "
                              f"recording={message['recording']} hz={message['observed_hz']}")
                    elif message["type"] == "window":
                        windows.append(message)
                        print(f"  window {message['index']:3d} "
                              f"[{message['start_s']:6.2f}-{message['end_s']:6.2f}s] "
                              f"{'ACCEPTED ' + str(message['activity']) if message['usable'] else 'rejected: ' + str(message['reason'])}")
        except asyncio.TimeoutError:
            pass

    print(f"\nmessages: {dict(counts)}")
    if windows:
        accepted = [w for w in windows if w["usable"]]
        print(f"windows: {len(windows)} received, {len(accepted)} passed the gate")
        sample = windows[-1]
        if sample["angles"]:
            for joint, series in sample["angles"].items():
                print(f"  {joint}: {len(series)} samples, "
                      f"{min(series):.1f} to {max(series):.1f} degrees")
    else:
        print("no windows arrived")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/live")
    parser.add_argument("--seconds", type=float, default=30)
    asyncio.run(main(**vars(parser.parse_args())))
