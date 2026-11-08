"""Read local Camargo ZIPs; write aggregate QC without extracting the dataset.

Requires numpy, pandas and mat-io. Outputs stay inside the ignored data folder.
This checks raw trials, not BioMAT activity segmentation or model performance.
"""
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
from matio import load_from_mat


ROOT = Path(__file__).resolve().parents[1] / "Camargo et al. Dataset"
FEATURES = [f"{s}_{f}" for s in ("foot", "shank", "thigh")
            for f in ("Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z")]
TARGETS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
MODES = ("levelground", "ramp", "stair")


def main():
    started = time.monotonic()
    records, archives, failures = [], [], []
    with tempfile.TemporaryDirectory(prefix="audit_", dir=ROOT) as tmp:
        temp = Path(tmp) / "current.mat"
        for archive in sorted(ROOT.glob("AB*.zip")):
            subject_records = []
            with zipfile.ZipFile(archive) as z:
                members = [i for i in z.infolist() if not i.is_dir()]
                names = {i.filename for i in members}
                archives.append({"subject": archive.stem, "compressed_bytes": archive.stat().st_size,
                                 "uncompressed_bytes": sum(i.file_size for i in members),
                                 "files": len(members)})
                trials = sorted(n for n in names if "/imu/" in n and n.endswith(".mat")
                                and n.split("/")[-3] in MODES)
                for name in trials:
                    record = {"subject": archive.stem, "trial": name,
                              "mode": name.split("/")[-3]}
                    try:
                        tables = {}
                        for modality in ("imu", "ik", "conditions"):
                            # Explicit trial-key matching, never ZIP or directory listing order.
                            partner = name.replace("/imu/", f"/{modality}/")
                            temp.write_bytes(z.read(partner))
                            loaded = load_from_mat(temp)
                            key = "labels" if modality == "conditions" else "data"
                            table = loaded[key]
                            if not isinstance(table, pd.DataFrame):
                                raise TypeError(f"{modality} is not a DataFrame")
                            tables[modality] = table
                            record[f"{modality}_rows"] = len(table)
                            stamp = table["Header"].to_numpy(dtype=float)
                            dt = np.diff(stamp)
                            record[f"{modality}_time_finite"] = bool(np.isfinite(stamp).all())
                            record[f"{modality}_time_increasing"] = bool((dt > 0).all())
                            record[f"{modality}_median_dt"] = float(np.median(dt)) if len(dt) else None
                            record[f"{modality}_irregular_steps"] = int(np.sum(~np.isclose(dt, 0.005, atol=1e-7)))
                        x = tables["imu"][FEATURES].to_numpy(dtype=float)
                        y = tables["ik"][TARGETS].to_numpy(dtype=float)
                        record["input_nonfinite"] = int((~np.isfinite(x)).sum())
                        record["target_nonfinite"] = int((~np.isfinite(y)).sum())
                        record["constant_inputs"] = [FEATURES[i] for i in np.where(np.ptp(x, axis=0) == 0)[0]]
                        record["target_min"] = np.nanmin(y, axis=0).tolist()
                        record["target_max"] = np.nanmax(y, axis=0).tolist()
                        record["upstream_angle_rule_flags_raw_trial"] = bool(np.nanmin(y[:, 2]) < -100 or np.nanmax(y[:, 0]) > 100)
                        stamp = tables["imu"]["Header"].to_numpy(dtype=float)
                        record["duration_seconds"] = float(stamp[-1] - stamp[0])
                        record["aligned"] = all(len(t) == len(stamp) and np.allclose(t["Header"], stamp, rtol=0, atol=1e-7)
                                                for t in tables.values())
                        labels = tables["conditions"]["Label"]
                        record["missing_labels"] = int(labels.isna().sum())
                        record["label_counts"] = {str(k): int(v) for k, v in labels.value_counts().items()}
                        subject_records.append(record)
                    except Exception as error:
                        failures.append({**record, "error": f"{type(error).__name__}: {error}"})
            records.extend(subject_records)
            print(f"{archive.stem}: {len(subject_records)} trials checked; {len(failures)} cumulative read errors", flush=True)
    labels = Counter()
    for r in records:
        labels.update(r["label_counts"])
    summary = {
        "scope": "All raw levelground/ramp/stair IMU, IK and conditions triples; no treadmill/static waveform QC.",
        "subjects": len(archives), "trials_checked": len(records), "read_errors": len(failures),
        "compressed_GB": sum(a["compressed_bytes"] for a in archives) / 1e9,
        "modes": dict(Counter(r["mode"] for r in records)),
        "samples": sum(r["imu_rows"] for r in records),
        "raw_trial_hours": sum(r["duration_seconds"] for r in records) / 3600,
        "misaligned_trials": sum(not r["aligned"] for r in records),
        "trials_with_nonfinite_input": sum(r["input_nonfinite"] > 0 for r in records),
        "trials_with_nonfinite_target": sum(r["target_nonfinite"] > 0 for r in records),
        "trials_with_constant_input": sum(bool(r["constant_inputs"]) for r in records),
        "trials_with_missing_labels": sum(r["missing_labels"] > 0 for r in records),
        "trials_flagged_by_upstream_angle_rule": sum(r["upstream_angle_rule_flags_raw_trial"] for r in records),
        "trials_with_time_issues": sum(any(not r[f"{m}_time_finite"] or not r[f"{m}_time_increasing"]
                                              or r[f"{m}_irregular_steps"] for m in ("imu", "ik", "conditions")) for r in records),
        "target_min": np.min([r["target_min"] for r in records], axis=0).tolist() if records else [],
        "target_max": np.max([r["target_max"] for r in records], axis=0).tolist() if records else [],
        "label_counts": dict(labels), "elapsed_seconds": time.monotonic() - started,
    }
    (ROOT / "biomat_quality_audit.json").write_text(json.dumps({"summary": summary, "archives": archives,
                                                              "trials": records, "failures": failures}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
