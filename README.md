# Movement Tracking: BioMAT reproduction

This project trains a paper-configured adaptation of [BioMAT](https://github.com/MohsenSharifi1991/BioMAT), upstream commit `0ca003db504514da1a45d2a2d9339934f6b68202`, on the locally stored Camargo dataset.

The reproduction is an experiment, not a claim of having matched the published result. See [the initial review](reports/biomat_review.md) and [the current run protocol](reports/reproduction_protocol.md).

## Setup

Python 3.11 and a CUDA-capable GPU are used. The current machine has a working environment at `../.biomat-review-env`. For a new checkout:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-reproduction.txt
```

Place original `AB06.zip` through `AB21.zip`, plus `AB23.zip`, `AB24.zip`, and `AB25.zip`, in `Camargo et al. Dataset`. Data, windows, logs, predictions and checkpoints stay inside that ignored folder.

## Prepare and train

For the current environment, run these from the repository directory:

```powershell
..\.biomat-review-env\Scripts\python.exe -m biomat.prepare
..\.biomat-review-env\Scripts\python.exe -m unittest discover -s tests -v
..\.biomat-review-env\Scripts\python.exe -m biomat.train --mode smoke
..\.biomat-review-env\Scripts\python.exe -u -m biomat.train --mode all
```

Use `.venv/Scripts/python.exe` instead for a newly created environment. The parity tests require the original BioMAT checkout in the sibling `BioMAT-review` directory. Tests compare model outputs and input gradients, activity selection, resampling/windowing, label decoding, and evaluation aggregation.

The full run trains five fresh models on subject folds, followed by a sixth model on all 16 development subjects. Every model trains for 50 epochs. The final model sees AB23/AB24/AB25 only after training. Additional subjects AB27/AB28/AB30 are not used.

## Progress and resuming

The active separate PowerShell window shows live training output. Keep that window open: it owns the training process.

Artifacts are in `Camargo et al. Dataset/biomat_reproduction/runs/paper_standard_v1/`:

- `training.log`: output captured by the initially opened PowerShell window.
- `fold_1` through `fold_5`, then `final`: each run's protocol, epoch log, status, scaler, checkpoints and final metrics.
- `latest.pt`: resumable model, optimizer, epoch and PyTorch/loader RNG state, saved after each epoch.
- `model.pt`: inference model saved after completing that run.
- `predictions.npz`: predictions, target angles, valid-frame lengths, subjects and activities.
- `summary.json`: written when the requested sequence completes.

After an interruption, rerun the same command with the same settings to resume. Do not start a second process for the same experiment while the first is active. Use a new `--experiment` name when changing settings. `--mode cv --fold 1` runs a single fold; `--mode final` runs only final training.

The reported validation RMSE is pooled across windows and averaged across joints. Final metrics also include valid-frame errors and per-subject/activity scores. Early validation values are not final test accuracy.

## Movement classification

The separate five-class movement classifier is trained from scratch using the prepared IMU windows. Run it with:

```powershell
..\.biomat-review-env\Scripts\python.exe -u -m biomat.classifier
```

It compares a 1D CNN against Extra Trees on five subject folds, selects the model family before final testing, and saves both final models. Completed experiments are not rerun automatically. Results are in [the classifier report](reports/movement_classifier_results.md). See [the three-phone guide](reports/phone_classifier_guide.md) for CSV input and calibration requirements. Phone accuracy remains unvalidated.

## Public phone benchmark

The [UCI HAR experiment](reports/phone_benchmark_results.md) adapts the same CNN to six channels from one waist-mounted phone and trains it from scratch. It tests unseen participants within a phone dataset, not transfer of existing Camargo weights to phones. The original three-phone adapter is not compatible with this checkpoint.

```powershell
..\.biomat-review-env\Scripts\python.exe -u -m biomat.phone_benchmark
```

To keep live output in a separate terminal, run `scripts/run_phone_benchmark.ps1` there. The experiment downloads the official archive, verifies its checksum and data integrity, uses five subject folds within the official training partition, and evaluates the final model on the official test partition after 30 fixed epochs. Data, logs, and checkpoints stay under the ignored `Camargo et al. Dataset/phone_benchmark/uci_har/` directory. Completed folds are reused; an interrupted fold restarts from its seeded initialization. Completed experiments are not retrained. Use a new output directory and protocol for any future experiment rather than overwriting this baseline.

## Camargo with thigh and shank only

The separate two-sensor experiment trains a 12-channel MovementCNN and BioMAT on Camargo data at 60 Hz with a shared, signal-derived orientation normalization. It does not use UCI HAR, existing three-sensor weights, or phone labels for training. See [results and limitations](reports/two_sensor_results.md).

```powershell
..\.biomat-review-env\Scripts\python.exe -u -m biomat.two_sensor
..\.biomat-review-env\Scripts\python.exe -m biomat.two_sensor_report
..\.biomat-review-env\Scripts\python.exe -m biomat.two_sensor_phone phone-imu/recordings/session_cbb59a86-dbbc-4a4d-8046-213dd4ad3430
```

`scripts/run_two_sensor.ps1` displays training and captures its log. Interrupted training resumes from saved optimizer and RNG state; completed models are reused. Artifacts stay in the ignored `Camargo et al. Dataset/two_sensor/canonical60_v1/`. Phone inference requires the completed experiment and the receiver's exported CSV schema (phone 1 thigh, phone 2 shank). Outputs are exploratory: they have no measured phone angle error and must not be presented as validated joint angles. Low-motion windows are omitted using a fixed gate, not a learned standing class. The original three-phone adapter is unchanged.
