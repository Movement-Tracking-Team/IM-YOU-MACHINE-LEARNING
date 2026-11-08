# Camargo validation dashboard

From the repository directory in PowerShell:

```powershell
uv pip install --python ..\.biomat-review-env\Scripts\python.exe -r dashboard/requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/run_validation_dashboard.ps1
```

Open http://localhost:8501 after Streamlit starts. The launcher binds only to this computer.
Keep the terminal open. Stop the server with Ctrl+C. A different Python environment can run
`python -m streamlit run dashboard/app.py` after installing the same requirements.

The dashboard opens the saved validation report, supports subject/activity filters, compares
the original and phone-adapted classifiers, and plots BioMAT predictions against reference joint angles.
It includes CSV/JSON downloads and a button to rerun inference. It does not train models.

Fresh checkpoint inference on 1,910 held-out windows reproduced the previous saved predictions:

| Measure | Result |
| --- | ---: |
| Original classifier accuracy | 95.71% |
| Phone-adapted classifier accuracy | 91.68% |
| Hip / knee / ankle RMSE | 9.24° / 7.10° / 5.26° |

Six backend tests passed, covering held-out subject enforcement, checkpoint compatibility,
prediction reproduction, source hashes, filtered metrics and exported window metrics.
Python source and PowerShell launcher syntax checks passed. The Streamlit UI has not been
runtime-tested in this environment; see the dependency limitation below.

Run validation without Streamlit:

```powershell
..\.biomat-review-env\Scripts\python.exe -m dashboard.backend
```

Required local files: the three checkpoints listed in `backend.CHECKPOINTS` and the prepared
AB23, AB24, AB25 NPZ files under `Camargo et al. Dataset/two_sensor/canonical100_v1/prepared`.
The existing repository model code is required too; this folder alone is not a portable model package.
Reports are written to `reports/camargo_dashboard_validation`. CUDA is used when available;
CPU inference is supported and slower. Concurrent revalidation in one server process is blocked.

The test subjects were excluded from training but evaluated in earlier experiments.
These are repeat held-out results, not a new cohort. Correlated overlapping windows do not
provide independent trial counts. The adapted classifier has been trained on the phone dataset;
this dashboard makes no new phone-generalization claim.

Development environment limitation: Streamlit was absent and package installation failed because
the configured network proxy was unreachable. Backend validation can run with the existing Python
environment; the Streamlit interface must be launched and checked after dependencies are installed.
