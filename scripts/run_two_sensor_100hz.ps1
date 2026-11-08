$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = '100 Hz Camargo - shank + thigh model training'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$trainingLogDirectory = Join-Path (Get-Location) 'Camargo et al. Dataset/two_sensor/canonical100_v1'
New-Item -ItemType Directory -Force -Path $trainingLogDirectory | Out-Null
Write-Host '100 Hz Camargo training: 250 samples, 12 channels (shank + thigh).' -ForegroundColor Cyan
Write-Host 'Classifier first, then BioMAT. Five subject folds plus final held-out evaluation.'
Write-Host 'The 60 Hz models remain available in canonical60_v1. No phone data is used.'
Write-Host "Training log: $trainingLogDirectory\training.log"
& '..\.biomat-review-env\Scripts\python.exe' -u -m biomat.two_sensor_100hz 2>&1 | Tee-Object -FilePath (Join-Path $trainingLogDirectory 'training.log') -Append
if ($LASTEXITCODE -ne 0) { throw "100 Hz training failed with exit code $LASTEXITCODE" }
& '..\.biomat-review-env\Scripts\python.exe' -u -m biomat.two_sensor_100hz_report 2>&1 | Tee-Object -FilePath (Join-Path $trainingLogDirectory 'training.log') -Append
if ($LASTEXITCODE -ne 0) { throw "100 Hz report failed with exit code $LASTEXITCODE" }
Write-Host 'Training complete. Models and verified results saved.' -ForegroundColor Green
