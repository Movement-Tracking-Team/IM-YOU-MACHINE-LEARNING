$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = '100 Hz classifier - phone adaptation with held-out trials'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$adaptationLogDirectory = Join-Path (Get-Location) 'Camargo et al. Dataset/two_sensor/canonical100_v1/phone_adaptation_v1'
New-Item -ItemType Directory -Force -Path $adaptationLogDirectory | Out-Null
Write-Host 'Classifier adaptation: whole-trial holdouts, Camargo replay, fixed 20 epochs.' -ForegroundColor Cyan
Write-Host 'The original classifiers and BioMAT angle models remain unchanged.'
& '..\.biomat-review-env\Scripts\python.exe' -u -m biomat.phone_classifier_adaptation 2>&1 | Tee-Object -FilePath (Join-Path $adaptationLogDirectory 'training.log') -Append
if ($LASTEXITCODE -ne 0) { throw "Classifier adaptation failed with exit code $LASTEXITCODE" }
Write-Host 'Finished. See reports/phone_classifier_adaptation_v1/README.md.' -ForegroundColor Green
