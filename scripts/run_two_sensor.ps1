$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'Camargo thigh + shank - classifier and BioMAT training'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$logDirectory = Join-Path (Get-Location) 'Camargo et al. Dataset/two_sensor/canonical60_v1'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
& '..\.biomat-review-env\Scripts\python.exe' -u -m biomat.two_sensor 2>&1 | Tee-Object -FilePath (Join-Path $logDirectory 'training.log') -Append
if ($LASTEXITCODE -ne 0) { throw "Two-sensor training failed with exit code $LASTEXITCODE" }
