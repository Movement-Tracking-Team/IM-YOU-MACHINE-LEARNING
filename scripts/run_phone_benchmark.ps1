$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'Phone CNN - UCI HAR subject-disjoint training'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$logDirectory = Join-Path (Get-Location) 'Camargo et al. Dataset/phone_benchmark/uci_har/cnn_v1'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
& '..\.biomat-review-env\Scripts\python.exe' -u -m biomat.phone_benchmark 2>&1 | Tee-Object -FilePath (Join-Path $logDirectory 'training.log')
if ($LASTEXITCODE -ne 0) { throw "Phone benchmark failed with exit code $LASTEXITCODE" }
