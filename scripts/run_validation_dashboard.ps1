$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$dashboardPython = Join-Path $workspaceRoot '.biomat-review-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $dashboardPython)) {
    $dashboardPython = (Get-Command python -ErrorAction Stop).Source
}
Push-Location $repoRoot
try {
    & $dashboardPython -c "import streamlit"
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Install the dashboard dependencies, then run this launcher again:'
        Write-Host "uv pip install --python `"$dashboardPython`" -r dashboard/requirements.txt"
        exit 1
    }
    & $dashboardPython -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --theme.base dark --theme.primaryColor '#64d8c4' --theme.backgroundColor '#0b1522' --theme.secondaryBackgroundColor '#13283c' --browser.gatherUsageStats false
} finally {
    Pop-Location
}
