$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$trainingPython = Join-Path $projectRoot 'work\train-2b-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trainingPython)) {
    python -m venv work/train-2b-env
    if ($LASTEXITCODE -ne 0) { throw 'Training environment creation failed.' }
}
& $trainingPython -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $trainingPython -m pip install -r training/requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Training package installation failed.' }
& $trainingPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Training dependency check failed.' }
Write-Output 'Separate training environment ready. Model download and smoke test are separate commands.'
