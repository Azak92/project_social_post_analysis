$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$runtime = Join-Path $PSScriptRoot 'work\ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $runtime)) { throw 'The local Ollama runtime has not been installed.' }
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $PSScriptRoot 'work\models'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_KEEP_ALIVE = '5m'
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null }
catch {
    Start-Process -FilePath $runtime -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput 'work\ollama.stdout.log' -RedirectStandardError 'work\ollama.stderr.log' | Out-Null
}
try { Invoke-RestMethod 'http://127.0.0.1:8000/api/models' -TimeoutSec 2 | Out-Null }
catch {
    Start-Process -FilePath 'python' -ArgumentList '-m','uvicorn','lab.app:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'work\app.stdout.log' -RedirectStandardError 'work\app.stderr.log' | Out-Null
}
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null
        Invoke-RestMethod 'http://127.0.0.1:8000/api/models' -TimeoutSec 2 | Out-Null
        Write-Output 'Language Lab is ready: http://127.0.0.1:8000'
        exit 0
    } catch { Start-Sleep -Seconds 1 }
}
throw 'The lab did not start in time. See the app and Ollama logs in work.'
