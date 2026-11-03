# Windows 11 / Docker Desktop (WSL2). Resume by rerunning the .bat after restarts.
param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Run-Native {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed ($LASTEXITCODE). Resolve the reported issue, then rerun setup." }
}
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
}
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'Install/update Microsoft App Installer (winget) from the Microsoft Store, then rerun.'
}
$drive = Get-PSDrive -Name (Get-Location).Drive.Name
if ($drive.Free -lt 20GB) { throw 'At least 20 GiB free is required for initial setup.' }
& wsl --status *> $null
if ($LASTEXITCODE -ne 0) {
    Start-Process wsl.exe -ArgumentList '--install --no-distribution' -Verb RunAs -Wait
    throw 'WSL installation requested. Complete any administrator prompts/restart, then rerun this script.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Run-Native -Program winget -Arguments @('install','--id','Docker.DockerDesktop','--exact','--silent','--accept-package-agreements','--accept-source-agreements')
    Refresh-Path
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (Test-Path $desktop) { Start-Process $desktop }
    throw 'Docker Desktop is starting. Finish initial setup, select Linux containers with WSL2, then rerun.'
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Run-Native -Program winget -Arguments @('install','--id','Ollama.Ollama','--exact','--silent','--accept-package-agreements','--accept-source-agreements')
    Refresh-Path
}
try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 5 | Out-Null }
catch {
    Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep -Seconds 5
    Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 10 | Out-Null
}
# Test actual host-loopback access; do not expose Ollama on 0.0.0.0.
& docker run --rm --network host python:3.10-slim python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5).read()"
if ($LASTEXITCODE -ne 0) {
    throw 'Enable Docker Desktop > Settings > Resources > Network > Enable host networking, Apply/restart, then rerun. Keep Ollama running on the Windows host.'
}
$runtime = 'cpu'; $device = 'auto'; $composeFiles = 'compose.yaml'
if ($Cpu) { $device = 'cpu' }
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    & docker run --rm --gpus all python:3.10-slim nvidia-smi
    if ($LASTEXITCODE -eq 0) { $runtime='gpu'; $device='auto'; $composeFiles='compose.yaml;compose.gpu.yaml' }
    else { Write-Host 'CPU fallback: NVIDIA GPU passthrough unavailable. Check Windows NVIDIA driver and Docker WSL2 backend.' }
} else { Write-Host 'CPU fallback: CPU mode selected or no NVIDIA driver detected.' }
# Reuse the installed model without checking the registry or updating its tag.
& ollama show granite3.2-vision:2b *> $null
if ($LASTEXITCODE -eq 0) { Write-Host 'Reusing installed Ollama model: granite3.2-vision:2b' }
else { Run-Native -Program ollama -Arguments @('pull','granite3.2-vision:2b') }
foreach ($group in @('input','output')) {
    foreach ($category in @('audio','text','pdf','excel','image','video','docs')) {
        New-Item -ItemType Directory -Force "$group/$category" | Out-Null
    }
}
foreach ($folder in @('models','runtime','reference_data')) { New-Item -ItemType Directory -Force $folder | Out-Null }
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
$lines = @(Get-Content '.env' | Where-Object { $_ -notmatch '^(RUNTIME|DEVICE|LOCAL_UID|LOCAL_GID|COMPOSE_FILE)=' })
$lines += @("RUNTIME=$runtime","DEVICE=$device",'LOCAL_UID=1000','LOCAL_GID=1000',"COMPOSE_FILE=$composeFiles")
[IO.File]::WriteAllLines((Join-Path (Get-Location) '.env'), $lines, (New-Object Text.UTF8Encoding($false)))
Run-Native -Program docker -Arguments @('compose','build')
Run-Native -Program docker -Arguments @('compose','run','--rm','--no-deps','app','python','-m','scripts.download_models')
Run-Native -Program docker -Arguments @('compose','run','--rm','--no-deps','app','python','-m','scripts.verify_runtime')
Run-Native -Program docker -Arguments @('compose','up','-d')
for ($attempt=0; $attempt -lt 60; $attempt++) {
    try {
        Invoke-RestMethod http://127.0.0.1:8000/api/health -TimeoutSec 3 | Out-Null
        Start-Process http://127.0.0.1:8000
        Write-Host 'Ready: http://127.0.0.1:8000'
        exit 0
    } catch { Start-Sleep -Seconds 2 }
}
throw 'Application did not become healthy. Run docker compose logs app.'
