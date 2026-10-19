# DevOps & MLOps 环境自检脚本
# 用法: powershell -ExecutionPolicy Bypass -File check-env.ps1

$ErrorActionPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..\..')).Path

function Line($name, $val) { '{0,-18} {1}' -f $name, $val }

function Find-Executable($name, $fallbacks) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($path in $fallbacks) {
        if ($path -and (Test-Path -LiteralPath $path)) { return $path }
    }
    return $null
}

Write-Output "===== DevOps & MLOps 环境自检 ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ====="
Write-Output "仓库: $repoRoot"

Write-Output "`n[Python]"
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $python) {
    Line 'python' (& $python --version 2>&1)
    Line 'source' '.venv\Scripts\python.exe'
} else {
    $python = Find-Executable 'python' @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    )
    if ($python) {
        Line 'python' (& $python --version 2>&1)
        Line 'source' $python
    } else {
        Line 'python' 'NOT FOUND (create .venv)'
    }
}

Write-Output "`n[Python 包]"
if ($python) {
    & $python -X utf8 -c "import importlib.metadata as m
for p in ['jupyterlab','mlflow','dvc','flask','fastapi','uvicorn','scikit-learn']:
    try: print(f'{p:15s} {m.version(p)}')
    except Exception: print(f'{p:15s} MISSING')"
} else {
    Write-Output '  skipped: Python not found'
}

Write-Output "`n[Git]"
$git = Find-Executable 'git' @(
    (Join-Path $env:ProgramFiles 'Git\cmd\git.exe')
)
if ($git) { Line 'git' (& $git --version 2>&1) } else { Line 'git' 'NOT FOUND' }

Write-Output "`n[Docker]"
$docker = Find-Executable 'docker' @(
    (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe')
)
if ($docker) {
    Line 'docker(cli)' (& $docker --version 2>&1)
    $server = & $docker info --format '{{.ServerVersion}}' 2>$null
    if ($LASTEXITCODE -eq 0 -and $server) {
        Line 'docker(engine)' $server
    } else {
        Line 'docker(engine)' 'NOT RUNNING (start Docker Desktop)'
    }
} else {
    Line 'docker' 'NOT FOUND'
}

Write-Output "`n[WSL]"
$wsl = Find-Executable 'wsl.exe' @(
    (Join-Path $env:SystemRoot 'System32\wsl.exe')
)
if ($wsl) {
    $tmp = [System.IO.Path]::GetTempFileName()
    & cmd.exe /c "`"$wsl`" --version > `"$tmp`" 2>&1"
    $wslVersion = Get-Content -LiteralPath $tmp -Encoding Unicode | Select-Object -First 1
    if (-not $wslVersion) {
        $wslVersion = Get-Content -LiteralPath $tmp | Select-Object -First 1
    }
    Remove-Item -LiteralPath $tmp -Force
    Line 'wsl' $wslVersion
} else {
    Line 'wsl' 'NOT FOUND'
}

Write-Output "`n[GitHub CLI]"
$gh = Find-Executable 'gh' @(
    (Join-Path $env:ProgramFiles 'GitHub CLI\gh.exe')
)
if ($gh) { Line 'gh' ((& $gh --version 2>&1) | Select-Object -First 1) } else { Line 'gh' 'NOT FOUND' }

Write-Output "`n[Azure CLI]"
$az = Find-Executable 'az' @(
    (Join-Path $env:ProgramFiles 'Microsoft SDKs\Azure\CLI2\wbin\az.cmd')
)
if ($az) {
    $azVersionJson = & $az version --output json 2>$null | Out-String
    try { $azVersion = ($azVersionJson | ConvertFrom-Json).'azure-cli' } catch { $azVersion = 'installed' }
    Line 'az' $azVersion
} else {
    Line 'az' 'NOT FOUND'
}

Write-Output "`n[IDE]"
$code = Find-Executable 'code' @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Microsoft VS Code\bin\code.cmd')
)
if ($code) { Line 'VS Code' 'installed' } else { Line 'VS Code' 'NOT FOUND' }

Write-Output "`n[DVC 远端]"
$dvc = Join-Path $repoRoot '.venv\Scripts\dvc.exe'
if (Test-Path -LiteralPath $dvc) {
    & $dvc remote list 2>&1 | ForEach-Object { Write-Output "  $_" }
} else {
    Write-Output '  skipped: .venv\dvc.exe not found'
}

Write-Output "`n===== 完成 ====="
