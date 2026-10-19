# 启用 WSL2 / 虚拟机平台并安装 WSL
# 用法: 右键“以管理员身份运行 PowerShell”，然后执行
#   powershell -ExecutionPolicy Bypass -File install-wsl.ps1
# 安装完成后必须重启 Windows。

$ErrorActionPreference = 'Continue'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path (Split-Path -Parent $scriptDir) 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("wsl-install-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host '请以管理员身份运行此脚本（右键 PowerShell -> 以管理员身份运行）。' -ForegroundColor Red
    exit 1
}

$transcript = $false
try {
    Start-Transcript -Path $log | Out-Null
    $transcript = $true
} catch {
    Write-Host "无法启动日志记录，继续安装。原因: $($_.Exception.Message)" -ForegroundColor Yellow
}

try {
    Write-Host "===== $(Get-Date -Format o) : start ====="

    foreach ($feature in 'Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform') {
        Write-Host "--- 启用 $feature ---"
        & dism.exe /online /enable-feature /featurename:$feature /all /norestart
        $code = $LASTEXITCODE
        if ($code -eq 0 -or $code -eq 3010) {
            Write-Host "$feature 已启用（exit code: $code，3010 表示需要重启）" -ForegroundColor Green
        } else {
            Write-Host "$feature 启用失败（exit code: $code）" -ForegroundColor Red
        }
    }

    Write-Host '--- wsl --install --no-distribution ---'
    & wsl.exe --install --no-distribution
    Write-Host "wsl --install exit code: $LASTEXITCODE"

    Write-Host '--- wsl --set-default-version 2 ---'
    & wsl.exe --set-default-version 2
    Write-Host "wsl --set-default-version exit code: $LASTEXITCODE"

    Write-Host '===== 完成，请重启 Windows 以启用 WSL2 / 虚拟机平台 =====' -ForegroundColor Green
} finally {
    Write-Host "日志文件: $log"
    if ($transcript) { Stop-Transcript | Out-Null }
}
