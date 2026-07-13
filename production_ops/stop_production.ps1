param(
    [switch]$Force,
    [int]$GraceSeconds = 90
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $RepoRoot "logs\factory_ai.pid"
$StopFile = Join-Path $RepoRoot "logs\factory_ai.stop"

Write-Host "Factory AI - stopping production system..."

New-Item -ItemType Directory -Path (Join-Path $RepoRoot "logs") -Force | Out-Null

$targetPids = @()

if (Test-Path -LiteralPath $PidFile) {
    try {
        $pidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($proc) {
            $targetPids += $pidValue
        } else {
            Write-Host "PID $pidValue not running."
        }
    } catch {
        Write-Host "PID file read/stop error: $_"
    }
}

$pythonProcs = Get-Process -Name python -ErrorAction SilentlyContinue
foreach ($proc in $pythonProcs) {
    try {
        $cmd = ""
        try {
            $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" -ErrorAction Stop).CommandLine
        } catch {
            $cmd = ""
        }
        if ($cmd -like "*main.py*" -or $cmd -eq "") {
            if ($targetPids -notcontains $proc.Id) {
                $targetPids += $proc.Id
            }
        }
    } catch {
        Write-Host "Could not inspect PID $($proc.Id): $_"
    }
}

if ($targetPids.Count -eq 0) {
    Write-Host "No running Factory AI main.py process found."
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
    exit 0
}

Write-Host "Requesting graceful stop for PID(s): $($targetPids -join ', ')"
Set-Content -LiteralPath $StopFile -Value "stop requested $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8

$deadline = (Get-Date).AddSeconds($GraceSeconds)
while ((Get-Date) -lt $deadline) {
    $alive = @()
    foreach ($pidValue in $targetPids) {
        if (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
            $alive += $pidValue
        }
    }
    if ($alive.Count -eq 0) {
        Write-Host "Factory AI stopped gracefully. Recordings and report finalized."
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
        exit 0
    }
    Start-Sleep -Seconds 2
}

Write-Host "Graceful stop timeout after $GraceSeconds seconds."
if ($Force) {
    foreach ($pidValue in $targetPids) {
        if (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
            Write-Host "Force stopping PID $pidValue..."
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Process still running. Use .\production_ops\stop_production.ps1 -Force only if emergency."
exit 1
