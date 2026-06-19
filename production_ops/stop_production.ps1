param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile  = Join-Path $RepoRoot "logs\factory_ai.pid"

Write-Host "Factory AI — stopping production system..."

# 1. Try PID file first (clean stop)
if (Test-Path $PidFile) {
    try {
        $pid_val = [int](Get-Content $PidFile -Raw).Trim()
        $proc = Get-Process -Id $pid_val -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Stopping process PID $pid_val ($($proc.Name))..."
            Stop-Process -Id $pid_val -Force:$Force
            Start-Sleep -Seconds 3
            Write-Host "Process stopped."
        } else {
            Write-Host "PID $pid_val not running (already stopped)."
        }
    } catch {
        Write-Host "PID file read error: $_"
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# 2. Fallback: kill any python.exe running main.py
$procs = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like "*main.py*" }

if ($procs) {
    foreach ($p in $procs) {
        Write-Host "Killing python main.py (PID $($p.ProcessId))..."
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Done."
} else {
    Write-Host "No running Factory AI process found."
}
