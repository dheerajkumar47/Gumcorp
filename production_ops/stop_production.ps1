param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $RepoRoot "logs\factory_ai.pid"

Write-Host "Factory AI - stopping production system..."

if (Test-Path -LiteralPath $PidFile) {
    try {
        $pidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Stopping process PID $pidValue ($($proc.Name))..."
            Stop-Process -Id $pidValue -Force:$Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3
            Write-Host "Process stopped."
        } else {
            Write-Host "PID $pidValue not running."
        }
    } catch {
        Write-Host "PID file read/stop error: $_"
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

$pythonProcs = Get-Process -Name python -ErrorAction SilentlyContinue
if (-not $pythonProcs) {
    Write-Host "No python.exe process found."
    exit 0
}

$stopped = 0
foreach ($proc in $pythonProcs) {
    try {
        $cmd = ""
        try {
            $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" -ErrorAction Stop).CommandLine
        } catch {
            $cmd = ""
        }
        if ($cmd -like "*main.py*" -or $cmd -eq "") {
            Write-Host "Stopping python PID $($proc.Id)..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $stopped += 1
        }
    } catch {
        Write-Host "Could not stop PID $($proc.Id): $_"
    }
}

if ($stopped -eq 0) {
    Write-Host "No running Factory AI main.py process found."
} else {
    Write-Host "Stopped $stopped python process(es)."
}
