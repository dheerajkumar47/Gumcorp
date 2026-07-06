param(
    [ValidateSet("auto", "shift_A", "shift_B")]
    [string]$Shift = "auto",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$CameraProfile = "main_stream",
    [ValidateSet("balanced", "speed", "quality")]
    [string]$PerformanceProfile = "balanced",
    [int]$InferenceWidth = 1280,
    [double]$ArucoZoom = 4.0,
    [double]$TargetFps = 12,
    [int]$BatchSize = 8,
    [int]$RecordingRetentionDays = 7,
    [int]$LogRetentionDays = 7,
    # Production storage root — change to D:\Factory_AI on the factory PC
    [string]$StorageRoot = "D:\Factory_AI"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Get-ProductionShift {
    param([string]$RequestedShift)
    if ($RequestedShift -ne "auto") {
        return $RequestedShift
    }

    $now = Get-Date
    $minutes = ($now.Hour * 60) + $now.Minute
    $shiftAStart = (8 * 60)
    $shiftAEnd = (20 * 60)
    $shiftBStart = (21 * 60)
    $shiftBEnd = (5 * 60)

    if ($minutes -ge $shiftAStart -and $minutes -lt $shiftAEnd) {
        return "shift_A"
    }
    if ($minutes -ge $shiftBStart -or $minutes -lt $shiftBEnd) {
        return "shift_B"
    }
    return "shift_A"
}

$runDate = Get-Date -Format "yyyy-MM-dd"
$activeShift = Get-ProductionShift -RequestedShift $Shift

# Use D:\Factory_AI if it exists, otherwise fall back to project folder
if (-not (Test-Path $StorageRoot)) {
    try { New-Item -ItemType Directory -Path $StorageRoot -Force | Out-Null }
    catch { $StorageRoot = Join-Path $RepoRoot "outputs\production" }
}
$recordingDir = Join-Path (Join-Path (Join-Path $StorageRoot "recordings") $runDate) $activeShift
$reportDir    = Join-Path (Join-Path (Join-Path $StorageRoot "reports")    $runDate) $activeShift
$logDir       = Join-Path (Join-Path (Join-Path $StorageRoot "logs")       $runDate) $activeShift

New-Item -ItemType Directory -Path $recordingDir -Force | Out-Null
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\live" -Force | Out-Null

$env:FACTORY_AI_PRODUCTION = "1"
$env:FACTORY_AI_SHIFT = $activeShift
$env:FACTORY_AI_RUN_DATE = $runDate
$env:FACTORY_AI_RECORDING_DIR = $recordingDir
$env:FACTORY_AI_REPORT_DIR = $reportDir
$env:FACTORY_AI_LOG_DIR = $logDir

Write-Host "Factory AI production start"
Write-Host "Date:       $runDate"
Write-Host "Shift:      $activeShift"
Write-Host "Storage:    $StorageRoot"
Write-Host "Recordings: $recordingDir"
Write-Host "Reports:    $reportDir"
Write-Host "Logs:       $logDir"
$LanIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike '*Loopback*' } | Select-Object -First 1).IPAddress
Write-Host "Dashboard:  http://localhost:8000  (LAN: http://${LanIp}:8000)"

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\production_ops\cleanup_retention.ps1" -StorageRoot $StorageRoot -RecordingDays $RecordingRetentionDays -LogDays $LogRetentionDays

# Run python — same as original (inline, Ctrl+C works normally)
& ".\venv\Scripts\python.exe" "main.py" `
    --camera-profile $CameraProfile `
    --performance-profile $PerformanceProfile `
    --inference-width $InferenceWidth `
    --aruco-zoom $ArucoZoom `
    --target-fps $TargetFps `
    --batch-size $BatchSize `
    --device $Device

$pythonExitCode = $LASTEXITCODE
if ($pythonExitCode -ne 0) {
    Write-Host "Factory AI exited with code $pythonExitCode"
    exit $pythonExitCode
}
