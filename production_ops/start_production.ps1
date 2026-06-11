param(
    [ValidateSet("auto", "shift_A", "shift_B")]
    [string]$Shift = "auto",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$CameraProfile = "main_stream",
    [ValidateSet("balanced", "speed", "quality")]
    [string]$PerformanceProfile = "speed",
    [int]$InferenceWidth = 1280,
    [double]$ArucoZoom = 8.0,
    [double]$TargetFps = 10,
    [int]$BatchSize = 8,
    [int]$RecordingRetentionDays = 7,
    [int]$LogRetentionDays = 7
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
    $shiftAEnd = (19 * 60) + 30
    $shiftBStart = (21 * 60)

    if ($minutes -ge $shiftAStart -and $minutes -le $shiftAEnd) {
        return "shift_A"
    }
    if ($minutes -ge $shiftBStart -or $minutes -lt (6 * 60)) {
        return "shift_B"
    }
    return "shift_A"
}

$runDate = Get-Date -Format "yyyy-MM-dd"
$activeShift = Get-ProductionShift -RequestedShift $Shift
$productionRoot = Join-Path $RepoRoot "outputs\production"
$recordingDir = Join-Path $productionRoot "recordings"
$reportDir = Join-Path $productionRoot "reports"
$logDir = Join-Path $productionRoot "logs"

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
Write-Host "Date: $runDate"
Write-Host "Shift: $activeShift"
Write-Host "Recordings: $recordingDir"
Write-Host "Reports: $reportDir"
Write-Host "Logs: $logDir"
Write-Host "Dashboard: http://SERVER-IP:8000"

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\production_ops\cleanup_retention.ps1" -RecordingDays $RecordingRetentionDays -LogDays $LogRetentionDays

& ".\venv\Scripts\python.exe" "main.py" `
    --camera-profile $CameraProfile `
    --performance-profile $PerformanceProfile `
    --inference-width $InferenceWidth `
    --aruco-zoom $ArucoZoom `
    --target-fps $TargetFps `
    --batch-size $BatchSize `
    --device $Device
