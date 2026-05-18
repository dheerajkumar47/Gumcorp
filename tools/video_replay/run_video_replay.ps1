param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [double]$ReplaySpeed = 1,
    [int]$InferenceWidth = 1280,
    [double]$TargetFps = 15,
    [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

& ".\venv\Scripts\python.exe" "main.py" `
    --camera-registry "data/video_replay/cameras_replay.json" `
    --camera-profile "main_stream" `
    --inference-width $InferenceWidth `
    --target-fps $TargetFps `
    --batch-size $BatchSize `
    --device $Device `
    --replay-speed $ReplaySpeed
