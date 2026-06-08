param(
    [int]$RecordingDays = 7,
    [int]$LogDays = 7
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$productionRoot = Join-Path $RepoRoot "outputs\production"
$recordingsRoot = Join-Path $productionRoot "recordings"
$logsRoot = Join-Path $productionRoot "logs"

function Remove-OldDateFolders {
    param(
        [string]$Root,
        [int]$KeepDays
    )
    if (-not (Test-Path -LiteralPath $Root)) {
        return
    }

    $cutoff = (Get-Date).Date.AddDays(-1 * [Math]::Max(1, $KeepDays))
    Get-ChildItem -LiteralPath $Root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $folderDate = [DateTime]::ParseExact($_.Name, "yyyy-MM-dd", [Globalization.CultureInfo]::InvariantCulture)
            if ($folderDate -lt $cutoff) {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force
                Write-Host "Removed old folder $($_.FullName)"
            }
        } catch {
            return
        }
    }
}

Remove-OldDateFolders -Root $recordingsRoot -KeepDays $RecordingDays
Remove-OldDateFolders -Root $logsRoot -KeepDays $LogDays
