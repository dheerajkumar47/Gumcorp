#Requires -RunAsAdministrator
<#
.SYNOPSIS
    One-time Factory AI System setup for production deployment.
    Run this ONCE on the factory AI PC as Administrator.

.EXAMPLE
    .\production_ops\setup_factory.ps1 -AiPcIp 172.16.0.206 -StorageRoot "D:\Factory_AI" -ShiftAStart "08:00" -ShiftAStop "20:00" -ShiftBStart "21:00" -ShiftBStop "05:00"
#>
param(
    [string]$AiPcIp = "172.16.0.206",
    [string]$StorageRoot = "D:\Factory_AI",
    [string]$ShareName = "Factory_AI",
    [int]$DashboardPort = 8000,
    [string]$WorkDays = "Monday,Tuesday,Wednesday,Thursday,Friday,Saturday",
    [string]$ShiftAStart = "08:00",
    [string]$ShiftAStop = "20:00",
    [string]$ShiftBStart = "21:00",
    [string]$ShiftBStop = "05:00",
    [bool]$ShiftBEnabled = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DaysList = $WorkDays -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }

Write-Host ""
Write-Host "======================================================"
Write-Host "  Factory AI System - Production Setup"
Write-Host "======================================================"
Write-Host "  AI PC IP:      $AiPcIp"
Write-Host "  Storage:       $StorageRoot"
Write-Host "  Dashboard:     http://${AiPcIp}:${DashboardPort}"
Write-Host "  Work days:     $($DaysList -join ', ')"
Write-Host "  Shift A:       $ShiftAStart -> $ShiftAStop"
if ($ShiftBEnabled) {
    Write-Host "  Shift B:       $ShiftBStart -> $ShiftBStop (next day)"
} else {
    Write-Host "  Shift B:       DISABLED"
}
Write-Host "======================================================"
Write-Host ""

Write-Host "[1/6] Creating storage folders at $StorageRoot ..."
foreach ($folder in @("recordings", "reports", "logs", "live")) {
    New-Item -ItemType Directory -Path (Join-Path $StorageRoot $folder) -Force | Out-Null
}
Write-Host "      OK"

Write-Host "[2/6] Adding Windows Firewall rule for port $DashboardPort ..."
$ruleName = "Factory_AI_Dashboard_$DashboardPort"
if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) {
    Write-Host "      Already exists."
} else {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort $DashboardPort `
        -Action Allow `
        -Profile Domain,Private `
        -Description "Factory AI web dashboard - LAN access" | Out-Null
    Write-Host "      Added: TCP port $DashboardPort inbound."
}

Write-Host "[3/6] Creating LAN share \\$env:COMPUTERNAME\$ShareName ..."
if (Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue) {
    Write-Host "      Already exists."
} else {
    New-SmbShare `
        -Name $ShareName `
        -Path $StorageRoot `
        -Description "Factory AI data - recordings, reports, logs" `
        -ReadAccess "Everyone" | Out-Null
    Write-Host "      Created: \\$env:COMPUTERNAME\$ShareName"
}
Write-Host "      LAN path: \\${AiPcIp}\$ShareName"

Write-Host "[4/6] Installing Task Scheduler shift tasks ..."
$StartScript = Join-Path $RepoRoot "production_ops\start_production.ps1"
$StopScript = Join-Path $RepoRoot "production_ops\stop_production.ps1"

$DayMap = @{
    "Monday" = [System.DayOfWeek]::Monday
    "Tuesday" = [System.DayOfWeek]::Tuesday
    "Wednesday" = [System.DayOfWeek]::Wednesday
    "Thursday" = [System.DayOfWeek]::Thursday
    "Friday" = [System.DayOfWeek]::Friday
    "Saturday" = [System.DayOfWeek]::Saturday
    "Sunday" = [System.DayOfWeek]::Sunday
}

$ScheduleDays = @()
foreach ($dayName in $DaysList) {
    if (-not $DayMap.ContainsKey($dayName)) {
        throw "Invalid work day '$dayName'. Use Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday."
    }
    $ScheduleDays += $DayMap[$dayName]
}

function Install-ShiftTask {
    param(
        [string]$Name,
        [string]$Script,
        [string]$ExtraArgs,
        [string]$TimeStr,
        [object[]]$Days
    )

    $hhmm = $TimeStr -split ":"
    $atTime = Get-Date -Hour ([int]$hhmm[0]) -Minute ([int]$hhmm[1]) -Second 0
    $triggers = @()
    foreach ($day in $Days) {
        $triggers += New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $atTime
    }

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`" $ExtraArgs" `
        -WorkingDirectory $RepoRoot

    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType S4U `
        -RunLevel Highest

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 14) `
        -RestartCount 2 `
        -RestartInterval (New-TimeSpan -Minutes 5)

    Register-ScheduledTask `
        -TaskName $Name `
        -Action $action `
        -Trigger $triggers `
        -Principal $principal `
        -Settings $settings `
        -Force | Out-Null

    Write-Host "      $Name @ $TimeStr"
}

$startAArgs = "-Shift shift_A -Device cuda -StorageRoot `"$StorageRoot`" -RecordingRetentionDays 7 -LogRetentionDays 7"
Install-ShiftTask "Factory_AI_ShiftA_Start" $StartScript $startAArgs $ShiftAStart $ScheduleDays
Install-ShiftTask "Factory_AI_ShiftA_Stop" $StopScript "" $ShiftAStop $ScheduleDays

if ($ShiftBEnabled) {
    $startBArgs = "-Shift shift_B -Device cuda -StorageRoot `"$StorageRoot`" -RecordingRetentionDays 7 -LogRetentionDays 7"
    Install-ShiftTask "Factory_AI_ShiftB_Start" $StartScript $startBArgs $ShiftBStart $ScheduleDays
    Install-ShiftTask "Factory_AI_ShiftB_Stop" $StopScript "" $ShiftBStop $ScheduleDays
}

Write-Host "      Tasks visible in: Task Scheduler -> Task Scheduler Library"

Write-Host "[5/6] Saving network info file ..."
$shiftBLine = if ($ShiftBEnabled) { "  Shift B: Starts $ShiftBStart -> stops $ShiftBStop next morning" } else { "  Shift B: DISABLED" }
$shiftBTaskLine = if ($ShiftBEnabled) { "  Factory_AI_ShiftB_Start - runs at $ShiftBStart on work days`n  Factory_AI_ShiftB_Stop  - runs at $ShiftBStop on work days" } else { "" }

$infoText = @"
Factory AI System - Network and Setup Info
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm")

NETWORK
AI PC Name:     $env:COMPUTERNAME
AI PC IP:       $AiPcIp
Dashboard URL:  http://${AiPcIp}:${DashboardPort}
LAN Share:      \\${AiPcIp}\${ShareName}

STORAGE
Root:           $StorageRoot
Recordings:     $StorageRoot\recordings\YYYY-MM-DD\shift_A or shift_B
Reports:        $StorageRoot\reports\YYYY-MM-DD\shift_A or shift_B
Logs:           $StorageRoot\logs\YYYY-MM-DD\shift_A or shift_B
Retention:      recordings/logs/reports date folders older than 7 days are deleted on start

SCHEDULE
Work days:      $($DaysList -join ', ')
  Shift A: Starts $ShiftAStart -> stops $ShiftAStop
$shiftBLine

Auto-start tasks:
  Factory_AI_ShiftA_Start - runs at $ShiftAStart on work days
  Factory_AI_ShiftA_Stop  - runs at $ShiftAStop on work days
$shiftBTaskLine

MANUAL COMMANDS
Start now: .\production_ops\start_production.ps1 -Shift shift_A -StorageRoot "$StorageRoot"
Stop now:  .\production_ops\stop_production.ps1

REMOTE ACCESS
Dashboard from office LAN/VPN: http://${AiPcIp}:${DashboardPort}
Files from office LAN/VPN:     \\${AiPcIp}\${ShareName}
"@

$infoPath = Join-Path $StorageRoot "FACTORY_AI_INFO.txt"
$infoText | Out-File $infoPath -Encoding utf8
Write-Host "      Saved to $infoPath"

Write-Host "[6/6] SETUP COMPLETE"
Write-Host ""
Write-Host "Dashboard: http://${AiPcIp}:${DashboardPort}"
Write-Host "Share:     \\${AiPcIp}\${ShareName}"
Write-Host "Schedule:  Shift A $ShiftAStart-$ShiftAStop, Shift B $ShiftBStart-$ShiftBStop"
Write-Host "Retention: 7 days"
