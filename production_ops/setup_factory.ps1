#Requires -RunAsAdministrator
<#
.SYNOPSIS
    One-time Factory AI System setup for production deployment.
    Run this ONCE on the factory AI PC as Administrator.

.PARAMETER AiPcIp
    Static IP of this AI PC on the factory LAN. Example: 192.168.3.50

.PARAMETER StorageRoot
    Where to store recordings/reports/logs. Default: D:\Factory_AI

.PARAMETER ShareName
    LAN share name other PCs use to access data. Default: Factory_AI

.PARAMETER DashboardPort
    Web dashboard port. Default: 8000

.PARAMETER WorkDays
    Which days to run. Default: Monday,Tuesday,Wednesday,Thursday,Friday,Saturday
    (Mon–Sat working week. Remove Saturday if not needed.)

.PARAMETER ShiftAStart / ShiftAStop
    Shift A window.  Default: 08:00 → 19:30

.PARAMETER ShiftBStart / ShiftBStop
    Shift B window.  Default: 21:00 → 06:00 (next morning)

.PARAMETER ShiftBEnabled
    Set to $false to disable night shift tasks entirely.

.EXAMPLE
    # Mon–Sat, both shifts
    .\production_ops\setup_factory.ps1 -AiPcIp 192.168.3.50

    # Mon–Fri only, no night shift
    .\production_ops\setup_factory.ps1 -AiPcIp 192.168.3.50 `
        -WorkDays "Monday,Tuesday,Wednesday,Thursday,Friday" -ShiftBEnabled $false
#>
param(
    [string]  $AiPcIp        = "192.168.3.50",
    [string]  $StorageRoot   = "D:\Factory_AI",
    [string]  $ShareName     = "Factory_AI",
    [int]     $DashboardPort = 8000,
    [string]  $WorkDays      = "Monday,Tuesday,Wednesday,Thursday,Friday,Saturday",
    [string]  $ShiftAStart   = "08:00",
    [string]  $ShiftAStop    = "19:30",
    [string]  $ShiftBStart   = "21:00",
    [string]  $ShiftBStop    = "06:00",
    [bool]    $ShiftBEnabled = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot  = Resolve-Path (Join-Path $PSScriptRoot "..")
$DaysList  = $WorkDays -split "," | ForEach-Object { $_.Trim() }

Write-Host ""
Write-Host "======================================================"
Write-Host "  Factory AI System — Production Setup"
Write-Host "======================================================"
Write-Host "  AI PC IP:      $AiPcIp"
Write-Host "  Storage:       $StorageRoot"
Write-Host "  Dashboard:     http://${AiPcIp}:${DashboardPort}"
Write-Host "  Work days:     $($DaysList -join ', ')"
Write-Host "  Shift A:       $ShiftAStart  →  $ShiftAStop"
if ($ShiftBEnabled) {
Write-Host "  Shift B:       $ShiftBStart  →  $ShiftBStop (next day)"
} else {
Write-Host "  Shift B:       DISABLED"
}
Write-Host "======================================================"
Write-Host ""

# ── STEP 1: Create storage folders ────────────────────────────────────────────
Write-Host "[1/6] Creating storage folders at $StorageRoot ..."
foreach ($f in @("recordings","reports","logs","live")) {
    New-Item -ItemType Directory -Path "$StorageRoot\$f" -Force | Out-Null
}
Write-Host "      OK"

# ── STEP 2: Windows Firewall — allow dashboard port ───────────────────────────
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
        -Description "Factory AI web dashboard — LAN access" | Out-Null
    Write-Host "      Added: TCP port $DashboardPort inbound (Domain+Private)."
}

# ── STEP 3: LAN network share ─────────────────────────────────────────────────
Write-Host "[3/6] Creating LAN share \\$env:COMPUTERNAME\$ShareName ..."
if (Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue) {
    Write-Host "      Already exists."
} else {
    New-SmbShare -Name $ShareName -Path $StorageRoot `
        -Description "Factory AI data — recordings, reports, logs" `
        -ReadAccess "Everyone" | Out-Null
    Write-Host "      Created: \\$env:COMPUTERNAME\$ShareName"
    Write-Host "      Also:    \\${AiPcIp}\$ShareName"
}

# ── STEP 4: Task Scheduler — working days only ────────────────────────────────
Write-Host "[4/6] Installing Task Scheduler shift tasks ($($DaysList -join ',')) ..."

$StartScript = Join-Path $RepoRoot "production_ops\start_production.ps1"
$StopScript  = Join-Path $RepoRoot "production_ops\stop_production.ps1"

# Map day names to DayOfWeek enum values
$DayMap = @{
    "Monday"    = [System.DayOfWeek]::Monday
    "Tuesday"   = [System.DayOfWeek]::Tuesday
    "Wednesday" = [System.DayOfWeek]::Wednesday
    "Thursday"  = [System.DayOfWeek]::Thursday
    "Friday"    = [System.DayOfWeek]::Friday
    "Saturday"  = [System.DayOfWeek]::Saturday
    "Sunday"    = [System.DayOfWeek]::Sunday
}
$ScheduleDays = $DaysList | ForEach-Object { $DayMap[$_] }

function Install-ShiftTask {
    param(
        [string]   $Name,
        [string]   $Script,
        [string]   $ExtraArgs,
        [string]   $TimeStr,
        [object[]] $Days
    )
    $hhmm   = $TimeStr -split ":"
    $atTime = Get-Date -Hour ([int]$hhmm[0]) -Minute ([int]$hhmm[1]) -Second 0

    # Weekly trigger for each work day
    $triggers = @()
    foreach ($day in $Days) {
        $t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $atTime
        $triggers += $t
    }

    $action    = New-ScheduledTaskAction `
                    -Execute "powershell.exe" `
                    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`" $ExtraArgs" `
                    -WorkingDirectory $RepoRoot

    $principal = New-ScheduledTaskPrincipal `
                    -UserId $env:USERNAME -LogonType S4U -RunLevel Highest

    $settings  = New-ScheduledTaskSettingsSet `
                    -AllowStartIfOnBatteries `
                    -DontStopIfGoingOnBatteries `
                    -ExecutionTimeLimit (New-TimeSpan -Hours 14) `
                    -RestartCount 2 `
                    -RestartInterval (New-TimeSpan -Minutes 5)

    Register-ScheduledTask -TaskName $Name -Action $action `
        -Trigger $triggers -Principal $principal -Settings $settings -Force | Out-Null

    Write-Host "      $Name  @ $TimeStr  ($($Days -join ', '))"
}

$startArgs = "-Shift shift_A -Device cuda -StorageRoot `"$StorageRoot`""
Install-ShiftTask "Factory_AI_ShiftA_Start" $StartScript $startArgs        $ShiftAStart $ScheduleDays
Install-ShiftTask "Factory_AI_ShiftA_Stop"  $StopScript  ""                $ShiftAStop  $ScheduleDays

if ($ShiftBEnabled) {
    $startBArgs = "-Shift shift_B -Device cuda -StorageRoot `"$StorageRoot`""
    Install-ShiftTask "Factory_AI_ShiftB_Start" $StartScript $startBArgs   $ShiftBStart $ScheduleDays
    Install-ShiftTask "Factory_AI_ShiftB_Stop"  $StopScript  ""            $ShiftBStop  $ScheduleDays
}

Write-Host "      Tasks visible in: Task Scheduler → Task Scheduler Library"

# ── STEP 5: Save network info file ────────────────────────────────────────────
Write-Host "[5/6] Saving network info file ..."
$shiftBLine = if ($ShiftBEnabled) { "  Shift B:  Starts $ShiftBStart  → Stops $ShiftBStop (next morning)" } else { "  Shift B:  DISABLED" }
@"
Factory AI System — Network & Setup Info
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm")

== NETWORK ==
AI PC Name:     $env:COMPUTERNAME
AI PC IP:       $AiPcIp  (set this as static IP in Windows Network Settings)
Dashboard URL:  http://${AiPcIp}:${DashboardPort}
LAN Share:      \\${AiPcIp}\${ShareName}   or   \\$env:COMPUTERNAME\$ShareName

== STORAGE ==
Root:           $StorageRoot
Recordings:     $StorageRoot\recordings\YYYY-MM-DD\shift_A (or shift_B)
Reports:        $StorageRoot\reports\YYYY-MM-DD\shift_A
Logs:           $StorageRoot\logs\YYYY-MM-DD\shift_A

== SCHEDULE ==
Work days:      $($DaysList -join ', ')
  Shift A:  Starts $ShiftAStart  → Stops $ShiftAStop
$shiftBLine

Auto-start tasks (Task Scheduler):
  Factory_AI_ShiftA_Start  — runs at $ShiftAStart on work days
  Factory_AI_ShiftA_Stop   — runs at $ShiftAStop  on work days
$(if ($ShiftBEnabled) { "  Factory_AI_ShiftB_Start  — runs at $ShiftBStart on work days`n  Factory_AI_ShiftB_Stop   — runs at $ShiftBStop  on work days" })

== MANUAL COMMANDS ==
Start now:   .\production_ops\start_production.ps1 -Shift shift_A
Stop now:    .\production_ops\stop_production.ps1

== REMOTE ACCESS ==
VPN:         Connect company VPN → open http://${AiPcIp}:${DashboardPort}
Remote PC:   AnyDesk / RustDesk / Windows Remote Desktop (via VPN)
"@ | Out-File "$StorageRoot\FACTORY_AI_INFO.txt" -Encoding utf8
Write-Host "      Saved to $StorageRoot\FACTORY_AI_INFO.txt"

# ── STEP 6: Final summary ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] SETUP COMPLETE"
Write-Host ""
Write-Host "  Remaining manual steps:"
Write-Host "  1. Set this PC IP to $AiPcIp (static) in:"
Write-Host "     Control Panel → Network → Ethernet → IPv4 Properties"
Write-Host "  2. Update camera RTSP URLs in: config/settings.yaml"
Write-Host "  3. Test cameras: python tools/check_cameras.py"
Write-Host "  4. Open dashboard to confirm: http://${AiPcIp}:${DashboardPort}"
Write-Host "  5. From office/admin PC: http://${AiPcIp}:${DashboardPort}"
Write-Host "  6. Network data share: \\${AiPcIp}\$ShareName"
Write-Host ""
Write-Host "  System will auto-start/stop daily. No manual terminal needed."
Write-Host ""
