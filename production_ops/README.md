# Factory AI Production Ops

This folder keeps production deployment scripts separate from the main tracking code.

## Production Shifts

- `shift_A`: first shift, around 08:00 to 19:30
- `shift_B`: second shift, around 21:00 to 06:00

Current ArUco tracking is expected mainly for `shift_A`.

## Start Production

From the repo root:

```powershell
.\production_ops\start_production.ps1 -Shift auto
```

Dashboard:

```text
http://SERVER-IP:8000
```

Example:

```text
http://192.168.1.50:8000
```

## Output Layout

Production scripts keep dated folders under:

```text
outputs/production/
  recordings/YYYY-MM-DD/
  reports/YYYY-MM-DD/
  logs/YYYY-MM-DD/
```

`outputs/live/` stays as the current live dashboard image folder.

## Retention

Recordings and production logs are kept for 7 days by default.
Reports are not deleted by the cleanup script.

Manual cleanup:

```powershell
.\production_ops\cleanup_retention.ps1 -RecordingDays 7 -LogDays 7
```

## Auto Start After Restart

Run PowerShell as Administrator:

```powershell
.\production_ops\install_startup_task.ps1
```

This creates a Windows Scheduled Task named `FactoryAiProduction`.

