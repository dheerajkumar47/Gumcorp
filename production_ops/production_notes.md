# Production Notes

## Location

The current machine can run production if it can reach camera RTSP URLs from the office network.

## Shifts

- First shift: `shift_A`, approximately 08:00 to 19:00/19:30
- Second shift: `shift_B`, approximately 21:00 to 06:00

At the moment, ArUco-based employee tracking is expected mainly for the first shift.

## Retention

- Recordings: keep 7 days
- Logs: keep 7 days
- Reports: keep by date folder, do not auto-delete

## Client Access

Use the machine IP address, not localhost:

```text
http://SERVER-IP:8000
```

Find IP:

```powershell
ipconfig
```

Use the IPv4 address on the camera/client network.

