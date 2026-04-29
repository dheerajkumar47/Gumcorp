# Factory AI Tracking System

Real-time factory camera monitoring for person detection, ArUco marker identification, movement trails, and operational reporting.

## Quick Start

```powershell
.\venv\Scripts\python.exe main.py
```

Dashboard:

```text
http://localhost:8000/dashboard.html
```

## Project Structure

```text
Factory_Ai_System/
├── config/          Runtime configuration
├── data/            Employee list, camera details, floor plans, marker images, sample videos
├── docs/            Proposal and camera layout documents
├── logs/            Runtime JSON, CSV, and decoder logs
├── models/          AI model weights
├── outputs/         Live dashboard images, map artifacts, recordings, and result files
├── src/             Core runtime, detection, mapping, logging, and utility modules
├── test_records/    Uploaded screenshots/videos used for debugging
├── tools/           Maintenance and diagnostic scripts
├── dashboard.html   Browser dashboard
├── main.py          Main runtime entry point
├── requirements.txt Python dependencies
└── README.md
```

## Main Runtime

The default runtime settings are stored in `config/settings.yaml`, so the normal command is short:

```powershell
.\venv\Scripts\python.exe main.py
```

Optional override example:

```powershell
.\venv\Scripts\python.exe main.py --camera-profile main_stream --inference-width 1280 --target-fps 15 --batch-size 4 --device cuda
```

## Important Folders

`outputs/live/` contains live dashboard camera refresh images.

`outputs/recordings/` contains per-employee MP4 movement-trail recordings.

`logs/live_stats.json` is the dashboard data feed.

`logs/employee_records.json` is the latest employee summary.

## Tools

Run helper scripts from the project root:

```powershell
.\venv\Scripts\python.exe tools\check_ai.py
.\venv\Scripts\python.exe tools\check_cameras.py
.\venv\Scripts\python.exe tools\generate_aruco_markers.py
.\venv\Scripts\python.exe tools\calibrate_fov.py
```

## Current Pipeline

- Threaded RTSP camera capture
- Batched YOLO person detection on CUDA when available
- CPU-worker ArUco marker detection
- Marker-to-person association before employee tracking
- Live dashboard output
- Per-employee MP4 movement-trail recording
- CPU/RAM/GPU telemetry in `logs/live_stats.json`
