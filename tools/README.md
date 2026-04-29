# Tools

Run these scripts from the project root.

```powershell
.\venv\Scripts\python.exe tools\check_ai.py
.\venv\Scripts\python.exe tools\check_cameras.py
.\venv\Scripts\python.exe tools\generate_aruco_markers.py
.\venv\Scripts\python.exe tools\calibrate_fov.py
```

## Script Purpose

- `check_ai.py`: verifies Python, PyTorch, CUDA, GPU, and YOLO inference.
- `check_cameras.py`: probes configured RTSP camera URLs.
- `generate_aruco_markers.py`: regenerates marker PNGs from `data/employees.csv`.
- `calibrate_fov.py`: generates factory camera coverage maps into `outputs/maps/`.
