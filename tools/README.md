# Tools

Run these scripts from the project root.

```powershell
.\venv\Scripts\python.exe tools\check_ai.py
.\venv\Scripts\python.exe tools\check_cameras.py
.\venv\Scripts\python.exe tools\generate_aruco_markers.py
```

## Script Purpose

- `check_ai.py`: verifies Python, PyTorch, CUDA, GPU, and YOLO inference.
- `check_cameras.py`: probes configured RTSP camera URLs.
- `generate_aruco_markers.py`: regenerates marker PNGs from `data/employees.csv`.

## Map Analysis

Map-analysis scripts are grouped in `tools/map_analysis/`.

```powershell
.\venv\Scripts\python.exe tools\map_analysis\render_rev03_manual_layout.py
.\venv\Scripts\python.exe tools\map_analysis\render_camera_coverage.py palletline
.\venv\Scripts\python.exe tools\map_analysis\generate_test_heatmap.py
```
