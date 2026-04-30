# Map Analysis Tools

Run these scripts from the project root.

```powershell
.\venv\Scripts\python.exe tools\map_analysis\render_rev03_manual_layout.py
.\venv\Scripts\python.exe tools\map_analysis\render_camera_coverage.py sheetline
.\venv\Scripts\python.exe tools\map_analysis\render_camera_coverage.py gc_production2
.\venv\Scripts\python.exe tools\map_analysis\generate_test_heatmap.py
```

Scripts:

- `render_rev03_manual_layout.py`
  - regenerates the base rev03 layout and zone overlays in `outputs/maps/layouts/`
- `render_camera_coverage.py`
  - regenerates per-camera coverage overlays in `outputs/maps/coverage/`
- `generate_test_heatmap.py`
  - regenerates movement test heatmaps in `outputs/maps/heatmaps/`
