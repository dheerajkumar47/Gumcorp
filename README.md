# Factory AI Tracking System

Real-time employee tracking, geo-mapping, and behavior analysis inside factories using CCTV cameras and AI models.

## System Overview

This system implements **Module 1** and **Module 2** of the Factory Intelligence Proposal:

- **Module 1 (Day 1-15)**: Factory Understanding + Mapping Foundation
- **Module 2 (Day 16-30)**: Multi-Camera Tracking + Behavior Analysis

## Project Structure (Professional Sequence)

```
Factory_Ai_System/
├── config/             # System configuration (settings.yaml)
├── data/               # Raw data (floor_plan, videos, markers)
├── docs/               # Project documentation (Proposals, PDFs)
├── logs/               # Run-time logs and CSV data
├── models/             # AI Models (YOLOv8 weights)
├── outputs/            # Annotated videos and result artifacts
├── src/                # Core logic (detection, mapping, tracking)
├── main.py             # Primary entry point for the pipeline
├── calibrate_fov.py    # Camera FOV and coverage analysis script
├── dashboard.html      # Visualization dashboard for monthly insights
├── requirements.txt    # Project dependencies
└── README.md           # This documentation
```

## Setup & Usage

### 1. Environment Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configuration
Edit `config/settings.yaml` to configure cameras, detection thresholds, and zone coordinates.

### 3. Run Analysis
To process a video feed:
```bash
python main.py
```

To run FOV coverage analysis:
```bash
python calibrate_fov.py
```

## Module 1 & 2 Progress
- [x] Professional project reorganization
- [x] Person detection (YOLOv8)
- [x] ArUco marker identification
- [x] Homography-based geo-mapping (Ready for calibration)
- [x] Multi-camera config support
- [ ] Live Dashboard integration (Ongoing)

## Output Data
The system generates CSV logs in `logs/` and annotated videos in `outputs/`. View `dashboard.html` for a summary of insights.
