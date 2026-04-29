# Factory Map Seed

This folder stores manually verified spatial seed data for the Gumcorp production area.

Purpose:

- Keep the client's floor-plan interpretation in one clean place.
- Separate spatial truth from runtime code.
- Provide a stable input for later homography calibration, zone mapping, and distance conversion.

Current source set:

- `docs/camera_layout_rev02.pdf`
- `docs/marking.jpg`
- user verified notes from April 29, 2026

Current file:

- `production_layout_seed.yaml`

This seed is not a trained ML model.
It is the spatial ground-truth package that the tracking and mapping pipeline should use next.

Next required step for accurate distance:

- For each active camera, record at least 4 homography anchor pairs:
  - `camera_points`: pixel points from the camera image
  - `map_points`: matching points from the floor-plan

Without those anchor pairs, the system can only use approximate plan geometry, not true camera-to-floor mapping.
