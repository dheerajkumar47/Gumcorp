from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "data" / "factory_map" / "rev03_manual_points.yaml"
MAPS_DIR = ROOT / "outputs" / "maps"
BASE_IMAGE = MAPS_DIR / "layouts" / "rev03_manual_layout_full_overlay.png"
OUT_DIR = MAPS_DIR / "coverage"


ZONE_BY_CAMERA = {
    "sheetline": "main_production_zone",
    "palletline": "main_production_zone",
    "main_production": "main_production_zone",
    "gc_production2": "gc_production2_zone",
}

DETAIL_CROPS = {
    "sheetline": (560, 1760, 2860, 3685),
    "palletline": (820, 1770, 2820, 3415),
    "main_production": (820, 1770, 2580, 3415),
    "gc_production2": (820, 1765, 2240, 3065),
}


def polygon_area(poly: list[tuple[int, int]]) -> float:
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: render_camera_coverage.py <camera_id>")

    camera_id = sys.argv[1]
    cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    image = cv2.imread(str(BASE_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Missing base image: {BASE_IMAGE}")

    coverage = next(item for item in cfg["camera_coverage_estimates"] if item["id"] == camera_id)
    zone = next(item for item in cfg["zones"] if item["id"] == ZONE_BY_CAMERA[camera_id])

    visible = [tuple(p) for p in coverage["visible_polygon"]]
    blocked = [tuple(p) for p in coverage.get("blocked_polygon", [])]

    zone_sqft = float(zone["metrics"]["area_sqft"])
    zone_px2 = float(zone["metrics"]["area_px2"])
    sqft_per_px2 = zone_sqft / zone_px2

    visible_px2 = polygon_area(visible)
    blocked_px2 = polygon_area(blocked) if blocked else 0.0
    visible_sqft = visible_px2 * sqft_per_px2
    blocked_sqft = blocked_px2 * sqft_per_px2
    clear_sqft = max(visible_sqft - blocked_sqft, 0.0)

    overlay = image.copy()
    canvas = image.copy()

    visible_np = np.array(visible, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [visible_np], (0, 220, 120))
    if blocked:
        blocked_np = np.array(blocked, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [blocked_np], (0, 60, 220))
        cv2.polylines(canvas, [blocked_np], True, (0, 0, 180), 5, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
    cv2.polylines(canvas, [visible_np], True, (0, 130, 70), 5, cv2.LINE_AA)

    panel_h = 256 if blocked else 226
    cv2.rectangle(canvas, (18, 86), (760, panel_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (18, 86), (760, panel_h), (40, 40, 40), 2)
    title = f"{camera_id.replace('_', ' ')} camera coverage estimate"
    cv2.putText(canvas, title, (32, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Seen room footprint: {visible_sqft:.1f} sq ft", (32, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 110, 50), 2, cv2.LINE_AA)
    if blocked:
        cv2.putText(canvas, f"Blocked by storage/equipment: {blocked_sqft:.1f} sq ft", (32, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 170), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Net clear floor: {clear_sqft:.1f} sq ft", (32, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Green = camera footprint | Red = blocked subset", (32, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, f"Net clear floor: {clear_sqft:.1f} sq ft", (32, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Green = clear visible zone", (32, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_image = OUT_DIR / f"{camera_id}_camera_coverage_overlay.png"
    detail_image = OUT_DIR / f"{camera_id}_camera_coverage_detail.png"
    cv2.imwrite(str(output_image), canvas)

    y1, y2, x1, x2 = DETAIL_CROPS[camera_id]
    crop = canvas[y1:y2, x1:x2].copy()
    cv2.imwrite(str(detail_image), crop)

    print(output_image)
    print(detail_image)
    print(f"visible_sqft={visible_sqft:.3f}")
    print(f"blocked_sqft={blocked_sqft:.3f}")
    print(f"clear_sqft={clear_sqft:.3f}")


if __name__ == "__main__":
    main()
