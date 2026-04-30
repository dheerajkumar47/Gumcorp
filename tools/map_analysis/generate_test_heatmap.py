from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "data" / "factory_map" / "rev03_manual_points.yaml"
MAPS_DIR = ROOT / "outputs" / "maps"
BASE_IMAGE = MAPS_DIR / "layouts" / "rev03_manual_layout_full_overlay.png"
OUTPUT_IMAGE = MAPS_DIR / "heatmaps" / "test_zigzag_heatmap.png"


def load_points() -> dict[str, tuple[int, int]]:
    config = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    return {name: tuple(values) for name, values in config["points"].items()}


def compute_main_zone_scales(points: dict[str, tuple[int, int]]) -> tuple[float, float]:
    # Use the approved full rev02 main-zone reference for movement conversion.
    # Horizontal: point_02 to point_15 ~= 76'-4"
    # Vertical: point_15 to point_35 ~= 104'-1"
    width_ft = 76 + 4 / 12
    height_ft = 104 + 1 / 12

    width_px = abs(points["point_15"][0] - points["point_02"][0])
    height_px = abs(points["point_15"][1] - points["point_35"][1])

    x_ft_per_px = width_ft / width_px
    y_ft_per_px = height_ft / height_px
    return x_ft_per_px, y_ft_per_px


def segment_distance_ft(p1: tuple[int, int], p2: tuple[int, int], x_ft_per_px: float, y_ft_per_px: float) -> float:
    dx_ft = abs(p2[0] - p1[0]) * x_ft_per_px
    dy_ft = abs(p2[1] - p1[1]) * y_ft_per_px
    return math.hypot(dx_ft, dy_ft)


def main() -> None:
    points = load_points()
    image = cv2.imread(str(BASE_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Missing base image: {BASE_IMAGE}")

    x_ft_per_px, y_ft_per_px = compute_main_zone_scales(points)

    # Synthetic zigzag walking test from palletline camera to sheetline camera.
    route = [
        points["point_13"],           # palletline camera
        points["point_14"],
        points["point_15"],
        (3160, 1560),
        (3320, 1440),
        (3040, 1320),
        (3311, 1029),                 # near sheet line machine
        (3160, 900),
        points["point_27"],           # sheetline camera
    ]

    total_ft = 0.0
    for i in range(len(route) - 1):
        total_ft += segment_distance_ft(route[i], route[i + 1], x_ft_per_px, y_ft_per_px)

    total_m = total_ft * 0.3048

    heat = np.zeros_like(image, dtype=np.uint8)
    for i in range(len(route) - 1):
        intensity = int(80 + (175 * (i + 1) / (len(route) - 1)))
        color = (0, intensity // 2, intensity)
        cv2.line(heat, route[i], route[i + 1], color, 34, cv2.LINE_AA)
        cv2.circle(heat, route[i], 22, color, -1, cv2.LINE_AA)
    cv2.circle(heat, route[-1], 22, (0, 140, 255), -1, cv2.LINE_AA)

    blended = cv2.addWeighted(image, 0.78, heat, 0.55, 0)

    poly = np.array(route, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(blended, [poly], False, (0, 0, 255), 5, cv2.LINE_AA)

    for idx, point in enumerate(route, start=1):
        cv2.circle(blended, point, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(blended, str(idx), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.rectangle(blended, (18, 86), (480, 188), (255, 255, 255), -1)
    cv2.rectangle(blended, (18, 86), (480, 188), (40, 40, 40), 2)
    cv2.putText(blended, "Test zigzag route: palletline -> sheetline", (32, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(blended, f"Distance: {total_ft:.1f} ft", (32, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(blended, f"Distance: {total_m:.2f} m", (32, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), blended)

    print(OUTPUT_IMAGE)
    print(f"distance_ft={total_ft:.3f}")
    print(f"distance_m={total_m:.3f}")


if __name__ == "__main__":
    main()
