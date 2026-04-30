from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "docs" / "rev03_page1.png"
YAML_PATH = ROOT / "data" / "factory_map" / "rev03_manual_points.yaml"
OUTPUT_DIR = ROOT / "outputs" / "maps"
LAYOUT_DIR = OUTPUT_DIR / "layouts"
FULL_OUTPUT = LAYOUT_DIR / "rev03_manual_layout_full.png"
DETAIL_OUTPUT = LAYOUT_DIR / "rev03_manual_layout_detail.png"
OVERLAY_OUTPUT = LAYOUT_DIR / "rev03_manual_layout_full_overlay.png"
DETAIL_OVERLAY_OUTPUT = LAYOUT_DIR / "rev03_manual_layout_detail_overlay.png"


def load_config() -> dict:
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def snap_axis_aligned(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], str]:
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx <= dy:
        x = int(round((x1 + x2) / 2))
        return (x, y1), (x, y2), "vertical"

    y = int(round((y1 + y2) / 2))
    return (x1, y), (x2, y), "horizontal"


def midpoint(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2


def polygon_area(poly: list[tuple[int, int]]) -> float:
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_centroid(poly: list[tuple[int, int]]) -> tuple[int, int]:
    sx = sum(x for x, _ in poly)
    sy = sum(y for _, y in poly)
    n = max(1, len(poly))
    return int(round(sx / n)), int(round(sy / n))


def draw_segments(canvas, pts: dict[str, tuple[int, int]], segments: list[dict], color, thickness: int, label_color=None, add_label=False):
    label_color = label_color or color
    for seg in segments:
        p1, p2, _ = snap_axis_aligned(pts[seg["from"]], pts[seg["to"]])
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        if add_label and seg.get("label"):
            mx, my = midpoint(p1, p2)
            cv2.putText(
                canvas,
                seg["label"],
                (mx + 12, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                2,
                cv2.LINE_AA,
            )


def draw_points(canvas, pts: dict[str, tuple[int, int]], ordered_names: list[str]):
    for idx, name in enumerate(ordered_names, start=1):
        x, y = pts[name]
        cv2.circle(canvas, (x, y), 11, (0, 215, 255), -1)
        cv2.circle(canvas, (x, y), 14, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            str(idx),
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )


def draw_cameras(canvas, pts: dict[str, tuple[int, int]], cameras: list[dict]):
    for cam in cameras:
        x, y = pts[cam["point"]]
        cv2.rectangle(canvas, (x - 14, y - 14), (x + 14, y + 14), (255, 0, 0), 3)
        cv2.putText(
            canvas,
            cam["id"],
            (x + 18, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )


def draw_reference_points(canvas, pts: dict[str, tuple[int, int]], reference_points: list[dict]):
    for ref in reference_points:
        x, y = pts[ref["point"]]
        cv2.putText(
            canvas,
            ref["label"],
            (x + 16, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 0, 160),
            2,
            cv2.LINE_AA,
        )


def draw_pillars(canvas, pts: dict[str, tuple[int, int]], pillars: list[dict]):
    for pillar in pillars:
        if "point" in pillar:
            x, y = pts[pillar["point"]]
        else:
            x, y = pillar["x"], pillar["y"]
        cv2.circle(canvas, (x, y), 18, (19, 69, 139), 4)
        cv2.circle(canvas, (x, y), 8, (19, 69, 139), -1)
        cv2.putText(
            canvas,
            pillar["label"],
            (x + 20, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (19, 69, 139),
            2,
            cv2.LINE_AA,
        )


def draw_zone_overlays(base_canvas, pts: dict[str, tuple[int, int]], zones: list[dict], calibration: dict | None = None):
    overlay = base_canvas.copy()
    canvas = base_canvas.copy()
    zone_styles = {
        "main_production_zone": ((255, 215, 0), (160, 120, 0)),
        "cooling_room2_zone": ((0, 200, 255), (0, 120, 180)),
        "mixers_zone": ((255, 0, 255), (160, 0, 160)),
        "gc_production2_zone": ((0, 220, 120), (0, 140, 60)),
    }
    for zone in zones:
        poly = [pts[name] for name in zone["points"]]
        poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        fill_color, stroke_color = zone_styles.get(zone["id"], ((200, 200, 0), (120, 120, 0)))
        cv2.fillPoly(overlay, [poly_np], fill_color)
        cv2.polylines(canvas, [poly_np], True, stroke_color, 5, cv2.LINE_AA)

        cx, cy = polygon_centroid(poly)
        metrics = zone.get("metrics", {})
        label_lines = [
            zone["label"],
            f"{metrics.get('area_sqft', '')} sq ft" if metrics.get("area_sqft") is not None else "",
            ", ".join(zone.get("cameras", [])),
        ]
        label_lines = [line for line in label_lines if line]
        box_w = max(190, max(len(line) for line in label_lines) * 8)
        box_h = 32 + (len(label_lines) * 18)
        cv2.rectangle(canvas, (cx - 10, cy - box_h), (cx - 10 + box_w, cy), (255, 255, 255), -1)
        cv2.rectangle(canvas, (cx - 10, cy - box_h), (cx - 10 + box_w, cy), stroke_color, 2)
        y = cy - box_h + 22
        for idx, line in enumerate(label_lines):
            font_scale = 0.6 if idx == 0 else 0.5
            cv2.putText(canvas, line, (cx, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, stroke_color, 2, cv2.LINE_AA)
            y += 20

    canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)

    return canvas


def add_header(canvas):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 76), (255, 255, 255), -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 76), (70, 70, 70), 2)
    cv2.putText(canvas, "rev03 manual layout preview v3", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(canvas, "green=door  orange=open  black=wall  blue=camera  yellow=manual point", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 2, cv2.LINE_AA)


def make_detail_crop(canvas, pts: dict[str, tuple[int, int]]):
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    pad = 170
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(canvas.shape[1], max(xs) + pad)
    y2 = min(canvas.shape[0], max(ys) + pad)
    detail = canvas[y1:y2, x1:x2].copy()
    cv2.rectangle(detail, (0, 0), (detail.shape[1] - 1, detail.shape[0] - 1), (30, 30, 30), 4)
    cv2.putText(detail, f"crop origin ({x1},{y1})", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return detail


def main():
    config = load_config()
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

    pts = {name: tuple(values) for name, values in config["points"].items()}
    canvas = image.copy()

    draw_segments(canvas, pts, config.get("walls", []), (40, 40, 40), 7)
    draw_segments(canvas, pts, config.get("doorways", []), (0, 180, 0), 7, label_color=(0, 130, 0), add_label=True)
    draw_segments(canvas, pts, config.get("open_paths", []), (0, 165, 255), 7, label_color=(0, 120, 200), add_label=True)

    p17, p25, _ = snap_axis_aligned(pts["point_17"], pts["point_25"])
    mx, my = midpoint(p17, p25)
    cv2.putText(canvas, "gap_for_sheetline_machine", (mx + 14, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 0, 160), 2, cv2.LINE_AA)

    draw_points(canvas, pts, list(config["points"].keys()))
    draw_cameras(canvas, pts, config.get("cameras", []))
    draw_reference_points(canvas, pts, config.get("reference_points", []))
    draw_pillars(canvas, pts, config.get("pillars", []))
    add_header(canvas)
    overlay_canvas = draw_zone_overlays(canvas, pts, config.get("zones", []), config.get("calibration"))

    detail = make_detail_crop(canvas, pts)
    detail_overlay = make_detail_crop(overlay_canvas, pts)

    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FULL_OUTPUT), canvas)
    cv2.imwrite(str(DETAIL_OUTPUT), detail)
    cv2.imwrite(str(OVERLAY_OUTPUT), overlay_canvas)
    cv2.imwrite(str(DETAIL_OVERLAY_OUTPUT), detail_overlay)

    print(FULL_OUTPUT)
    print(DETAIL_OUTPUT)
    print(OVERLAY_OUTPUT)
    print(DETAIL_OVERLAY_OUTPUT)


if __name__ == "__main__":
    main()
