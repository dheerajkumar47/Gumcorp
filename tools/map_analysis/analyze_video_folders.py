from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIR = ROOT / "data" / "videos"
MAP_DATA = ROOT / "data" / "factory_map" / "rev03_manual_points.yaml"
OUT_DIR = ROOT / "outputs" / "video_analysis"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DATE_FOLDER_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")

CAMERA_ALIASES = {
    "cooling room 2": "cooling_room2",
    "gc production 2": "gc_production2",
    "gc production": "gc_production2",
    "main production": "main_production",
    "mixer 1": "mixer1",
    "mixer 2": "mixer2",
    "mixer 3": "mixer3",
    "mixer 4": "mixer4",
    "mixer 4": "mixer4",
    "pallet line": "palletline",
    "production outside": "production_outside",
    "sheet line": "sheetline",
}


def normalize_camera_id(path: Path) -> str:
    name = path.stem.lower().strip()
    name = re.sub(r"\s+", " ", name)
    return CAMERA_ALIASES.get(name, name.replace(" ", "_"))


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def load_camera_area_lookup() -> dict[str, dict[str, float | str]]:
    if not MAP_DATA.exists():
        return {}
    cfg = yaml.safe_load(MAP_DATA.read_text(encoding="utf-8")) or {}
    zones = {zone["id"]: zone for zone in cfg.get("zones", [])}
    camera_to_zone = {}
    for zone in cfg.get("zones", []):
        for camera_id in zone.get("cameras", []):
            camera_to_zone[camera_id] = zone["id"]

    lookup: dict[str, dict[str, float | str]] = {}
    for item in cfg.get("camera_coverage_estimates", []):
        camera_id = item["id"]
        zone_id = camera_to_zone.get(camera_id)
        zone = zones.get(zone_id or "")
        visible_px2 = polygon_area(item.get("visible_polygon", []))
        blocked_px2 = polygon_area(item.get("blocked_polygon", []))
        zone_px2 = float(zone.get("metrics", {}).get("area_px2", 0.0)) if zone else 0.0
        zone_sqft = float(zone.get("metrics", {}).get("area_sqft", 0.0)) if zone else 0.0
        sqft_per_px2 = zone_sqft / zone_px2 if zone_px2 else 0.0
        visible_sqft = visible_px2 * sqft_per_px2
        blocked_sqft = blocked_px2 * sqft_per_px2
        lookup[camera_id] = {
            "zone_id": zone_id or "",
            "visible_px2": round(visible_px2, 2),
            "blocked_px2": round(blocked_px2, 2),
            "visible_sqft": round(visible_sqft, 2),
            "blocked_sqft": round(blocked_sqft, 2),
            "clear_sqft": round(max(visible_sqft - blocked_sqft, 0.0), 2),
            "confidence": item.get("confidence", ""),
        }
    for camera_id, zone_id in camera_to_zone.items():
        if camera_id in lookup:
            continue
        zone = zones.get(zone_id)
        if not zone:
            continue
        metrics = zone.get("metrics", {})
        zone_sqft = float(metrics.get("area_sqft") or 0.0)
        lookup[camera_id] = {
            "zone_id": zone_id,
            "visible_px2": float(metrics.get("area_px2") or 0.0),
            "blocked_px2": 0.0,
            "visible_sqft": round(zone_sqft, 2),
            "blocked_sqft": 0.0,
            "clear_sqft": round(zone_sqft, 2),
            "confidence": "zone_area_fallback_not_camera_fov",
        }
    return lookup


def video_files_by_date() -> dict[str, list[Path]]:
    folders = [p for p in VIDEOS_DIR.iterdir() if p.is_dir() and DATE_FOLDER_RE.match(p.name)]
    grouped: dict[str, list[Path]] = {}
    for folder in sorted(folders):
        grouped[folder.name] = sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])
    return grouped


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def write_snapshot(path: Path, frame: np.ndarray | None) -> str:
    if frame is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return str(path.relative_to(ROOT))


def analyze_video(video_path: Path, output_folder: Path, area_lookup: dict[str, dict[str, float | str]]) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": str(video_path.relative_to(ROOT)),
            "camera_id": normalize_camera_id(video_path),
            "ok": False,
            "error": "could_not_open",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    sample_count = min(240, max(30, frame_count // max(1, int(fps * 2)) if fps else 120))
    step = max(1, frame_count // sample_count) if frame_count else 1

    bg = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=36, detectShadows=False)
    union_mask = None
    motion_values: list[float] = []
    active_frames = 0
    best_motion = -1.0
    best_frame = None
    first_frame = None
    mid_frame = None
    last_frame = None

    processed = 0
    for frame_idx in range(0, frame_count if frame_count else 1_000_000, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        if first_frame is None:
            first_frame = frame.copy()
        last_frame = frame.copy()
        if frame_count and abs(frame_idx - frame_count // 2) <= step:
            mid_frame = frame.copy()

        small_w = 360
        scale = small_w / frame.shape[1]
        small = cv2.resize(frame, (small_w, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        fg = bg.apply(small)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=1)

        motion_pct = float(cv2.countNonZero(fg)) / float(fg.size) * 100.0
        # The first few background-subtractor frames are noisy. Keep them for learning only.
        if processed >= 5:
            motion_values.append(motion_pct)
            if motion_pct >= 0.35:
                active_frames += 1
            if motion_pct > best_motion:
                best_motion = motion_pct
                best_frame = frame.copy()
            union_mask = fg.copy() if union_mask is None else cv2.bitwise_or(union_mask, fg)
        processed += 1

    cap.release()

    avg_motion = float(np.mean(motion_values)) if motion_values else 0.0
    peak_motion = float(np.max(motion_values)) if motion_values else 0.0
    active_ratio = active_frames / len(motion_values) * 100.0 if motion_values else 0.0
    union_pct = float(cv2.countNonZero(union_mask)) / float(union_mask.size) * 100.0 if union_mask is not None else 0.0

    if active_ratio >= 35 or avg_motion >= 1.2:
        activity = "working_or_high_activity"
    elif active_ratio >= 10 or avg_motion >= 0.45:
        activity = "some_activity"
    else:
        activity = "low_or_no_activity"

    camera_id = normalize_camera_id(video_path)
    out_base = output_folder / safe_name(video_path)
    snapshots = {
        "first": write_snapshot(out_base.with_name(out_base.name + "_first.jpg"), first_frame),
        "middle": write_snapshot(out_base.with_name(out_base.name + "_middle.jpg"), mid_frame),
        "peak_motion": write_snapshot(out_base.with_name(out_base.name + "_peak_motion.jpg"), best_frame),
        "last": write_snapshot(out_base.with_name(out_base.name + "_last.jpg"), last_frame),
    }

    return {
        "video": str(video_path.relative_to(ROOT)),
        "camera_id": camera_id,
        "ok": True,
        "resolution": {"width": width, "height": height},
        "fps": round(fps, 3),
        "frames": frame_count,
        "duration_sec": round(duration_sec, 2),
        "sampled_frames": len(motion_values),
        "activity": activity,
        "active_frame_ratio_pct": round(active_ratio, 2),
        "avg_motion_area_pct": round(avg_motion, 3),
        "peak_motion_area_pct": round(peak_motion, 3),
        "union_motion_area_pct": round(union_pct, 3),
        "estimated_active_sqft": round((area_lookup.get(camera_id, {}).get("clear_sqft", 0.0) or 0.0) * union_pct / 100.0, 2),
        "mapped_camera_area": area_lookup.get(camera_id, {}),
        "snapshots": snapshots,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "camera_id",
        "video",
        "duration_sec",
        "activity",
        "active_frame_ratio_pct",
        "avg_motion_area_pct",
        "peak_motion_area_pct",
        "union_motion_area_pct",
        "mapped_visible_sqft",
        "mapped_clear_sqft",
        "estimated_active_sqft",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            area = row.get("mapped_camera_area", {}) or {}
            writer.writerow(
                {
                    "date": row.get("date", ""),
                    "camera_id": row.get("camera_id", ""),
                    "video": row.get("video", ""),
                    "duration_sec": row.get("duration_sec", ""),
                    "activity": row.get("activity", ""),
                    "active_frame_ratio_pct": row.get("active_frame_ratio_pct", ""),
                    "avg_motion_area_pct": row.get("avg_motion_area_pct", ""),
                    "peak_motion_area_pct": row.get("peak_motion_area_pct", ""),
                    "union_motion_area_pct": row.get("union_motion_area_pct", ""),
                    "mapped_visible_sqft": area.get("visible_sqft", ""),
                    "mapped_clear_sqft": area.get("clear_sqft", ""),
                    "estimated_active_sqft": row.get("estimated_active_sqft", ""),
                }
            )


def write_markdown(rows: list[dict], path: Path) -> None:
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        by_date.setdefault(row.get("date", ""), []).append(row)

    lines = [
        "# Video Folder Analysis",
        "",
        "Motion is estimated from sampled frames. Camera square feet use approved map coverage where available; otherwise zone area fallback is marked in the JSON report.",
        "",
    ]
    for date in sorted(by_date):
        date_rows = by_date[date]
        active = [r for r in date_rows if r.get("activity") != "low_or_no_activity"]
        lines.append(f"## {date}")
        lines.append("")
        lines.append(f"- videos: {len(date_rows)}")
        lines.append(f"- active/some activity cameras: {len(active)}")
        if active:
            lines.append("- active cameras: " + ", ".join(r["camera_id"] for r in active))
        lines.append("")
        lines.append("| camera | activity | active frames % | union motion % | mapped clear sq ft | estimated active sq ft |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in date_rows:
            area = row.get("mapped_camera_area", {}) or {}
            lines.append(
                "| {camera} | {activity} | {active} | {union} | {clear} | {active_sqft} |".format(
                    camera=row.get("camera_id", ""),
                    activity=row.get("activity", ""),
                    active=row.get("active_frame_ratio_pct", ""),
                    union=row.get("union_motion_area_pct", ""),
                    clear=area.get("clear_sqft", ""),
                    active_sqft=row.get("estimated_active_sqft", ""),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    area_lookup = load_camera_area_lookup()
    grouped = video_files_by_date()
    all_rows: list[dict] = []
    report: dict[str, list[dict]] = {}

    for date, videos in grouped.items():
        frame_dir = OUT_DIR / date / "snapshots"
        date_rows = []
        for video in videos:
            row = analyze_video(video, frame_dir, area_lookup)
            row["date"] = date
            date_rows.append(row)
            all_rows.append(row)
            print(f"{date} {video.name}: {row.get('activity')} avg={row.get('avg_motion_area_pct')} active={row.get('active_frame_ratio_pct')}")
        report[date] = date_rows
        (OUT_DIR / date / "report.json").write_text(json.dumps(date_rows, indent=2), encoding="utf-8")
        write_csv(date_rows, OUT_DIR / date / "summary.csv")

    (OUT_DIR / "video_folder_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(all_rows, OUT_DIR / "summary.csv")
    write_markdown(all_rows, OUT_DIR / "report.md")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'video_folder_analysis.json'}")
    print(f"wrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
