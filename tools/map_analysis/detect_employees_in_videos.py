from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIR = ROOT / "data" / "videos"
SETTINGS = ROOT / "config" / "settings.yaml"
OUT_DIR = ROOT / "outputs" / "video_analysis" / "employee_detection"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DATE_FOLDER = "02-05-26"

CAMERA_ALIASES = {
    "cooling room 2": "cooling_room2",
    "gc production 2": "gc_production2",
    "main production": "main_production",
    "mixer 1": "mixer1",
    "mixer 2": "mixer2",
    "mixer 3": "mixer3",
    "mixer 4": "mixer4",
    "pallet line": "palletline",
    "production outside": "production_outside",
    "sheet line": "sheetline",
}


def normalize_camera_id(path: Path) -> str:
    name = re.sub(r"\s+", " ", path.stem.lower().strip())
    return CAMERA_ALIASES.get(name, name.replace(" ", "_"))


def resize_to_width(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    resized = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_aruco(frame: np.ndarray, dictionary_type: str, max_width: int) -> list[dict]:
    small, scale = resize_to_width(frame, max_width)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_type))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    detections: list[dict] = []
    if ids is None:
        return detections
    for corner, marker_id in zip(corners, ids.flatten()):
        pts = corner[0] / scale
        area = float(cv2.contourArea(pts.astype(np.float32)))
        center = pts.mean(axis=0)
        detections.append(
            {
                "id": int(marker_id),
                "area_px2": round(area, 2),
                "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
                "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in pts],
            }
        )
    return detections


def marker_inside_person(marker: dict, bbox: list[float], margin: float = 0.18) -> bool:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    cx, cy = marker["center"]
    return x1 <= cx <= x2 and y1 <= cy <= y2


def annotate(frame: np.ndarray, persons: list[dict], markers: list[dict], matches: list[dict]) -> np.ndarray:
    out = frame.copy()
    for idx, person in enumerate(persons, start=1):
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"person {idx} {person['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
    for marker in markers:
        pts = np.array(marker["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
        cx, cy = [int(v) for v in marker["center"]]
        cv2.putText(out, f"aruco {marker['id']}", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    for match in matches:
        cx, cy = [int(v) for v in match["marker_center"]]
        cv2.circle(out, (cx, cy), 18, (0, 255, 255), 3, cv2.LINE_AA)
    return out


def analyze_video(video_path: Path, model: YOLO, cfg: dict, device: str) -> dict:
    camera_id = normalize_camera_id(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"camera_id": camera_id, "video": str(video_path.relative_to(ROOT)), "ok": False, "error": "could_not_open"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps)))
    max_samples = 75

    person_frames = 0
    marker_frames = 0
    matched_frames = 0
    total_persons = 0
    marker_ids: dict[int, int] = {}
    best_frame = None
    best_score = -1
    best_payload = None
    samples = 0

    for frame_idx in range(0, frame_count, step):
        if samples >= max_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        samples += 1

        inference_width = int(cfg["runtime"].get("inference_width", 1280))
        small, scale = resize_to_width(frame, inference_width)
        results = model.predict(
            small,
            conf=float(cfg["detection"].get("confidence_threshold", 0.5)),
            iou=float(cfg["detection"].get("iou_threshold", 0.45)),
            classes=[0],
            device=device,
            verbose=False,
        )
        persons = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (box.xyxy[0].detach().cpu().numpy() / scale).tolist()
            conf = float(box.conf[0].detach().cpu().item())
            persons.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

        markers = detect_aruco(
            frame,
            cfg["aruco"].get("dictionary_type", "DICT_4X4_50"),
            int(cfg["runtime"].get("aruco_max_width", 1920)),
        )
        filtered_markers = [m for m in markers if m["area_px2"] >= float(cfg["runtime"].get("min_marker_area", 80))]

        matches = []
        for marker in filtered_markers:
            marker_ids[marker["id"]] = marker_ids.get(marker["id"], 0) + 1
            for person_idx, person in enumerate(persons):
                if marker_inside_person(marker, person["bbox"]):
                    matches.append(
                        {
                            "marker_id": marker["id"],
                            "person_index": person_idx,
                            "marker_center": marker["center"],
                        }
                    )
                    break

        if persons:
            person_frames += 1
            total_persons += len(persons)
        if filtered_markers:
            marker_frames += 1
        if matches:
            matched_frames += 1

        score = len(matches) * 100 + len(filtered_markers) * 20 + len(persons)
        if score > best_score:
            best_score = score
            best_payload = {
                "frame_index": frame_idx,
                "persons": persons,
                "markers": filtered_markers,
                "matches": matches,
            }
            best_frame = annotate(frame, persons, filtered_markers, matches)

    cap.release()

    snapshot_path = ""
    if best_frame is not None:
        snap_dir = OUT_DIR / DATE_FOLDER / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snap_dir / f"{camera_id}_best_employee_marker_frame.jpg"
        cv2.imwrite(str(snapshot), best_frame)
        snapshot_path = str(snapshot.relative_to(ROOT))

    return {
        "camera_id": camera_id,
        "video": str(video_path.relative_to(ROOT)),
        "ok": True,
        "sampled_frames": samples,
        "person_frames": person_frames,
        "person_frame_ratio_pct": round((person_frames / samples * 100.0) if samples else 0.0, 2),
        "total_person_detections": total_persons,
        "marker_frames": marker_frames,
        "marker_frame_ratio_pct": round((marker_frames / samples * 100.0) if samples else 0.0, 2),
        "matched_person_marker_frames": matched_frames,
        "matched_frame_ratio_pct": round((matched_frames / samples * 100.0) if samples else 0.0, 2),
        "marker_ids": marker_ids,
        "best_detection": best_payload,
        "snapshot": snapshot_path,
    }


def main() -> None:
    cfg = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(ROOT / cfg["detection"]["model_path"]))
    folder = VIDEOS_DIR / DATE_FOLDER
    videos = sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for video in videos:
        row = analyze_video(video, model, cfg, device)
        results.append(row)
        print(
            f"{row['camera_id']}: persons={row.get('person_frame_ratio_pct')}% "
            f"markers={row.get('marker_ids')} matched={row.get('matched_frame_ratio_pct')}%"
        )

    out_json = OUT_DIR / DATE_FOLDER / "employee_marker_detection.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Employee And ArUco Detection",
        "",
        f"Date folder: `{DATE_FOLDER}`",
        f"Device: `{device}`",
        "",
        "| camera | person frames % | marker ids | matched marker-on-person frames % | snapshot |",
        "|---|---:|---|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['camera_id']} | {row.get('person_frame_ratio_pct', 0)} | "
            f"{json.dumps(row.get('marker_ids', {}))} | {row.get('matched_frame_ratio_pct', 0)} | {row.get('snapshot', '')} |"
        )
    out_md = OUT_DIR / DATE_FOLDER / "employee_marker_detection.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

