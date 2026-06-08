import cv2
import yaml
import os
import sys
import pandas as pd
import json
import time
import threading
import argparse
import concurrent.futures
import numpy as np
from pathlib import Path
from datetime import datetime
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
from collections import deque
from urllib.parse import urlparse
from ultralytics import YOLO
from src.runtime.camera_capture import ThreadedCamera
from src.runtime.performance import PerformanceMonitor

try:
    import torch
except Exception:
    torch = None

os.environ.setdefault("MPLCONFIGDIR", str((Path("logs") / "matplotlib").resolve()))
try:
    import supervision as sv
except Exception:
    sv = None

# Keep RTSP stable and reduce FFmpeg console noise during multi-camera runs.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "stimeout;5000000|"
    "max_delay;500000|"
    "fflags;discardcorrupt"
)
try:
    cv2.setLogLevel(0)
except Exception:
    pass
import traceback

def load_config(config_path="config/settings.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_rtsp_credentials(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "rtsp":
            return "", ""
        return parsed.username or "", parsed.password or ""
    except Exception:
        return "", ""


def normalize_camera_entry(cam):
    stream_profiles = cam.get("stream_profiles", {}) or {}
    video_path = cam.get("video_path", "")
    video_playlist = cam.get("video_playlist") or []
    main_stream = stream_profiles.get("main_stream") or video_path
    sub_stream = stream_profiles.get("sub_stream") or ""
    username, password = parse_rtsp_credentials(main_stream or video_path)
    return {
        "id": str(cam.get("id", "")).strip(),
        "label": cam.get("label") or cam.get("id", ""),
        "zone": cam.get("zone", ""),
        "role": cam.get("role") or cam.get("id", ""),
        "enabled": bool(cam.get("enabled", True)),
        "video_path": video_path,
        "video_playlist": video_playlist,
        "video_start_offset_seconds": float(cam.get("video_start_offset_seconds", 0.0) or 0.0),
        "replay": bool(cam.get("replay", False)),
        "stream_profiles": {
            "main_stream": main_stream,
            "sub_stream": sub_stream,
        },
        "username": cam.get("username", username),
        "password": cam.get("password", password),
        "position": cam.get("position", [0, 0]),
        "height_ft": cam.get("height_ft"),
        "notes": cam.get("notes", ""),
        "calibration_points": cam.get("calibration_points", {"camera_points": [], "map_points": []}),
    }


def ensure_camera_registry(config, registry_path):
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        return
    cameras = [normalize_camera_entry(cam) for cam in config.get("cameras", [])]
    payload = {
        "version": 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": cameras,
    }
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_camera_registry(config, registry_path):
    registry_path = Path(registry_path)
    ensure_camera_registry(config, registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    cameras = [normalize_camera_entry(cam) for cam in payload.get("cameras", [])]
    return payload, cameras


def write_camera_registry(registry_path, cameras):
    payload = {
        "version": 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": [normalize_camera_entry(cam) for cam in cameras],
    }
    Path(registry_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

def parse_args():
    parser = argparse.ArgumentParser(description="Run Factory AI real-time tracking")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--camera-profile", default=None, help="Camera stream profile, e.g. main_stream or sub_stream")
    parser.add_argument("--inference-width", type=int, default=None, help="Resize width before AI inference")
    parser.add_argument("--target-fps", type=float, default=None, help="Main processing loop FPS cap")
    parser.add_argument("--batch-size", type=int, default=None, help="Max camera frames per YOLO batch")
    parser.add_argument("--performance-profile", choices=["balanced", "speed", "quality"], default="balanced", help="Performance tuning profile for inference and ArUco detection")
    parser.add_argument("--device", type=str, default="auto", help="Inference device, e.g. cpu, cuda, cuda:0,1")
    parser.add_argument("--aruco-zoom", type=float, default=None, help="Zoom factor for ArUco crop detection")
    parser.add_argument("--camera-registry", default=None, help="Override camera registry JSON file")
    parser.add_argument("--replay-speed", type=float, default=None, help="Local video replay speed multiplier")
    return parser.parse_args()

def get_runtime_config(config, args):
    runtime = config.get("runtime", {})
    log_dir = os.environ.get("FACTORY_AI_LOG_DIR", "logs")
    recording_dir = os.environ.get("FACTORY_AI_RECORDING_DIR", "outputs/recordings")
    report_dir = os.environ.get("FACTORY_AI_REPORT_DIR", "outputs/reports")
    profile_name = args.camera_profile or runtime.get("camera_profile", "main_stream")
    profile = config.get("camera_profiles", {}).get(profile_name, {})
    return {
        "camera_profile": profile_name,
        "camera_profile_settings": profile,
        "inference_width": max(160, int(args.inference_width or runtime.get("inference_width", 1024))),
        "target_fps": max(1.0, float(args.target_fps or config.get("system", {}).get("fps_target", 15))),
        "replay_speed": max(0.1, float(args.replay_speed or runtime.get("replay_speed", 1.0))),
        "batch_size": max(1, int(args.batch_size or runtime.get("batch_size", 8))),
        "aruco_max_width": max(160, int(runtime.get("aruco_max_width", 1920))),
        "aruco_every_n_frames": max(1, int(runtime.get("aruco_every_n_frames", 3))),
        "aruco_workers": max(1, int(runtime.get("aruco_workers", 2))),
        "aruco_crop_first_enabled": bool(runtime.get("aruco_crop_first_enabled", True)),
        "aruco_crop_zoom": max(1.0, float(args.aruco_zoom or runtime.get("aruco_crop_zoom", 4.0))),
        "aruco_crop_margin_ratio": max(0.0, float(runtime.get("aruco_crop_margin_ratio", 0.08))),
        "aruco_full_frame_fallback": bool(runtime.get("aruco_full_frame_fallback", False)),
        "dashboard_image_width": max(320, int(runtime.get("dashboard_image_width", 960))),
        "dashboard_fps": max(1.0, float(runtime.get("dashboard_fps", 5.0))),
        "image_writer_workers": max(1, int(runtime.get("image_writer_workers", 2))),
        "path_video_every_n_frames": max(1, int(runtime.get("path_video_every_n_frames", 10))),
        "path_video_fps": max(0.5, float(runtime.get("path_video_fps", 5.0))),
        "path_video_min_frames": max(1, int(runtime.get("path_video_min_frames", 3))),
        "path_video_codec": str(runtime.get("path_video_codec", "mp4v")),
        "redirect_decoder_logs": bool(runtime.get("redirect_decoder_logs", True)),
        "log_dir": log_dir,
        "recording_dir": recording_dir,
        "report_dir": report_dir,
        "live_stats_file": os.path.join(log_dir, "live_stats.json"),
        "dashboard_live_stats_file": "logs/live_stats.json",
        "decoder_log_file": os.path.join(log_dir, "decoder_ffmpeg.log"),
        "monitor_interval_seconds": runtime.get("monitor_interval_seconds", 2.0),
        "jpeg_quality_live": runtime.get("jpeg_quality_live", 75),
        "reconnect_after": runtime.get("reconnect_after", 10),
        "reconnect_interval_seconds": float(runtime.get("reconnect_interval_seconds", 5.0)),
        "stale_camera_after_seconds": float(runtime.get("stale_camera_after_seconds", 3.0)),
        "marker_confirmations_required": max(1, int(runtime.get("marker_confirmations_required", 2))),
        "marker_stale_after_seconds": float(runtime.get("marker_stale_after_seconds", 8.0)),
        "marker_hidden_grace_seconds": float(runtime.get("marker_hidden_grace_seconds", 12.0)),
        "marker_hidden_match_radius_px": float(runtime.get("marker_hidden_match_radius_px", 180.0)),
        "marker_hidden_max_seconds": float(runtime.get("marker_hidden_max_seconds", 4.0)),
        "marker_hidden_min_iou": float(runtime.get("marker_hidden_min_iou", 0.08)),
        "tracker_iou_threshold": float(runtime.get("tracker_iou_threshold", 0.12)),
        "tracker_center_radius_px": float(runtime.get("tracker_center_radius_px", 180.0)),
        "tracker_max_age_seconds": float(runtime.get("tracker_max_age_seconds", 6.0)),
        "reid_match_threshold": float(runtime.get("reid_match_threshold", 0.72)),
        "reid_max_gap_seconds": float(runtime.get("reid_max_gap_seconds", 45.0)),
        "person_box_margin": float(runtime.get("person_box_margin", 0.15)),
        "activity_upper_motion_threshold": float(runtime.get("activity_upper_motion_threshold", 0.026)),
        "activity_gesture_motion_threshold": float(runtime.get("activity_gesture_motion_threshold", 0.018)),
        "activity_idle_after_seconds": float(runtime.get("activity_idle_after_seconds", 20.0)),
        "min_marker_area": float(runtime.get("min_marker_area", 80.0)),
        "employee_records_file": os.path.join(log_dir, "employee_records.json"),
        "camera_registry_file": args.camera_registry or runtime.get("camera_registry_file", "data/cameras.json"),
        "performance_profile": args.performance_profile or runtime.get("performance_profile", "balanced"),
    }

def select_devices(requested_device):
    if requested_device is None:
        requested_device = "auto"
    requested = [item.strip() for item in requested_device.split(",") if item.strip()]
    if not requested or requested == ["auto"]:
        if torch is not None and torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]

    devices = []
    for item in requested:
        if item == "cpu":
            devices.append("cpu")
        elif item == "cuda":
            if torch is not None and torch.cuda.is_available():
                devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
            else:
                devices.append("cpu")
        elif item.startswith("cuda:"):
            if torch is not None and torch.cuda.is_available():
                try:
                    idx = int(item.split(":", 1)[1])
                    if 0 <= idx < torch.cuda.device_count():
                        devices.append(item)
                except ValueError:
                    pass
            else:
                devices.append("cpu")
        else:
            if torch is not None and torch.cuda.is_available() and item.isdigit():
                idx = int(item)
                if 0 <= idx < torch.cuda.device_count():
                    devices.append(f"cuda:{idx}")
    if not devices:
        return ["cpu"]
    return list(dict.fromkeys(devices))


def load_models_for_devices(model_path, devices):
    models = []
    for device in devices:
        worker_model = YOLO(model_path)
        if hasattr(worker_model, "to"):
            worker_model.to(device)
        models.append({"device": device, "model": worker_model})
    if torch is not None and any(str(device).startswith("cuda") for device in devices):
        torch.backends.cudnn.benchmark = True
    return models


def run_yolo_inference(worker, frames, confidence, classes):
    return worker["model"](
        frames,
        conf=confidence,
        classes=classes,
        verbose=False,
        device=worker["device"],
        half=(worker["device"] != "cpu"),
    )


def apply_performance_profile(runtime):
    profile = runtime.get("performance_profile", "balanced")
    if profile == "balanced":
        runtime["inference_width"] = min(runtime["inference_width"], 1280)
        runtime["target_fps"] = min(runtime["target_fps"], 10.0)
        runtime["batch_size"] = max(runtime["batch_size"], 6)
        runtime["aruco_max_width"] = min(runtime["aruco_max_width"], 1280)
        runtime["aruco_every_n_frames"] = max(runtime["aruco_every_n_frames"], 6)
        runtime["aruco_workers"] = max(1, min(runtime["aruco_workers"], 2))
        runtime["aruco_full_frame_fallback"] = False
        runtime["dashboard_image_width"] = min(runtime["dashboard_image_width"], 1280)
        runtime["dashboard_fps"] = min(runtime["dashboard_fps"], 4.0)
        runtime["jpeg_quality_live"] = max(75, min(runtime["jpeg_quality_live"], 82))
        runtime["stale_camera_after_seconds"] = max(runtime["stale_camera_after_seconds"], 45.0)
        runtime["reconnect_interval_seconds"] = max(runtime["reconnect_interval_seconds"], 15.0)
    elif profile == "speed":
        runtime["inference_width"] = min(runtime["inference_width"], 960)
        runtime["target_fps"] = min(runtime["target_fps"], 8.0)
        runtime["batch_size"] = max(runtime["batch_size"], 8)
        runtime["aruco_max_width"] = min(runtime["aruco_max_width"], 960)
        runtime["aruco_every_n_frames"] = max(2, int(runtime["aruco_every_n_frames"] * 2))
        runtime["aruco_workers"] = max(1, min(runtime["aruco_workers"], 2))
        runtime["aruco_full_frame_fallback"] = False
        runtime["dashboard_image_width"] = min(runtime["dashboard_image_width"], 640)
        runtime["dashboard_fps"] = max(2, min(runtime["dashboard_fps"], 3))
        runtime["jpeg_quality_live"] = min(runtime["jpeg_quality_live"], 55)
        runtime["stale_camera_after_seconds"] = max(runtime["stale_camera_after_seconds"], 60.0)
        runtime["reconnect_interval_seconds"] = max(runtime["reconnect_interval_seconds"], 20.0)
    elif profile == "quality":
        runtime["inference_width"] = max(runtime["inference_width"], min(1920, int(runtime["inference_width"] * 1.25)))
        runtime["aruco_max_width"] = max(runtime["aruco_max_width"], 2560)
        runtime["aruco_crop_zoom"] = max(runtime["aruco_crop_zoom"], 6.0)
        runtime["aruco_every_n_frames"] = max(1, int(runtime["aruco_every_n_frames"] / 2))
        runtime["aruco_workers"] = max(runtime["aruco_workers"], 4)
        runtime["batch_size"] = max(runtime["batch_size"], 4)
    return runtime

def get_acceleration_info(requested_device, selected_device):
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    return {
        "requested_device": requested_device,
        "selected_device": selected_device,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_warning": "CUDA PyTorch is not installed; YOLO will run on CPU." if selected_device == "cpu" else "",
    }

def resolve_camera_source(cam, profile_name):
    stream_profiles = cam.get("stream_profiles", {})
    if profile_name in stream_profiles:
        return stream_profiles[profile_name]
    return cam.get("video_path")


def camera_runtime_signature(cam, profile_name):
    return {
        "id": cam.get("id"),
        "enabled": bool(cam.get("enabled", True)),
        "source": resolve_camera_source(cam, profile_name),
        "video_playlist": cam.get("video_playlist") or [],
        "video_start_offset_seconds": float(cam.get("video_start_offset_seconds", 0.0) or 0.0),
        "fps": cam.get("fps"),
        "position": cam.get("position"),
        "height_ft": cam.get("height_ft"),
        "stream_profiles": cam.get("stream_profiles", {}),
    }

def parse_resolution(resolution):
    if not resolution:
        return None, None
    try:
        width, height = resolution.lower().split("x", 1)
        return int(width), int(height)
    except ValueError:
        return None, None

def chunked(items, size):
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]

def resize_to_width(frame, max_width):
    h, w = frame.shape[:2]
    if not max_width or w <= max_width:
        return frame, 1.0
    scale = w / max_width
    resized = cv2.resize(frame, (max_width, int(h * (max_width / w))))
    return resized, scale

def detect_aruco_markers(aruco_dict, frame, max_width):
    aruco_frame, aruco_scale = resize_to_width(frame, max_width)
    gray = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detector_params())
    corners, ids = detect_markers_with_variants(detector, gray)
    if corners is not None and aruco_scale != 1.0:
        corners = tuple(corner * aruco_scale for corner in corners)
    return corners, ids

def aruco_detector_params():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.035
    params.errorCorrectionRate = 0.8
    return params

def detect_markers_with_variants(detector, gray):
    variants = [gray, cv2.equalizeHist(gray)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))
    variants.append(cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)))

    seen = set()
    merged_corners = []
    merged_ids = []
    for variant in variants:
        corners, ids, _ = detector.detectMarkers(variant)
        if ids is None:
            continue
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id in seen:
                continue
            seen.add(marker_id)
            merged_corners.append(marker_corners)
            merged_ids.append(marker_id)
    if not merged_ids:
        return None, None
    return tuple(merged_corners), np.array([[marker_id] for marker_id in merged_ids], dtype=np.int32)

def detect_aruco_in_person_crops(aruco_dict, frame, person_boxes, zoom_scale, margin_ratio, min_marker_area):
    if not person_boxes:
        return None, None, {}
    height, width = frame.shape[:2]
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detector_params())
    candidates = []

    for person_index, person in enumerate(person_boxes):
        x1, y1, x2, y2 = person["bbox"]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad_x = box_w * margin_ratio
        pad_y = box_h * margin_ratio
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(width, int(round(x2 + pad_x)))
        cy2 = min(height, int(round(y2 + pad_y)))
        if cx2 - cx1 < 20 or cy2 - cy1 < 30:
            continue

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        zoom = cv2.resize(crop, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
        crop_corners, crop_ids = detect_markers_with_variants(detector, gray)
        if crop_ids is None:
            continue

        for marker_id, marker_corners in zip(crop_ids.flatten(), crop_corners):
            pts_zoom = marker_corners[0].astype(np.float32)
            area = float(cv2.contourArea(pts_zoom)) / (zoom_scale * zoom_scale)
            if area < min_marker_area:
                continue
            pts_full = (pts_zoom / zoom_scale) + np.array([cx1, cy1], dtype=np.float32)
            marker_center = pts_full.mean(axis=0)
            inside_score = 1.0 if expanded_contains(person["bbox"], marker_center, 0.0) else 0.0
            candidates.append({
                "marker_id": int(marker_id),
                "corners": pts_full,
                "person": person,
                "person_index": person_index,
                "score": (inside_score * 100000.0) + area - bbox_area(person["bbox"]) * 0.0001,
            })

    if not candidates:
        return None, None, {}

    best_by_marker = {}
    for item in candidates:
        marker_id = item["marker_id"]
        previous = best_by_marker.get(marker_id)
        if previous is None or item["score"] > previous["score"]:
            best_by_marker[marker_id] = item

    ordered = sorted(best_by_marker.values(), key=lambda item: item["score"], reverse=True)
    corners = tuple(np.array([item["corners"]], dtype=np.float32) for item in ordered)
    ids = np.array([[item["marker_id"]] for item in ordered], dtype=np.int32)
    marker_person_matches = {idx: item["person"] for idx, item in enumerate(ordered)}
    return corners, ids, marker_person_matches

def merge_aruco_detections(primary_corners, primary_ids, primary_matches, fallback_corners, fallback_ids):
    merged_corners = []
    merged_ids = []
    merged_matches = {}
    seen_ids = set()

    if primary_ids is not None and primary_corners is not None:
        for idx, marker_id in enumerate(primary_ids.flatten()):
            marker_id = int(marker_id)
            merged_matches[len(merged_ids)] = primary_matches.get(idx)
            merged_corners.append(primary_corners[idx])
            merged_ids.append(marker_id)
            seen_ids.add(marker_id)

    if fallback_ids is not None and fallback_corners is not None:
        for idx, marker_id in enumerate(fallback_ids.flatten()):
            marker_id = int(marker_id)
            if marker_id in seen_ids:
                continue
            merged_corners.append(fallback_corners[idx])
            merged_ids.append(marker_id)
            seen_ids.add(marker_id)

    if not merged_ids:
        return None, None, {}
    return tuple(merged_corners), np.array([[marker_id] for marker_id in merged_ids], dtype=np.int32), merged_matches

def scale_person_boxes(results, scale):
    boxes = []
    if results.boxes is None:
        return boxes
    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v * scale) for v in box]
        boxes.append({"bbox": (x1, y1, x2, y2), "confidence": float(conf)})
    return boxes

def polygon_area(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

def bbox_from_points(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) == 0:
        return None
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    return float(min_x), float(min_y), float(max_x), float(max_y)

def point_in_polygon(point, polygon):
    if point is None or not polygon:
        return False
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), tuple(map(float, point)), False) >= 0

def load_floor_map_model(path="data/factory_map/rev03_manual_points.yaml"):
    model_path = Path(path)
    if not model_path.exists():
        return {}
    with model_path.open("r", encoding="utf-8") as f:
        model = yaml.safe_load(f) or {}

    point_lookup = {name: tuple(value) for name, value in (model.get("points") or {}).items()}
    zones = {}
    camera_zone = {}
    for zone in model.get("zones") or []:
        zone_points = [point_lookup[name] for name in zone.get("points", []) if name in point_lookup]
        metrics = zone.get("metrics", {}) or {}
        area_px2 = float(metrics.get("area_px2") or polygon_area(zone_points))
        area_sqft = float(metrics.get("area_sqft") or 0.0)
        ft_per_map_px = (area_sqft / area_px2) ** 0.5 if area_px2 > 0 and area_sqft > 0 else None
        zone_info = {
            "id": zone.get("id"),
            "label": zone.get("label") or zone.get("id"),
            "points": zone_points,
            "bbox": bbox_from_points(zone_points),
            "ft_per_map_px": ft_per_map_px,
            "area_px2": area_px2,
            "area_sqft": area_sqft,
        }
        zones[zone.get("id")] = zone_info
        for camera_id in zone.get("cameras") or []:
            camera_zone[camera_id] = zone_info

    coverage = {}
    for item in model.get("camera_coverage_estimates") or []:
        visible = item.get("visible_polygon") or []
        coverage[item.get("id")] = {
            "visible_polygon": visible,
            "bbox": bbox_from_points(visible),
        }

    return {
        "zones": zones,
        "camera_zone": camera_zone,
        "coverage": coverage,
    }

def resolve_worker_zone(map_pos, cam_id, floor_model):
    for zone in floor_model.get("zones", {}).values():
        if point_in_polygon(map_pos, zone.get("points")):
            return zone
    return floor_model.get("camera_zone", {}).get(cam_id)

def build_distance_calibrations(cameras, floor_model):
    calibrations = {}
    for cam in cameras:
        cam_id = cam.get("id")
        points = cam.get("calibration_points") or {}
        camera_points = points.get("camera_points") or []
        map_points = points.get("map_points") or []
        zone = floor_model.get("camera_zone", {}).get(cam_id)
        coverage = floor_model.get("coverage", {}).get(cam_id)
        ft_per_map_px = zone.get("ft_per_map_px") if zone else None

        homography = None
        if len(camera_points) >= 4 and len(camera_points) == len(map_points):
            src = np.array(camera_points, dtype=np.float32)
            dst = np.array(map_points, dtype=np.float32)
            homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        target_bbox = None
        method = "legacy_video_pixels"
        if homography is not None and ft_per_map_px:
            method = "homography_to_approved_map_scale"
        elif coverage and coverage.get("bbox") and ft_per_map_px:
            target_bbox = coverage["bbox"]
            method = "coverage_bbox_to_approved_map_scale"
        elif zone and zone.get("bbox") and ft_per_map_px:
            target_bbox = zone["bbox"]
            method = "zone_bbox_to_approved_map_scale"

        calibrations[cam_id] = {
            "homography": homography,
            "target_bbox": target_bbox,
            "ft_per_map_px": ft_per_map_px,
            "method": method,
            "zone_id": zone.get("id") if zone else None,
        }
    return calibrations

def frame_point_to_map(point, frame_shape, calibration):
    if not calibration:
        return None
    point = np.array(point, dtype=np.float32)
    homography = calibration.get("homography")
    if homography is not None:
        src = point.reshape(1, 1, 2)
        return cv2.perspectiveTransform(src, homography).reshape(2)

    bbox = calibration.get("target_bbox")
    if not bbox:
        return None
    height, width = frame_shape[:2]
    if width <= 1 or height <= 1:
        return None
    min_x, min_y, max_x, max_y = bbox
    nx = float(np.clip(point[0] / width, 0.0, 1.0))
    ny = float(np.clip(point[1] / height, 0.0, 1.0))
    return np.array([
        min_x + nx * (max_x - min_x),
        min_y + ny * (max_y - min_y),
    ], dtype=np.float32)

def distance_feet_between_points(prev_point, next_point, calibration, legacy_scale):
    if prev_point is not None and next_point is not None and calibration and calibration.get("ft_per_map_px"):
        d_map_px = float(np.linalg.norm(np.array(next_point) - np.array(prev_point)))
        return d_map_px * float(calibration["ft_per_map_px"]), calibration.get("method", "map_scale")
    if prev_point is None or next_point is None:
        return 0.0, "not_enough_points"
    d_pixels = float(np.linalg.norm(np.array(next_point) - np.array(prev_point)))
    return (d_pixels / legacy_scale) * 3.28084, "legacy_video_pixels"

def expanded_contains(box, point, margin_ratio):
    x1, y1, x2, y2 = box
    px, py = point
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    return x1 - margin_x <= px <= x2 + margin_x and y1 - margin_y <= py <= y2 + margin_y

def marker_area(corners):
    return float(cv2.contourArea(corners.astype(np.float32)))

def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union

def bbox_aspect(box):
    x1, y1, x2, y2 = box
    return max(1.0, float(y2 - y1)) / max(1.0, float(x2 - x1))

def person_appearance_feature(frame, bbox):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 12 or y2 - y1 < 18:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    feature = cv2.normalize(hist, None).flatten().astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        return None
    return feature / norm

def feature_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return 0.0
    return float(np.dot(feature_a, feature_b))

class ByteTrackStyleTracker:
    """Camera-local tracking plus lightweight appearance ReID.

    This is intentionally small and dependency-free. It gives us the behavior
    we need now: stable local person IDs, ArUco-to-track binding, and a first
    appearance-based bridge between cameras. A deep OSNet model can replace the
    feature extractor later without changing the main loop contract.
    """

    def __init__(self, runtime):
        self.iou_threshold = float(runtime["tracker_iou_threshold"])
        self.center_radius_px = float(runtime["tracker_center_radius_px"])
        self.max_age_seconds = float(runtime["tracker_max_age_seconds"])
        self.reid_threshold = float(runtime["reid_match_threshold"])
        self.reid_max_gap_seconds = float(runtime["reid_max_gap_seconds"])
        self.next_track_id = 1
        self.next_global_id = 1
        self.byte_trackers = {}
        self.tracks_by_camera = {}
        self.global_profiles = {}

    def _new_track_id(self):
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _new_global_id(self):
        global_id = f"person_{self.next_global_id:04d}"
        self.next_global_id += 1
        return global_id

    def _cleanup(self, cam_id, now_ts):
        tracks = self.tracks_by_camera.setdefault(cam_id, {})
        stale = [
            track_id for track_id, track in tracks.items()
            if now_ts - track.get("last_seen_ts", now_ts) > self.max_age_seconds
        ]
        for track_id in stale:
            tracks.pop(track_id, None)

    def _byte_tracker(self, cam_id):
        if sv is None or not hasattr(sv, "ByteTrack"):
            return None
        tracker = self.byte_trackers.get(cam_id)
        if tracker is None:
            tracker = sv.ByteTrack(
                track_activation_threshold=0.2,
                lost_track_buffer=max(15, int(self.max_age_seconds * 15)),
                minimum_matching_threshold=0.72,
                frame_rate=15,
                minimum_consecutive_frames=1,
            )
            self.byte_trackers[cam_id] = tracker
        return tracker

    def _assign_global_id(self, cam_id, feature, now_ts):
        if feature is None:
            return self._new_global_id()
        best_id = None
        best_score = 0.0
        for global_id, profile in self.global_profiles.items():
            if profile.get("employee_marker_id") is not None:
                continue
            if profile.get("last_camera") == cam_id:
                continue
            if now_ts - profile.get("last_seen_ts", 0.0) > self.reid_max_gap_seconds:
                continue
            score = feature_similarity(feature, profile.get("feature"))
            if score > best_score:
                best_id = global_id
                best_score = score
        if best_id and best_score >= self.reid_threshold:
            return best_id
        return self._new_global_id()

    def _update_global_profile(self, global_id, cam_id, feature, now_ts, employee_marker_id=None):
        profile = self.global_profiles.get(global_id, {})
        old_feature = profile.get("feature")
        if feature is not None and old_feature is not None:
            feature = (old_feature * 0.7) + (feature * 0.3)
            norm = float(np.linalg.norm(feature))
            if norm > 1e-6:
                feature = feature / norm
        elif feature is None:
            feature = old_feature
        if employee_marker_id is None:
            employee_marker_id = profile.get("employee_marker_id")
        profile.update({
            "feature": feature,
            "last_camera": cam_id,
            "last_seen_ts": now_ts,
            "employee_marker_id": employee_marker_id,
        })
        self.global_profiles[global_id] = profile

    def update(self, cam_id, detections, frame, now_ts):
        self._cleanup(cam_id, now_ts)
        tracks = self.tracks_by_camera.setdefault(cam_id, {})
        for det in detections:
            det["feature"] = person_appearance_feature(frame, det["bbox"])

        byte_tracker = self._byte_tracker(cam_id)
        if byte_tracker is not None and detections:
            xyxy = np.array([det["bbox"] for det in detections], dtype=np.float32)
            confidence = np.array([det.get("confidence", 0.0) for det in detections], dtype=np.float32)
            class_id = np.zeros(len(detections), dtype=int)
            sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = byte_tracker.update_with_detections(sv_detections)
            matched_det_indices = set()
            for box, tracker_id in zip(tracked.xyxy, tracked.tracker_id):
                if tracker_id is None:
                    continue
                best_idx = None
                best_iou = 0.0
                for idx, det in enumerate(detections):
                    if idx in matched_det_indices:
                        continue
                    iou = bbox_iou(tuple(box.tolist()), det["bbox"])
                    if iou > best_iou:
                        best_idx = idx
                        best_iou = iou
                if best_idx is None or best_iou <= 0.05:
                    continue
                matched_det_indices.add(best_idx)
                det = detections[best_idx]
                track_id = f"bt_{int(tracker_id)}"
                track = tracks.get(track_id)
                if track is None:
                    global_id = self._assign_global_id(cam_id, det.get("feature"), now_ts)
                    track = {
                        "bbox": det["bbox"],
                        "last_seen_ts": now_ts,
                        "hits": 0,
                        "confidence": det.get("confidence", 0.0),
                        "feature": det.get("feature"),
                        "global_id": global_id,
                        "employee_marker_id": None,
                    }
                    tracks[track_id] = track
                track["bbox"] = det["bbox"]
                track["last_seen_ts"] = now_ts
                track["hits"] = track.get("hits", 0) + 1
                track["confidence"] = det.get("confidence", 0.0)
                if det.get("feature") is not None:
                    track["feature"] = det["feature"]
                self._update_global_profile(
                    track["global_id"],
                    cam_id,
                    det.get("feature"),
                    now_ts,
                    track.get("employee_marker_id"),
                )
                det["track_id"] = track_id
                det["global_id"] = track["global_id"]
                det["employee_marker_id"] = track.get("employee_marker_id")

        candidates = []
        for track_id, track in tracks.items():
            track_center = bbox_center(track["bbox"])
            for det_idx, det in enumerate(detections):
                if det.get("track_id") is not None:
                    continue
                iou = bbox_iou(track["bbox"], det["bbox"])
                center_dist = float(np.linalg.norm(track_center - bbox_center(det["bbox"])))
                if iou >= self.iou_threshold or center_dist <= self.center_radius_px:
                    score = (iou * 100.0) - (center_dist * 0.05) + (det.get("confidence", 0.0) * 5.0)
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True, key=lambda item: item[0])

        assigned_tracks = set()
        assigned_dets = set()
        for _, track_id, det_idx in candidates:
            if track_id in assigned_tracks or det_idx in assigned_dets:
                continue
            track = tracks.get(track_id)
            if track is None:
                continue
            det = detections[det_idx]
            assigned_tracks.add(track_id)
            assigned_dets.add(det_idx)
            track["bbox"] = det["bbox"]
            track["last_seen_ts"] = now_ts
            track["hits"] = track.get("hits", 0) + 1
            track["confidence"] = det.get("confidence", 0.0)
            if det.get("feature") is not None:
                track["feature"] = det["feature"]
            self._update_global_profile(
                track["global_id"],
                cam_id,
                det.get("feature"),
                now_ts,
                track.get("employee_marker_id"),
            )
            det["track_id"] = track_id
            det["global_id"] = track["global_id"]
            det["employee_marker_id"] = track.get("employee_marker_id")

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets or det.get("track_id") is not None:
                continue
            track_id = self._new_track_id()
            global_id = self._assign_global_id(cam_id, det.get("feature"), now_ts)
            tracks[track_id] = {
                "bbox": det["bbox"],
                "last_seen_ts": now_ts,
                "hits": 1,
                "confidence": det.get("confidence", 0.0),
                "feature": det.get("feature"),
                "global_id": global_id,
                "employee_marker_id": None,
            }
            self._update_global_profile(global_id, cam_id, det.get("feature"), now_ts)
            det["track_id"] = track_id
            det["global_id"] = global_id
            det["employee_marker_id"] = None

        return detections

    def bind_employee(self, cam_id, track_id, marker_id):
        track = self.tracks_by_camera.get(cam_id, {}).get(track_id)
        if not track:
            return
        old_global_id = track.get("global_id")
        employee_global_id = f"emp_{int(marker_id)}"
        track["employee_marker_id"] = int(marker_id)
        track["global_id"] = employee_global_id
        self._update_global_profile(
            employee_global_id,
            cam_id,
            track.get("feature"),
            track.get("last_seen_ts", time.time()),
            int(marker_id),
        )
        if old_global_id and old_global_id != employee_global_id:
            self.global_profiles.pop(old_global_id, None)

    def find_employee_detection(self, cam_id, marker_id, detections, used_box_ids):
        for idx, det in enumerate(detections):
            if idx in used_box_ids:
                continue
            if det.get("employee_marker_id") == int(marker_id):
                used_box_ids.add(idx)
                return det
        return None

def find_person_for_marker(marker_center, person_boxes, margin_ratio):
    strict_containing = [
        person for person in person_boxes
        if expanded_contains(person["bbox"], marker_center, 0.0)
    ]
    containing = strict_containing or [
        person for person in person_boxes
        if expanded_contains(person["bbox"], marker_center, margin_ratio)
    ]
    if not containing:
        return None
    marker_point = np.array(marker_center, dtype=np.float32)
    return min(
        containing,
        key=lambda person: (
            float(np.linalg.norm(bbox_center(person["bbox"]) - marker_point)),
            bbox_area(person["bbox"]),
            -float(person.get("confidence", 0.0)),
        ),
    )

def person_floor_point(person):
    x1, y1, x2, y2 = person["bbox"]
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)

def person_posture(person):
    x1, y1, x2, y2 = person["bbox"]
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    aspect = height / width
    if aspect < 1.35:
        return "sitting_or_bending"
    if aspect < 1.7:
        return "leaning_or_working"
    return "standing"

def upper_body_patch(frame, bbox):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    upper_y2 = y1 + max(8, int((y2 - y1) * 0.58))
    crop = frame[y1:upper_y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)

def classify_worker_activity(worker_state, person, frame, d_pixels, now_ts, runtime):
    patch = upper_body_patch(frame, person["bbox"])
    motion_score = 0.0
    previous_patch = worker_state.get("activity_patch")
    if patch is not None and previous_patch is not None:
        diff = cv2.absdiff(patch, previous_patch)
        motion_score = float(np.mean(diff)) / 255.0
    if patch is not None:
        worker_state["activity_patch"] = patch

    previous_ema = float(worker_state.get("activity_motion_ema", 0.0))
    motion_ema = (previous_ema * 0.65) + (motion_score * 0.35)
    worker_state["activity_motion_ema"] = motion_ema

    moving = d_pixels > 8.0
    posture = person_posture(person)
    if moving:
        worker_state["last_move_ts"] = now_ts

    upper_threshold = runtime["activity_upper_motion_threshold"]
    gesture_threshold = runtime["activity_gesture_motion_threshold"]
    idle_after = runtime["activity_idle_after_seconds"]

    if motion_ema >= upper_threshold:
        status = "WORKING"
        reason = "upper_body_or_hand_motion"
        worker_state["last_work_motion_ts"] = now_ts
    elif moving and motion_ema >= gesture_threshold:
        status = "WORKING"
        reason = "walking_with_gesture_or_material"
        worker_state["last_work_motion_ts"] = now_ts
    elif moving:
        status = "WALKING"
        reason = "walking_without_work_motion"
    else:
        last_active = max(
            worker_state.get("last_work_motion_ts", now_ts),
            worker_state.get("last_move_ts", now_ts),
        )
        if now_ts - last_active >= idle_after:
            status = "IDLE_SITTING" if posture == "sitting_or_bending" else "IDLE_STANDING"
            reason = "no_body_motion_after_idle_window"
        else:
            status = "WORKING"
            reason = "stationary_work_grace"

    worker_state["status"] = status
    worker_state["activity_reason"] = reason
    worker_state["posture"] = posture
    worker_state["activity_motion_score"] = round(motion_score, 4)
    worker_state["activity_motion_ema"] = round(motion_ema, 4)
    return status

def find_continuation_person(worker_state, person_boxes, used_box_ids, max_radius_px, min_iou):
    if not person_boxes:
        return None
    last_points = worker_state.get("path") or []
    if not last_points:
        return None
    last_pos = np.array(last_points[-1], dtype=np.float32)
    last_bbox = worker_state.get("last_bbox")
    last_center = bbox_center(last_bbox) if last_bbox else None
    last_aspect = bbox_aspect(last_bbox) if last_bbox else None
    last_track_id = worker_state.get("track_id")
    last_feature = worker_state.get("last_feature")
    candidates = []
    for idx, person in enumerate(person_boxes):
        if idx in used_box_ids:
            continue
        if last_track_id is not None and person.get("track_id") != last_track_id:
            continue
        floor_pos = person_floor_point(person)
        distance_px = float(np.linalg.norm(floor_pos - last_pos))
        iou = bbox_iou(last_bbox, person["bbox"]) if last_bbox else 0.0
        center_distance = float(np.linalg.norm(bbox_center(person["bbox"]) - last_center)) if last_center is not None else distance_px
        feature_score = feature_similarity(last_feature, person.get("feature"))
        same_track = last_track_id is not None and person.get("track_id") == last_track_id
        spatial_good = distance_px <= max_radius_px or center_distance <= max_radius_px
        strong_visual = feature_score >= 0.58
        weak_visual = feature_score >= 0.48

        if not same_track and not (spatial_good and weak_visual) and not strong_visual:
            continue

        aspect_penalty = 0.0
        if last_aspect:
            aspect_penalty = abs(np.log(bbox_aspect(person["bbox"]) / last_aspect))
        area_penalty = 0.0
        if last_bbox:
            prev_area = max(1.0, bbox_area(last_bbox))
            next_area = max(1.0, bbox_area(person["bbox"]))
            area_penalty = abs(np.log(next_area / prev_area))
            if not same_track and area_penalty > 1.45 and not strong_visual:
                continue
            if not same_track and aspect_penalty > 1.20 and not strong_visual:
                continue
            if not same_track and iou < min_iou and not weak_visual:
                continue

        score = (
            distance_px * 0.45
            + center_distance * 0.30
            - iou * 180.0
            - feature_score * 180.0
            + area_penalty * 22.0
            + aspect_penalty * 18.0
        )
        if same_track:
            score -= 250.0
        candidates.append((score, idx, person))
    if not candidates:
        return None
    _, idx, person = min(candidates, key=lambda item: item[0])
    used_box_ids.add(idx)
    return person

def accept_path_step(worker_state, dist_feet, d_pixels, tracking_mode):
    if len(worker_state.get("path", [])) <= 1:
        return True
    if dist_feet > 20:
        return False
    if tracking_mode != "aruco" and d_pixels > 220:
        return False
    return True

def camera_time_scale(stream_status, runtime):
    if stream_status and stream_status.get("replay_speed"):
        return max(1.0, float(stream_status.get("replay_speed") or runtime.get("replay_speed", 1.0)))
    return 1.0

def likely_recent_tracked_person(person, worker_states, cam_id, now_ts, max_seconds, max_radius_px):
    floor_pos = person_floor_point(person)
    center = bbox_center(person["bbox"])
    for state in worker_states:
        if state.get("cam_id") != cam_id:
            continue
        if now_ts - state.get("last_marker_seen_ts", 0.0) > max_seconds:
            continue
        last_points = state.get("path") or []
        last_bbox = state.get("last_bbox")
        if not last_points or not last_bbox:
            continue
        floor_distance = float(np.linalg.norm(floor_pos - np.array(last_points[-1], dtype=np.float32)))
        center_distance = float(np.linalg.norm(center - bbox_center(last_bbox)))
        if floor_distance <= max_radius_px * 0.75 or center_distance <= max_radius_px * 0.65:
            return True
    return False

def account_worker_time(worker_state, now_ts, stale_after_seconds):
    last_account_ts = worker_state.get("last_account_ts", now_ts)
    delta = max(0.0, min(now_ts - last_account_ts, 1.0))
    worker_state["last_account_ts"] = now_ts

    if delta <= 0:
        return

    # Marker/ArUco can be hidden by body angle, product stacks, or camera glare.
    # Do not change the employee to LOST just because the marker is temporarily stale.
    if now_ts - worker_state.get("last_seen_ts", now_ts) > stale_after_seconds:
        return

    cam_id = worker_state.get("cam_id", "unknown")
    worker_state.setdefault("camera_times", {})
    worker_state.setdefault("status_times", {"WALKING": 0.0, "WORKING": 0.0})
    worker_state["camera_times"][cam_id] = worker_state["camera_times"].get(cam_id, 0.0) + delta
    status = worker_state.get("status", "WORKING")
    if status == "LOST":
        status = "WORKING"
        worker_state["status"] = status
    worker_state["status_times"].pop("LOST", None)
    worker_state["status_times"][status] = worker_state["status_times"].get(status, 0.0) + delta

def build_employee_record(marker_id, worker_state, now_ts):
    if worker_state.get("status") == "LOST":
        worker_state["status"] = "WORKING"
    if "status_times" in worker_state:
        worker_state["status_times"].pop("LOST", None)
    last_seen_age = now_ts - worker_state.get("last_seen_ts", now_ts)
    visible_now = bool(worker_state.get("visible_now", False))
    display_status = worker_state["status"] if visible_now or last_seen_age <= 8.0 else "NOT_VISIBLE"
    tracking_mode = worker_state.get("tracking_mode", "unknown") if visible_now or last_seen_age <= 8.0 else "waiting_for_reappearance"
    return {
        "id": worker_state.get("display_marker_id", marker_id),
        "opencv_marker_id": worker_state.get("opencv_marker_id", marker_id),
        "name": worker_state["name"],
        "dept": worker_state["dept"],
        "status": display_status,
        "current_camera": worker_state.get("cam_id"),
        "zone": worker_state.get("zone_label", "Unknown Zone"),
        "zone_id": worker_state.get("zone_id", "unknown"),
        "total_distance_ft": float(round(worker_state["dist"], 2)),
        "first_seen_ts": round(float(worker_state.get("first_seen_ts", now_ts)), 3),
        "last_seen_ts": round(float(worker_state.get("last_seen_ts", now_ts)), 3),
        "last_seen_age": round(last_seen_age, 2),
        "visible_now": visible_now,
        "person_conf": round(worker_state.get("person_conf", 0.0), 3),
        "distance_method": worker_state.get("distance_method", "unknown"),
        "tracking_mode": tracking_mode,
        "posture": worker_state.get("posture", "unknown"),
        "activity_reason": worker_state.get("activity_reason", "unknown"),
        "activity_motion": worker_state.get("activity_motion_ema", 0.0),
        "camera_times_sec": {k: round(v, 2) for k, v in worker_state.get("camera_times", {}).items()},
        "status_times_sec": {k: round(v, 2) for k, v in worker_state.get("status_times", {}).items()},
        "path_view": worker_state.get("path_view", recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"], worker_state.get("cam_id", "unknown"))),
    }

def recording_root():
    return os.environ.get("FACTORY_AI_RECORDING_DIR", "outputs/recordings")

def report_root():
    return os.environ.get("FACTORY_AI_REPORT_DIR", "outputs/reports")

def safe_filename(value):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "employee"

def recording_filename(marker_id, employee_name, cam_id):
    return f"{marker_id}_{safe_filename(employee_name)}_{safe_filename(cam_id)}.mp4"

def recording_disk_path(marker_id, employee_name, cam_id):
    return str(Path(recording_root()) / recording_filename(marker_id, employee_name, cam_id))

def recording_view_path(marker_id, employee_name, cam_id):
    return "/" + str(Path(recording_root()) / recording_filename(marker_id, employee_name, cam_id)).replace("\\", "/")

def write_jpeg_atomic(path, frame, quality):
    temp_path = path.replace(".jpg", "_tmp.jpg")
    cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    try:
        os.replace(temp_path, path)
    except OSError:
        pass

def clear_startup_outputs(runtime):
    Path("logs").mkdir(exist_ok=True)
    Path("outputs/live").mkdir(parents=True, exist_ok=True)
    Path(runtime["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(runtime["recording_dir"]).mkdir(parents=True, exist_ok=True)
    Path(runtime["report_dir"]).mkdir(parents=True, exist_ok=True)

    empty_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(runtime["employee_records_file"]).write_text(
        json.dumps({"timestamp": empty_timestamp, "employees": []}, indent=2),
        encoding="utf-8",
    )
    startup_payload = json.dumps(
        {
            "timestamp": empty_timestamp,
            "uptime": 0,
            "cameras": [],
            "total_person_count": 0,
            "stats": [],
            "performance": {},
            "runtime": {},
            "acceleration": {},
            "lunch_mode": False,
            "break_message": "",
            "startup_message": "Backend starting. Waiting for first processed frames.",
        },
        indent=2,
    )
    for live_stats_path in {runtime["live_stats_file"], runtime["dashboard_live_stats_file"]}:
        Path(live_stats_path).parent.mkdir(parents=True, exist_ok=True)
        Path(live_stats_path).write_text(startup_payload, encoding="utf-8")

    for path in Path("outputs/live").glob("*.jpg"):
        try:
            path.unlink()
        except OSError:
            pass
class VideoRecorder:
    def __init__(self, fps, codec, min_frames=3):
        self.fps = fps
        self.codec = codec
        self.min_frames = min_frames
        self._writers = {}

    def write(self, path, frame):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        height, width = frame.shape[:2]
        writer_info = self._writers.get(path)
        if writer_info is None or writer_info["size"] != (width, height):
            if writer_info is not None:
                writer_info["writer"].release()
            fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
            writer = cv2.VideoWriter(path, fourcc, self.fps, (width, height))
            if not writer.isOpened():
                return False
            self._writers[path] = {"writer": writer, "size": (width, height), "frames": 0}
            writer_info = self._writers[path]
        writer_info["writer"].write(frame)
        writer_info["frames"] += 1
        return True

    def close(self):
        for path, writer_info in self._writers.items():
            writer_info["writer"].release()
            if writer_info.get("frames", 0) < self.min_frames:
                try:
                    Path(path).unlink()
                except OSError:
                    pass
        self._writers.clear()

def export_employee_report_on_shutdown():
    try:
        from tools.reporting.export_employee_pdf import build_pdf, load_json, timestamped_report_path

        log_dir = Path(os.environ.get("FACTORY_AI_LOG_DIR", "logs"))
        records_payload = load_json(log_dir / "employee_records.json")
        live_payload = load_json(log_dir / "live_stats.json")
        if records_payload.get("employees") or records_payload.get("stats"):
            output_path = timestamped_report_path(report_root())
            build_pdf(records_payload, live_payload, output_path)
            print(f"Employee activity PDF saved to {output_path}")
    except Exception as exc:
        print(f"Employee activity PDF export skipped: {exc}")

def redirect_decoder_stderr(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_handle = open(log_file, "a", buffering=1, encoding="utf-8", errors="replace")
    log_handle.write(f"\n--- decoder log started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    sys.stderr.flush()
    os.dup2(log_handle.fileno(), 2)
    return log_handle

def get_break_info():
    now = datetime.now()
    day = now.weekday() # 0=Mon, 4=Fri
    hour = now.hour
    minute = now.minute
    
    # 9:00 - 9:15 Team Break
    if hour == 9 and minute < 15:
        return "TEA BREAK IN PROGRESS"
    
    # 1:00 PM Lunch Break
    if hour == 13:
        if day == 4: # Friday: 1:00 - 2:00
            return "FRIDAY PRAYER / LUNCH BREAK"
        elif minute < 30: # Other days: 1:00 - 1:30
            return "LUNCH BREAK IN PROGRESS"
            
    # 3:00 - 3:20 Tea Break (Extended buffer for demo stability)
    if hour == 15 and minute < 20:
        return "TEA BREAK IN PROGRESS"
        
    return None

def start_server(registry_path, registry_lock, port=8000):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload, status=200):
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/cameras"):
                with registry_lock:
                    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
                return self._send_json(payload)
            return super().do_GET()

        def do_POST(self):
            if self.path == "/api/cameras":
                payload = self._read_json()
                cameras = payload.get("cameras", [])
                if not isinstance(cameras, list):
                    return self._send_json({"error": "cameras must be a list"}, status=400)
                normalized = [normalize_camera_entry(cam) for cam in cameras if str(cam.get("id", "")).strip()]
                with registry_lock:
                    saved = write_camera_registry(registry_path, normalized)
                return self._send_json(saved)

            if self.path == "/api/cameras/test":
                payload = self._read_json()
                camera = normalize_camera_entry(payload.get("camera", {}))
                source = resolve_camera_source(camera, payload.get("profile", "main_stream"))
                if not source:
                    return self._send_json({"ok": False, "error": "missing camera source"}, status=400)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    return self._send_json({"ok": False, "error": "unable to read frame"}, status=200)
                snapshot_name = f"test_{camera['id'] or 'camera'}.jpg"
                snapshot_path = Path("outputs/live") / snapshot_name
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                write_jpeg_atomic(str(snapshot_path), frame, 70)
                return self._send_json({
                    "ok": True,
                    "snapshot": f"/outputs/live/{snapshot_name}",
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                })

            return self._send_json({"error": "not found"}, status=404)

    class ReusableThreadingTCPServer(ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    while port < 8010:
        try:
            with ReusableThreadingTCPServer(("", port), QuietHandler) as httpd:
                with open("logs/current_port.txt", "w") as f:
                    f.write(str(port))
                httpd.serve_forever()
        except OSError:
            port += 1


def build_threaded_camera(cam, runtime):
    profile = runtime["camera_profile_settings"]
    cap_width, cap_height = parse_resolution(profile.get("resolution"))
    source = resolve_camera_source(cam, runtime["camera_profile"])
    if not source:
        return None
    camera = ThreadedCamera(
        camera_id=cam["id"],
        video_path=source,
        video_playlist=cam.get("video_playlist") or None,
        video_start_offset_seconds=cam.get("video_start_offset_seconds", 0.0),
        replay_speed=runtime["replay_speed"],
        width=cap_width,
        height=cap_height,
        fps=profile.get("fps"),
        reconnect_after=runtime["reconnect_after"],
        reconnect_interval_seconds=runtime["reconnect_interval_seconds"],
        stale_after_seconds=runtime["stale_camera_after_seconds"],
    )
    camera.label = cam.get("label") or cam.get("id")
    camera.zone = cam.get("zone", "")
    camera.role = cam.get("role", "")
    camera.start()
    return camera


def apply_camera_metadata(camera, cam_config):
    camera.label = cam_config.get("label") or cam_config.get("id")
    camera.zone = cam_config.get("zone", "")
    camera.role = cam_config.get("role", "")


def sync_runtime_cameras(active_cameras, active_specs, ordered_ids, config, runtime, registry_path, registry_lock):
    with registry_lock:
        _, registry_cameras = load_camera_registry(config, registry_path)

    desired_cameras = [cam for cam in registry_cameras if cam.get("enabled", True)]
    desired_by_id = {cam["id"]: cam for cam in desired_cameras}
    desired_order = [cam["id"] for cam in desired_cameras]

    for cam_id in list(active_cameras.keys()):
        if cam_id not in desired_by_id:
            active_cameras[cam_id].stop()
            active_cameras.pop(cam_id, None)
            active_specs.pop(cam_id, None)

    for cam_id, cam in desired_by_id.items():
        signature = camera_runtime_signature(cam, runtime["camera_profile"])
        if cam_id not in active_cameras:
            camera = build_threaded_camera(cam, runtime)
            if camera is not None:
                apply_camera_metadata(camera, cam)
                active_cameras[cam_id] = camera
                active_specs[cam_id] = signature
            continue
        if active_specs.get(cam_id) != signature:
            active_cameras[cam_id].stop()
            camera = build_threaded_camera(cam, runtime)
            if camera is not None:
                apply_camera_metadata(camera, cam)
                active_cameras[cam_id] = camera
                active_specs[cam_id] = signature
        else:
            apply_camera_metadata(active_cameras[cam_id], cam)

    ordered_ids[:] = [cam_id for cam_id in desired_order if cam_id in active_cameras]

def main():
    args = parse_args()
    config = load_config(args.config)
    runtime = get_runtime_config(config, args)
    registry_path = runtime["camera_registry_file"]
    registry_lock = threading.Lock()
    ensure_camera_registry(config, registry_path)
    clear_startup_outputs(runtime)
    runtime = apply_performance_profile(runtime)
    devices = select_devices(args.device)
    device = devices[0]
    device_desc = ",".join(devices)
    acceleration = get_acceleration_info(args.device, device)
    decoder_log_handle = None
    if runtime["redirect_decoder_logs"]:
        decoder_log_handle = redirect_decoder_stderr(runtime["decoder_log_file"])

    employee_df = pd.read_csv("data/employees.csv")
    employee_map = {
        int(r.get("opencv_marker_id", r["marker_id"])): {
            "id": int(r["marker_id"]),
            "opencv_marker_id": int(r.get("opencv_marker_id", r["marker_id"])),
            "name": r["name"],
            "dept": r.get("department", "Production"),
            "active": str(r.get("active", "yes")).strip().lower() != "no",
        }
        for _, r in employee_df.iterrows()
    }

    inference_workers = load_models_for_devices(config["detection"]["model_path"], devices)
    inference_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=len(inference_workers))
        if len(inference_workers) > 1
        else None
    )
    if acceleration["cuda_warning"]:
        print(acceleration["cuda_warning"])
    print(
        f"Runtime: device={device_desc}, profile={runtime['performance_profile']}, batch={runtime['batch_size']}, "
        f"inference_width={runtime['inference_width']}, aruco_every={runtime['aruco_every_n_frames']}, "
        f"dashboard={runtime['dashboard_image_width']}px/{runtime['dashboard_fps']}fps, "
        f"tracker={'ByteTrack' if sv is not None and hasattr(sv, 'ByteTrack') else 'local'}+ReID"
    )
    if runtime["redirect_decoder_logs"]:
        print(f"Decoder warnings redirected to {runtime['decoder_log_file']}")

    active_cameras = {}
    active_specs = {}
    camera_order = []
    sync_runtime_cameras(active_cameras, active_specs, camera_order, config, runtime, registry_path, registry_lock)

    history = {}
    marker_confirmations = {}
    pending_track_switches = {}
    last_render_frames = {}
    last_aruco_frame_index = {}
    last_dashboard_save_ts = {}
    pending_live_writes = {}
    loop_counter = 0
    os.makedirs("logs", exist_ok=True)
    os.makedirs(runtime["log_dir"], exist_ok=True)
    os.makedirs("outputs/live", exist_ok=True)
    os.makedirs(runtime["recording_dir"], exist_ok=True)
    os.makedirs(runtime["report_dir"], exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["aruco"]["dictionary_type"]))
    threading.Thread(target=start_server, args=(registry_path, registry_lock), daemon=True).start()
    time.sleep(1)

    session_start = time.time()
    scale = config["mapping"].get("scale_factor", 50.0)
    floor_model = load_floor_map_model()
    with registry_lock:
        _, registry_cameras = load_camera_registry(config, registry_path)
    distance_calibrations = build_distance_calibrations(registry_cameras, floor_model)
    distance_methods = {
        cam_id: info.get("method", "legacy_video_pixels")
        for cam_id, info in distance_calibrations.items()
    }
    print(f"Distance methods: {distance_methods}")
    monitor = PerformanceMonitor(runtime["monitor_interval_seconds"])
    loop_interval = 1.0 / max(float(runtime["target_fps"]), 1.0)
    confidence = config["detection"].get("confidence_threshold", 0.25)
    classes = config["detection"].get("classes", [0])
    person_tracker = ByteTrackStyleTracker(runtime)

    aruco_pool = concurrent.futures.ThreadPoolExecutor(max_workers=runtime["aruco_workers"])
    image_writer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=runtime["image_writer_workers"])
    path_video_recorder = VideoRecorder(
        runtime["path_video_fps"],
        runtime["path_video_codec"],
        runtime["path_video_min_frames"],
    )
    last_camera_sync_ts = 0.0

    try:
        while True:
            loop_counter += 1
            loop_started = time.time()
            try:
                now_ts = time.time()
                if now_ts - last_camera_sync_ts >= 3.0:
                    sync_runtime_cameras(active_cameras, active_specs, camera_order, config, runtime, registry_path, registry_lock)
                    last_camera_sync_ts = now_ts
                cameras = [active_cameras[cam_id] for cam_id in camera_order if cam_id in active_cameras]
                if not cameras:
                    time.sleep(0.5)
                    continue
                break_msg = get_break_info()
                frame_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "uptime": int(now_ts - session_start),
                    "cameras": [],
                    "total_person_count": 0,
                    "stats": [],
                    "performance": monitor.sample(),
                    "runtime": {
                        "camera_profile": runtime["camera_profile"],
                        "replay_speed": runtime["replay_speed"],
                        "inference_width": runtime["inference_width"],
                        "aruco_max_width": runtime["aruco_max_width"],
                        "aruco_every_n_frames": runtime["aruco_every_n_frames"],
                        "aruco_crop_first_enabled": runtime["aruco_crop_first_enabled"],
                        "aruco_crop_zoom": runtime["aruco_crop_zoom"],
                        "aruco_workers": runtime["aruco_workers"],
                        "dashboard_image_width": runtime["dashboard_image_width"],
                        "dashboard_fps": runtime["dashboard_fps"],
                        "batch_size": runtime["batch_size"],
                        "device": device_desc,
                        "performance_profile": runtime["performance_profile"],
                        "target_fps": runtime["target_fps"],
                        "path_video_every_n_frames": runtime["path_video_every_n_frames"],
                        "path_video_fps": runtime["path_video_fps"],
                        "distance_methods": distance_methods,
                        "marker_confirmations_required": runtime["marker_confirmations_required"],
                        "marker_stale_after_seconds": runtime["marker_stale_after_seconds"],
                        "marker_hidden_grace_seconds": runtime["marker_hidden_grace_seconds"],
                        "marker_hidden_match_radius_px": runtime["marker_hidden_match_radius_px"],
                        "tracker_max_age_seconds": runtime["tracker_max_age_seconds"],
                        "reid_match_threshold": runtime["reid_match_threshold"],
                        "activity_idle_after_seconds": runtime["activity_idle_after_seconds"],
                    },
                    "acceleration": acceleration,
                    "lunch_mode": break_msg is not None,
                    "break_message": break_msg or "",
                }

                prepared_frames = []
                aruco_futures = {}
                for cam in cameras:
                    frame, read_ts = cam.latest()
                    if frame is None:
                        continue

                    render_frame = frame.copy()
                    proc_w = int(runtime["inference_width"])
                    h, w = frame.shape[:2]
                    yolo_scale = w / proc_w
                    proc_frame = cv2.resize(frame, (proc_w, int(h * (proc_w / w))))
                    last_render_frames[cam.id] = render_frame
                    stream_status = cam.status()
                    frame_index = stream_status.get("frames_read", 0)
                    item = {
                        "id": cam.id,
                        "frame": proc_frame,
                        "render_frame": render_frame,
                        "yolo_scale": yolo_scale,
                        "frame_index": frame_index,
                        "read_age_ms": int((now_ts - read_ts) * 1000) if read_ts else None,
                        "stream_status": stream_status,
                    }
                    prepared_frames.append(item)

                    last_index = last_aruco_frame_index.get(cam.id, 0)
                    if (
                        (not runtime["aruco_crop_first_enabled"] or runtime["aruco_full_frame_fallback"])
                        and frame_index - last_index >= runtime["aruco_every_n_frames"]
                    ):
                        last_aruco_frame_index[cam.id] = frame_index
                        aruco_futures[cam.id] = {
                            "frame_index": frame_index,
                            "future": aruco_pool.submit(
                                detect_aruco_markers,
                                aruco_dict,
                                render_frame,
                                runtime["aruco_max_width"],
                            ),
                        }

                total_detected_people = 0
                inference_tasks = []
                for batch_index, frame_batch in enumerate(chunked(prepared_frames, int(runtime["batch_size"]))):
                    if not frame_batch:
                        continue
                    worker = inference_workers[batch_index % len(inference_workers)]
                    frames = [item["frame"] for item in frame_batch]
                    if inference_executor is not None:
                        task = inference_executor.submit(run_yolo_inference, worker, frames, confidence, classes)
                        inference_tasks.append((frame_batch, task, None))
                    else:
                        results_batch = run_yolo_inference(worker, frames, confidence, classes)
                        inference_tasks.append((frame_batch, None, results_batch))

                for frame_batch, task, direct_results in inference_tasks:
                    results_batch = direct_results if task is None else task.result()
                    for item, results in zip(frame_batch, results_batch):
                        cam_id = item["id"]
                        render_frame = item["render_frame"]
                        time_scale = camera_time_scale(item.get("stream_status"), runtime)
                        person_boxes = scale_person_boxes(results, item["yolo_scale"])
                        person_boxes = person_tracker.update(cam_id, person_boxes, render_frame, now_ts)
                        used_person_box_ids = set()
                        marker_seen_workers = set()

                        person_count = len(results.boxes)
                        total_detected_people += person_count
                        marker_person_matches = {}
                        if runtime["aruco_crop_first_enabled"]:
                            corners, ids, marker_person_matches = detect_aruco_in_person_crops(
                                aruco_dict,
                                render_frame,
                                person_boxes,
                                runtime["aruco_crop_zoom"],
                                runtime["aruco_crop_margin_ratio"],
                                runtime["min_marker_area"],
                            )
                        else:
                            corners, ids = None, None

                        full_corners, full_ids = None, None
                        aruco_future_info = aruco_futures.get(cam_id)
                        if (
                            aruco_future_info is not None
                            and aruco_future_info.get("frame_index") == item.get("frame_index")
                        ):
                            try:
                                full_corners, full_ids = aruco_future_info["future"].result(timeout=1.0)
                            except concurrent.futures.TimeoutError:
                                full_corners, full_ids = None, None
                        corners, ids, marker_person_matches = merge_aruco_detections(
                            corners,
                            ids,
                            marker_person_matches,
                            full_corners,
                            full_ids,
                        )

                        if ids is not None:
                            for i, marker_id in enumerate(ids.flatten()):
                                marker_id = int(marker_id)
                                emp = employee_map.get(marker_id)
                                if not emp:
                                    continue

                                marker_corners = corners[i][0]
                                if marker_area(marker_corners) < runtime["min_marker_area"]:
                                    continue

                                raw_pos = marker_corners.mean(axis=0)
                                matched_person = marker_person_matches.get(i)
                                if matched_person is None:
                                    matched_person = find_person_for_marker(
                                        raw_pos,
                                        person_boxes,
                                        runtime["person_box_margin"],
                                    )
                                if matched_person is None:
                                    marker_confirmations.pop((cam_id, marker_id), None)
                                    continue
                                existing_state = history.get(marker_id)
                                matched_track_id = matched_person.get("track_id")
                                if existing_state and existing_state.get("cam_id") == cam_id:
                                    current_track_id = existing_state.get("track_id")
                                    if current_track_id and matched_track_id and current_track_id != matched_track_id:
                                        current_track_visible = any(
                                            person.get("track_id") == current_track_id
                                            for person in person_boxes
                                        )
                                        switch_key = (cam_id, marker_id, matched_track_id)
                                        if current_track_visible:
                                            pending_track_switches.pop(switch_key, None)
                                            continue
                                        pending_track_switches[switch_key] = pending_track_switches.get(switch_key, 0) + 1
                                        if pending_track_switches[switch_key] < max(3, runtime["marker_confirmations_required"] + 1):
                                            continue
                                        last_bbox = existing_state.get("last_bbox")
                                        if last_bbox is not None:
                                            switch_distance = float(np.linalg.norm(
                                                bbox_center(last_bbox) - bbox_center(matched_person["bbox"])
                                            ))
                                            if switch_distance > runtime["marker_hidden_match_radius_px"]:
                                                continue
                                try:
                                    used_person_box_ids.add(person_boxes.index(matched_person))
                                except ValueError:
                                    pass
                                if matched_person.get("track_id") is not None:
                                    person_tracker.bind_employee(cam_id, matched_person["track_id"], marker_id)
                                    matched_person["employee_marker_id"] = marker_id
                                    matched_person["global_id"] = f"emp_{marker_id}"

                                confirmation_key = (cam_id, marker_id)
                                marker_confirmations[confirmation_key] = marker_confirmations.get(confirmation_key, 0) + 1
                                if (
                                    marker_id not in history
                                    and marker_confirmations[confirmation_key] < runtime["marker_confirmations_required"]
                                ):
                                    continue

                                x1f, y1f, x2f, y2f = matched_person["bbox"]
                                floor_pos = np.array([(x1f + x2f) / 2.0, y2f], dtype=np.float32)
                                calibration = distance_calibrations.get(cam_id, {})
                                map_pos = frame_point_to_map(floor_pos, render_frame.shape, calibration)
                                worker_zone = resolve_worker_zone(map_pos, cam_id, floor_model)

                                if marker_id not in history:
                                    history[marker_id] = {
                                        "display_marker_id": emp.get("id", marker_id),
                                        "opencv_marker_id": marker_id,
                                        "name": emp["name"],
                                        "dept": emp["dept"],
                                        "pos_buffer": deque(maxlen=5),
                                        "path": [floor_pos.tolist()],
                                        "camera_paths": {cam_id: [floor_pos.tolist()]},
                                        "map_pos_buffer": deque(maxlen=5),
                                        "last_map_pos": map_pos.tolist() if map_pos is not None else None,
                                        "dist": 0.0,
                                        "distance_method": calibration.get("method", "legacy_video_pixels"),
                                        "status": "WORKING",
                                        "last_move_ts": now_ts,
                                        "first_seen_ts": now_ts,
                                        "last_seen_ts": now_ts,
                                        "last_marker_seen_ts": now_ts,
                                        "last_account_ts": now_ts,
                                        "person_conf": matched_person["confidence"],
                                        "camera_times": {},
                                        "status_times": {"WALKING": 0.0, "WORKING": 0.0},
                                        "path_view": recording_view_path(emp.get("id", marker_id), emp["name"], cam_id),
                                        "cam_id": cam_id,
                                        "track_id": matched_person.get("track_id"),
                                        "last_bbox": matched_person["bbox"],
                                        "last_feature": matched_person.get("feature"),
                                        "last_visual_frame": -1,
                                        "tracking_mode": "aruco",
                                        "last_work_motion_ts": now_ts,
                                        "activity_reason": "initial_marker_match",
                                        "posture": person_posture(matched_person),
                                        "zone_id": worker_zone.get("id") if worker_zone else "unknown",
                                        "zone_label": worker_zone.get("label") if worker_zone else "Unknown Zone",
                                    }

                                worker_state = history[marker_id]
                                marker_seen_workers.add(marker_id)
                                if worker_state["cam_id"] != cam_id:
                                    # Keep total distance cumulative, but start a fresh visual path per camera.
                                    worker_state["cam_id"] = cam_id
                                    worker_state["track_id"] = matched_person.get("track_id")
                                    worker_state["pos_buffer"].clear()
                                    worker_state.setdefault("camera_paths", {})
                                    worker_state["camera_paths"].setdefault(cam_id, [floor_pos.tolist()])
                                    worker_state["path"] = worker_state["camera_paths"][cam_id]
                                    worker_state["last_map_pos"] = map_pos.tolist() if map_pos is not None else None
                                    worker_state.setdefault("map_pos_buffer", deque(maxlen=5)).clear()
                                    worker_state["last_move_ts"] = now_ts
                                    worker_state["last_seen_ts"] = now_ts
                                    worker_state["last_marker_seen_ts"] = now_ts
                                    worker_state["person_conf"] = matched_person["confidence"]
                                    worker_state["status"] = "WORKING"
                                    worker_state["distance_method"] = calibration.get("method", "legacy_video_pixels")
                                    worker_state["path_view"] = recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"], cam_id)
                                    worker_state["last_bbox"] = matched_person["bbox"]
                                    if matched_person.get("feature") is not None:
                                        worker_state["last_feature"] = matched_person.get("feature")
                                    worker_state["tracking_mode"] = "aruco"
                                    worker_state["last_visual_frame"] = loop_counter
                                    worker_state["zone_id"] = worker_zone.get("id") if worker_zone else "unknown"
                                    worker_state["zone_label"] = worker_zone.get("label") if worker_zone else "Unknown Zone"
                                    classify_worker_activity(worker_state, matched_person, render_frame, 0.0, now_ts, runtime)
                                    continue

                                worker_state["last_seen_ts"] = now_ts
                                worker_state["last_marker_seen_ts"] = now_ts
                                worker_state["person_conf"] = matched_person["confidence"]
                                worker_state["track_id"] = matched_person.get("track_id")
                                worker_state["path_view"] = recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"], cam_id)
                                worker_state["last_bbox"] = matched_person["bbox"]
                                if matched_person.get("feature") is not None:
                                    worker_state["last_feature"] = matched_person.get("feature")
                                worker_state["tracking_mode"] = "aruco"
                                worker_state["last_visual_frame"] = loop_counter
                                worker_state["zone_id"] = worker_zone.get("id") if worker_zone else "unknown"
                                worker_state["zone_label"] = worker_zone.get("label") if worker_zone else "Unknown Zone"
                                worker_state.setdefault("camera_paths", {}).setdefault(cam_id, worker_state.get("path", [floor_pos.tolist()]))
                                worker_state["path"] = worker_state["camera_paths"][cam_id]
                                worker_state["pos_buffer"].append(floor_pos)
                                smooth_pos = np.mean(worker_state["pos_buffer"], axis=0)
                                last_pt = np.array(worker_state["path"][-1])
                                d_pixels = float(np.sqrt(np.sum((smooth_pos - last_pt) ** 2)))
                                if map_pos is not None:
                                    worker_state.setdefault("map_pos_buffer", deque(maxlen=5)).append(map_pos)
                                    smooth_map_pos = np.mean(worker_state["map_pos_buffer"], axis=0)
                                else:
                                    smooth_map_pos = None

                                if d_pixels > 8:
                                    distance_prev = worker_state.get("last_map_pos") if smooth_map_pos is not None else last_pt
                                    distance_next = smooth_map_pos if smooth_map_pos is not None else smooth_pos
                                    dist_feet, distance_method = distance_feet_between_points(
                                        distance_prev,
                                        distance_next,
                                        calibration,
                                        scale,
                                    )
                                    if not accept_path_step(worker_state, dist_feet, d_pixels, "aruco"):
                                        worker_state["pos_buffer"].clear()
                                        worker_state.setdefault("map_pos_buffer", deque(maxlen=5)).clear()
                                        continue

                                    worker_state["dist"] += dist_feet
                                    worker_state["last_map_pos"] = smooth_map_pos.tolist() if smooth_map_pos is not None else None
                                    worker_state["distance_method"] = distance_method
                                    worker_state["path"].append(smooth_pos.tolist())
                                    worker_state["camera_paths"][cam_id] = worker_state["path"]
                                    worker_state["cam_id"] = cam_id
                                classify_worker_activity(worker_state, matched_person, render_frame, d_pixels, now_ts, runtime)

                                pts = marker_corners.astype(int)
                                x1, y1, x2, y2 = [int(v) for v in matched_person["bbox"]]
                                cv2.rectangle(render_frame, (x1, y1), (x2, y2), (56, 189, 248), 2)
                                cv2.polylines(render_frame, [pts], True, (34, 197, 94), 2)
                                cv2.putText(
                                    render_frame,
                                    emp["name"],
                                    (pts[0][0], pts[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )

                        for marker_id, worker_state in history.items():
                            if worker_state.get("cam_id") != cam_id or marker_id in marker_seen_workers:
                                continue
                            marker_age = (now_ts - worker_state.get("last_marker_seen_ts", 0.0)) * time_scale
                            matched_person = person_tracker.find_employee_detection(
                                cam_id,
                                marker_id,
                                person_boxes,
                                used_person_box_ids,
                            )
                            tracking_mode = "byte_track_reid"
                            if matched_person is None:
                                if marker_age > runtime["marker_hidden_max_seconds"]:
                                    continue
                                continuation_radius = runtime["marker_hidden_match_radius_px"] + min(180.0, marker_age * 10.0)
                                matched_person = find_continuation_person(
                                    worker_state,
                                    person_boxes,
                                    used_person_box_ids,
                                    continuation_radius,
                                    runtime["marker_hidden_min_iou"],
                                )
                                tracking_mode = "marker_hidden"
                            if matched_person is None:
                                continue

                            if matched_person.get("track_id") is not None:
                                person_tracker.bind_employee(cam_id, matched_person["track_id"], marker_id)
                                matched_person["employee_marker_id"] = marker_id
                                matched_person["global_id"] = f"emp_{marker_id}"

                            floor_pos = person_floor_point(matched_person)
                            calibration = distance_calibrations.get(cam_id, {})
                            map_pos = frame_point_to_map(floor_pos, render_frame.shape, calibration)
                            worker_zone = resolve_worker_zone(map_pos, cam_id, floor_model)

                            worker_state["last_seen_ts"] = now_ts
                            worker_state["person_conf"] = min(worker_state.get("person_conf", 0.0), matched_person["confidence"])
                            worker_state["track_id"] = matched_person.get("track_id")
                            worker_state["path_view"] = recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"], cam_id)
                            worker_state["last_bbox"] = matched_person["bbox"]
                            if matched_person.get("feature") is not None:
                                worker_state["last_feature"] = matched_person.get("feature")
                            worker_state["tracking_mode"] = tracking_mode
                            worker_state["last_visual_frame"] = loop_counter
                            worker_state["zone_id"] = worker_zone.get("id") if worker_zone else "unknown"
                            worker_state["zone_label"] = worker_zone.get("label") if worker_zone else "Unknown Zone"
                            worker_state.setdefault("camera_paths", {}).setdefault(cam_id, worker_state.get("path", [floor_pos.tolist()]))
                            worker_state["path"] = worker_state["camera_paths"][cam_id]
                            worker_state["pos_buffer"].append(floor_pos)
                            smooth_pos = np.mean(worker_state["pos_buffer"], axis=0)
                            last_pt = np.array(worker_state["path"][-1])
                            d_pixels = float(np.linalg.norm(smooth_pos - last_pt))

                            if map_pos is not None:
                                worker_state.setdefault("map_pos_buffer", deque(maxlen=5)).append(map_pos)
                                smooth_map_pos = np.mean(worker_state["map_pos_buffer"], axis=0)
                            else:
                                smooth_map_pos = None

                            if d_pixels > 8:
                                distance_prev = worker_state.get("last_map_pos") if smooth_map_pos is not None else last_pt
                                distance_next = smooth_map_pos if smooth_map_pos is not None else smooth_pos
                                dist_feet, distance_method = distance_feet_between_points(
                                    distance_prev,
                                    distance_next,
                                    calibration,
                                    scale,
                                )
                                if accept_path_step(worker_state, dist_feet, d_pixels, tracking_mode):
                                    worker_state["dist"] += dist_feet
                                    worker_state["last_map_pos"] = smooth_map_pos.tolist() if smooth_map_pos is not None else None
                                    worker_state["distance_method"] = distance_method
                                    worker_state["path"].append(smooth_pos.tolist())
                                    worker_state["camera_paths"][cam_id] = worker_state["path"]
                                else:
                                    worker_state["pos_buffer"].clear()
                                    worker_state.setdefault("map_pos_buffer", deque(maxlen=5)).clear()
                            classify_worker_activity(worker_state, matched_person, render_frame, d_pixels, now_ts, runtime)

                            x1, y1, x2, y2 = [int(v) for v in matched_person["bbox"]]
                            cv2.rectangle(render_frame, (x1, y1), (x2, y2), (0, 214, 255), 2)
                            cv2.putText(
                                render_frame,
                                f"{worker_state['name']} | {tracking_mode}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 255, 255),
                                2,
                            )

                        for idx, person in enumerate(person_boxes):
                            if idx in used_person_box_ids:
                                continue
                            if likely_recent_tracked_person(
                                person,
                                history.values(),
                                cam_id,
                                now_ts,
                                runtime["marker_hidden_max_seconds"],
                                runtime["marker_hidden_match_radius_px"],
                            ):
                                continue
                            x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
                            cv2.rectangle(render_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            track_label = person.get("global_id") or f"track_{person.get('track_id', '?')}"
                            cv2.putText(
                                render_frame,
                                f"UNKNOWN {track_label} | no marker {person['confidence']:.2f}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2,
                            )

                        live_path = f"outputs/live/{cam_id}.jpg"
                        save_interval = 1.0 / runtime["dashboard_fps"]
                        pending_live = pending_live_writes.get(cam_id)
                        if (
                            now_ts - last_dashboard_save_ts.get(cam_id, 0.0) >= save_interval
                            and (pending_live is None or pending_live.done())
                        ):
                            dashboard_frame, _ = resize_to_width(render_frame, runtime["dashboard_image_width"])
                            pending_live_writes[cam_id] = image_writer_pool.submit(
                                write_jpeg_atomic,
                                live_path,
                                dashboard_frame.copy(),
                                runtime["jpeg_quality_live"],
                            )
                            last_dashboard_save_ts[cam_id] = now_ts

                        frame_data["cameras"].append({
                            "id": cam_id,
                            "label": getattr(active_cameras.get(cam_id), "label", cam_id),
                            "zone": getattr(active_cameras.get(cam_id), "zone", ""),
                            "role": getattr(active_cameras.get(cam_id), "role", ""),
                            "person_count": person_count,
                            "live_view": live_path,
                            "view_ts": round(last_dashboard_save_ts.get(cam_id, 0.0), 3),
                            "read_age_ms": item["read_age_ms"],
                            "stream_status": item["stream_status"],
                        })

                reported_camera_ids = {camera_info["id"] for camera_info in frame_data["cameras"]}
                for cam in cameras:
                    if cam.id not in reported_camera_ids:
                        frame_data["cameras"].append({
                            "id": cam.id,
                            "label": getattr(cam, "label", cam.id),
                            "zone": getattr(cam, "zone", ""),
                            "role": getattr(cam, "role", ""),
                            "person_count": 0,
                            "live_view": f"outputs/live/{cam.id}.jpg",
                            "view_ts": round(last_dashboard_save_ts.get(cam.id, 0.0), 3),
                            "read_age_ms": None,
                            "stream_status": cam.status(),
                        })

                employee_records = []
                for mid, state in history.items():
                    account_worker_time(state, now_ts, runtime["marker_stale_after_seconds"])
                    current_cam = state.get("cam_id", "unknown")
                    visible_this_loop = state.get("last_visual_frame") == loop_counter
                    state["visible_now"] = visible_this_loop
                    state["path"] = state.get("camera_paths", {}).get(current_cam, state.get("path", []))
                    display_marker_id = state.get("display_marker_id", mid)
                    state["path_view"] = recording_view_path(display_marker_id, state["name"], current_cam)
                    employee_record = build_employee_record(mid, state, now_ts)
                    employee_records.append(employee_record)

                    if visible_this_loop and loop_counter % runtime["path_video_every_n_frames"] == 0:
                        target_frame = last_render_frames.get(current_cam)
                    else:
                        target_frame = None

                    if target_frame is not None:
                        path_frame = target_frame.copy()
                        if state["path"]:
                            pts = np.array(state["path"], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(path_frame, [pts], False, (0, 255, 255), 3)
                            cv2.circle(path_frame, tuple(pts[-1][0]), 8, (0, 0, 255), -1)
                        cv2.putText(
                            path_frame,
                            f"{state['name']} | {current_cam} | {state['status']} | {state.get('tracking_mode', 'unknown')} | {state['dist']:.2f} ft",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
                        path_frame, _ = resize_to_width(path_frame, runtime["dashboard_image_width"])
                        path_video_recorder.write(recording_disk_path(display_marker_id, state["name"], current_cam), path_frame)

                    frame_data["stats"].append(employee_record)

                frame_data["total_person_count"] = total_detected_people
                with open(runtime["employee_records_file"], "w") as f:
                    json.dump({
                        "timestamp": frame_data["timestamp"],
                        "employees": employee_records,
                    }, f, indent=2)
                for live_stats_path in {runtime["live_stats_file"], runtime["dashboard_live_stats_file"]}:
                    with open(live_stats_path, "w") as f:
                        json.dump(frame_data, f, indent=2)

                elapsed = time.time() - loop_started
                time.sleep(max(0.0, loop_interval - elapsed))

            except Exception as e:
                print(f"INNER LOOP ERROR: {e}")
                traceback.print_exc()
                time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        path_video_recorder.close()
        export_employee_report_on_shutdown()
        aruco_pool.shutdown(wait=False, cancel_futures=True)
        image_writer_pool.shutdown(wait=False, cancel_futures=True)
        for cam in active_cameras.values():
            cam.stop()
        if decoder_log_handle is not None:
            decoder_log_handle.close()

if __name__ == "__main__":
    main()
