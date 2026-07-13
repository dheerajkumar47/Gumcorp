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
import csv
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
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        print(f"Camera registry reload skipped: {exc}")
        fallback = {"version": 1, "updated_at": "", "cameras": config.get("cameras", [])}
        payload = fallback
    cameras = [normalize_camera_entry(cam) for cam in payload.get("cameras", [])]
    return payload, cameras


def write_camera_registry(registry_path, cameras):
    # Preserve calibration_points from the existing registry — the dashboard
    # does not include them in its PUT payload, so they would be silently wiped
    # on every camera toggle without this merge.
    existing_calibration = {}
    try:
        existing_text = Path(registry_path).read_text(encoding="utf-8-sig")
        existing_payload = json.loads(existing_text)
        for cam in existing_payload.get("cameras", []):
            cam_id = str(cam.get("id", "")).strip()
            cal = cam.get("calibration_points")
            if cam_id and cal and (cal.get("camera_points") or cal.get("map_points")):
                existing_calibration[cam_id] = cal
    except Exception:
        pass

    normalized = []
    for cam in cameras:
        entry = normalize_camera_entry(cam)
        cam_id = entry["id"]
        cal = entry.get("calibration_points", {})
        # Only restore from existing if the incoming entry has no points
        if cam_id in existing_calibration and not cal.get("camera_points") and not cal.get("map_points"):
            entry["calibration_points"] = existing_calibration[cam_id]
        normalized.append(entry)

    payload = {
        "version": 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": normalized,
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
        "path_video_segment_seconds": max(10.0, float(runtime.get("path_video_segment_seconds", 60.0))),
        "path_video_codec": str(runtime.get("path_video_codec", "mp4v")),
        "redirect_decoder_logs": bool(runtime.get("redirect_decoder_logs", True)),
        "log_dir": log_dir,
        "recording_dir": recording_dir,
        "report_dir": report_dir,
        "live_stats_file": os.path.join(log_dir, "live_stats.json"),
        "dashboard_live_stats_file": "logs/live_stats.json",
        "decoder_log_file": os.path.join(log_dir, "decoder_ffmpeg.log"),
        "production_watchdog_timeout_seconds": float(runtime.get("production_watchdog_timeout_seconds", 120.0)),
        "production_watchdog_check_seconds": float(runtime.get("production_watchdog_check_seconds", 10.0)),
        "monitor_interval_seconds": runtime.get("monitor_interval_seconds", 2.0),
        "jpeg_quality_live": runtime.get("jpeg_quality_live", 75),
        "reconnect_after": runtime.get("reconnect_after", 10),
        "reconnect_interval_seconds": float(runtime.get("reconnect_interval_seconds", 5.0)),
        "stale_camera_after_seconds": float(runtime.get("stale_camera_after_seconds", 3.0)),
        "marker_confirmations_required": max(1, int(runtime.get("marker_confirmations_required", 2))),
        "unassigned_marker_confirmations_required": max(1, int(runtime.get("unassigned_marker_confirmations_required", 10))),
        "marker_instant_lock_score": float(runtime.get("marker_instant_lock_score", 90000.0)),
        "marker_stale_after_seconds": float(runtime.get("marker_stale_after_seconds", 8.0)),
        "marker_hidden_grace_seconds": float(runtime.get("marker_hidden_grace_seconds", 12.0)),
        "marker_hidden_match_radius_px": float(runtime.get("marker_hidden_match_radius_px", 180.0)),
        "marker_hidden_max_seconds": float(runtime.get("marker_hidden_max_seconds", 4.0)),
        "marker_hidden_min_iou": float(runtime.get("marker_hidden_min_iou", 0.08)),
        "tracker_confidence_threshold": float(runtime.get("tracker_confidence_threshold", 0.16)),
        "display_confidence_threshold": float(runtime.get("display_confidence_threshold", config.get("detection", {}).get("confidence_threshold", 0.25))),
        "track_binding_ttl_seconds": float(runtime.get("track_binding_ttl_seconds", 12.0)),
        "identity_continuation_max_seconds": float(runtime.get("identity_continuation_max_seconds", 20.0)),
        "identity_continuation_min_confidence": float(runtime.get("identity_continuation_min_confidence", 0.35)),
        "single_person_continuation_max_seconds": float(runtime.get("single_person_continuation_max_seconds", 180.0)),
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
        "assigned_min_marker_area": float(runtime.get("assigned_min_marker_area", runtime.get("min_marker_area", 80.0))),
        "unassigned_min_marker_area": float(runtime.get("unassigned_min_marker_area", max(float(runtime.get("min_marker_area", 80.0)), 120.0))),
        "ignored_opencv_marker_ids": {int(marker_id) for marker_id in runtime.get("ignored_opencv_marker_ids", [])},
        "employee_records_file": os.path.join(log_dir, "employee_records.json"),
        "camera_registry_file": args.camera_registry or runtime.get("camera_registry_file", "data/cameras.json"),
        "debug_mode": bool(config.get("system", {}).get("debug", False)),
        "performance_profile": args.performance_profile or runtime.get("performance_profile", "balanced"),
        "tracker_backend": str(runtime.get("tracker_backend", "botsort")).strip().lower(),
        "botsort_tracker_yaml": str(runtime.get("botsort_tracker_yaml", "botsort.yaml")),
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
        runtime["aruco_every_n_frames"] = max(2, int(runtime["aruco_every_n_frames"]))
        runtime["aruco_workers"] = max(2, runtime["aruco_workers"])  # allow config value, min 2
        runtime["dashboard_image_width"] = min(runtime["dashboard_image_width"], 1280)
        runtime["dashboard_fps"] = min(runtime["dashboard_fps"], 4.0)
        runtime["jpeg_quality_live"] = max(75, min(runtime["jpeg_quality_live"], 82))
        runtime["stale_camera_after_seconds"] = max(runtime["stale_camera_after_seconds"], 30.0)
        runtime["reconnect_interval_seconds"] = max(runtime["reconnect_interval_seconds"], 3.0)
    elif profile == "speed":
        runtime["inference_width"] = min(runtime["inference_width"], 1536)
        runtime["target_fps"] = min(runtime["target_fps"], 12.0)
        runtime["batch_size"] = max(runtime["batch_size"], 8)
        runtime["aruco_max_width"] = min(runtime["aruco_max_width"], 1280)
        runtime["aruco_crop_zoom"] = max(runtime["aruco_crop_zoom"], 6.0)
        runtime["aruco_every_n_frames"] = max(1, min(2, int(runtime["aruco_every_n_frames"])))
        runtime["aruco_workers"] = max(2, runtime["aruco_workers"])  # allow config value, min 2
        runtime["aruco_full_frame_fallback"] = False
        runtime["dashboard_image_width"] = min(runtime["dashboard_image_width"], 960)
        runtime["dashboard_fps"] = min(runtime["dashboard_fps"], 8)
        runtime["jpeg_quality_live"] = max(65, min(runtime["jpeg_quality_live"], 75))
        runtime["stale_camera_after_seconds"] = max(runtime["stale_camera_after_seconds"], 30.0)
        runtime["reconnect_interval_seconds"] = max(runtime["reconnect_interval_seconds"], 3.0)
    elif profile == "quality":
        # Do NOT bump inference_width — larger images cost GPU without helping ArUco accuracy
        # (ArUco runs on cropped regions, not the full YOLO inference frame)
        runtime["aruco_max_width"] = max(runtime["aruco_max_width"], 2560)
        # Do NOT force a minimum zoom — let caller set zoom via --aruco-zoom
        runtime["aruco_every_n_frames"] = max(2, int(runtime["aruco_every_n_frames"] / 2))
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
    try:
        aruco_frame, aruco_scale = resize_to_width(frame, max_width)
        gray = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detector_params())
        corners, ids = detect_markers_with_variants(detector, gray)
        if ids is not None:
            print(f"[ARUCO FULL-FRAME] Found {len(ids)} markers: {ids.flatten().tolist()}")
        else:
            print("[ARUCO FULL-FRAME] No markers found")
        if corners is not None and aruco_scale != 1.0:
            corners = tuple(corner * aruco_scale for corner in corners)
        return corners, ids
    except Exception as e:
        print(f"[ARUCO FULL-FRAME] Exception: {e}")
        return None, None

def aruco_detector_params():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 10    # bright/white dusty backgrounds
    params.minMarkerPerimeterRate = 0.012  # detect smaller/farther markers
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.08  # accept distorted quads (overhead angle = parallelogram)
    params.minCornerDistanceRate = 0.01        # corners can be close (flattened marker from above)
    params.perspectiveRemoveIgnoredMarginPerCell = 0.05  # tighter margin = better bit read on angled markers
    params.errorCorrectionRate = 0.9       # tolerate 1-bit errors from perspective distortion
    return params

def detect_markers_with_variants(detector, gray):
    # Performance-critical: try variants in order, EXIT EARLY once marker found.
    # Benchmark: raw=160ms, each extra variant=160ms, stretch=200ms on 1200x2400 crop.
    # Early exit means common clean markers cost ~160ms instead of 1250ms (7.8x faster).
    #
    # Variants in priority order (fastest/most-likely first):
    # 1. raw        — clean markers, good lighting (exits here ~80% of time)
    # 2. hist-eq    — white dusty background (exits here most remaining cases)
    # 3. CLAHE      — shadows, uneven light
    # 4. sharpen    — blurry/soft markers
    # 5-7. stretch  — overhead camera foreshortening (only if 1-4 all failed)
    #                 stretch images are capped to 640px to avoid huge CPU cost
    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sharp = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32))

    # Downscale source for stretch variants (cap at 640px max dim)
    MAX_STRETCH_DIM = 640
    sf  = min(1.0, MAX_STRETCH_DIM / max(h, w, 1))
    sg  = cv2.resize(gray,  (max(1, int(w*sf)), max(1, int(h*sf))), interpolation=cv2.INTER_LINEAR) if sf < 1.0 else gray
    ss  = cv2.resize(sharp, (max(1, int(w*sf)), max(1, int(h*sf))), interpolation=cv2.INTER_LINEAR) if sf < 1.0 else sharp
    sh_s, sw_s = sg.shape[:2]

    variants = [
        (gray,                                                                       1.0,          1.0),
        (cv2.equalizeHist(gray),                                                     1.0,          1.0),
        (clahe.apply(gray),                                                          1.0,          1.0),
        (sharp,                                                                      1.0,          1.0),
        (cv2.resize(sg, (sw_s, max(1, sh_s*2)),   interpolation=cv2.INTER_LINEAR),  1.0/sf,       1.0/(2.0*sf)),
        (cv2.resize(sg, (max(1,sw_s*2), sh_s),   interpolation=cv2.INTER_LINEAR),  1.0/(2.0*sf), 1.0/sf),
        (cv2.resize(ss, (sw_s, max(1,int(sh_s*1.6))), interpolation=cv2.INTER_LINEAR), 1.0/sf,   1.0/(1.6*sf)),
    ]

    seen = set()
    merged_corners = []
    merged_ids = []
    for variant_img, sx, sy in variants:
        corners, ids, _ = detector.detectMarkers(variant_img)
        if ids is None:
            continue
        new_found = False
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id in seen:
                continue
            seen.add(marker_id)
            new_found = True
            if sx != 1.0 or sy != 1.0:
                mc = marker_corners.copy().astype(np.float32)
                mc[0, :, 0] *= sx
                mc[0, :, 1] *= sy
                merged_corners.append(mc)
            else:
                merged_corners.append(marker_corners)
            merged_ids.append(marker_id)
        # Early exit: if this variant found new markers, skip remaining variants.
        # Only continue to next variant if this one found nothing new.
        if new_found:
            break

    if not merged_ids:
        return None, None
    return tuple(merged_corners), np.array([[marker_id] for marker_id in merged_ids], dtype=np.int32)

def marker_area_threshold(marker_id, assigned_marker_ids, assigned_min_marker_area, unassigned_min_marker_area):
    return assigned_min_marker_area if int(marker_id) in assigned_marker_ids else unassigned_min_marker_area


def detect_aruco_in_person_crops(
    aruco_dict,
    frame,
    person_boxes,
    zoom_scale,
    margin_ratio,
    min_marker_area,
    debug=False,
    assigned_marker_ids=None,
    assigned_min_marker_area=None,
    unassigned_min_marker_area=None,
    ignored_marker_ids=None,
    pos_hints=None,   # dict: track_id -> rel_y (0..1) of last seen marker within person bbox
):
    if not person_boxes:
        return None, None, {}
    assigned_marker_ids = assigned_marker_ids or set()
    ignored_marker_ids = ignored_marker_ids or set()
    assigned_min_marker_area = min_marker_area if assigned_min_marker_area is None else assigned_min_marker_area
    unassigned_min_marker_area = min_marker_area if unassigned_min_marker_area is None else unassigned_min_marker_area
    height, width = frame.shape[:2]
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detector_params())
    except Exception:
        return None, None, {}
    candidates = []
    crop_detections_log = []

    for person_index, person in enumerate(person_boxes):
        try:
            x1, y1, x2, y2 = person["bbox"]
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            pad_x = box_w * margin_ratio
            pad_y = box_h * margin_ratio
            cx1 = max(0, int(round(x1 - pad_x)))
            cy1 = max(0, int(round(y1 - pad_y)))
            cx2 = min(width, int(round(x2 + pad_x)))
            # Markers on chest/back/elbows — search upper 88% (covers bent/crouching workers).
            cy2 = min(height, int(round(y1 + box_h * 0.88 + pad_y)))
            if cx2 - cx1 < 20 or cy2 - cy1 < 30:
                continue

            # ── ROI hint fast-path ─────────────────────────────────────────
            # If we know where the marker was last frame (rel_y within bbox),
            # try a narrow horizontal strip first — 5x less area to scan.
            track_id = person.get("track_id") or person.get("global_id")
            hint_rel_y = (pos_hints or {}).get(track_id)
            if hint_rel_y is not None:
                full_h = cy2 - cy1
                sy1 = max(cy1, int(cy1 + (hint_rel_y - 0.20) * full_h))
                sy2 = min(cy2, int(cy1 + (hint_rel_y + 0.20) * full_h))
                if sy2 - sy1 > 12 and cx2 - cx1 > 12:
                    strip = frame[sy1:sy2, cx1:cx2]
                    if strip.size > 0:
                        zs = cv2.resize(strip, None, fx=zoom_scale, fy=zoom_scale,
                                        interpolation=cv2.INTER_CUBIC)
                        gs = cv2.cvtColor(zs, cv2.COLOR_BGR2GRAY)
                        sc, si = detect_markers_with_variants(detector, gs)
                        if si is not None:
                            for mid, mc in zip(si.flatten(), sc):
                                if int(mid) in ignored_marker_ids:
                                    continue
                                pts_z = mc[0].astype(np.float32)
                                area = float(cv2.contourArea(pts_z)) / (zoom_scale * zoom_scale)
                                if area < marker_area_threshold(mid, assigned_marker_ids,
                                                                assigned_min_marker_area,
                                                                unassigned_min_marker_area):
                                    continue
                                pts_f = (pts_z / zoom_scale) + np.array([cx1, sy1], dtype=np.float32)
                                if not marker_quality_ok(pts_f, person["bbox"], min_side_px=8.0):
                                    continue
                                mc_xy = pts_f.mean(axis=0)
                                if not expanded_contains(person["bbox"], mc_xy, 0.0):
                                    continue
                                candidates.append({
                                    "marker_id": int(mid), "corners": pts_f,
                                    "person": person, "person_index": person_index,
                                    "score": 100000.0 + area - bbox_area(person["bbox"]) * 0.0001,
                                })
                                crop_detections_log.append(
                                    f"Person {person_index}: hint-strip found marker {int(mid)}")
                            if candidates and any(c["person_index"] == person_index for c in candidates):
                                continue   # found via fast-path — skip full crop
            # ── end ROI hint ───────────────────────────────────────────────

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            zoom = cv2.resize(crop, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)

            # Cap gray to 640px max so ArUco variants don't run on huge zoomed images.
            # At 4x zoom a 150x300px bbox → 600x1200px → cap → ~320x640px (5x fewer pixels).
            # Corners are scaled back to zoom/gray space before the caller maps to full frame.
            _MAX_DIM = 640
            _zh, _zw = gray.shape[:2]
            if max(_zh, _zw) > _MAX_DIM:
                _as = _MAX_DIM / max(_zh, _zw)
                gray_det = cv2.resize(gray, (max(1, int(_zw*_as)), max(1, int(_zh*_as))), interpolation=cv2.INTER_LINEAR)
            else:
                _as = 1.0
                gray_det = gray

            crop_corners, crop_ids = detect_markers_with_variants(detector, gray_det)

            # Scale corners from capped space back to full zoom/gray space
            if crop_ids is not None and _as < 1.0:
                crop_corners = tuple(
                    np.array([mc[0] / _as], dtype=np.float32)
                    for mc in crop_corners
                )

            if crop_ids is None:
                crop_detections_log.append(f"Person {person_index}: no markers")
                continue

            crop_detections_log.append(f"Person {person_index}: found {len(crop_ids)} markers: {crop_ids.flatten().tolist()}")

            for marker_id, marker_corners in zip(crop_ids.flatten(), crop_corners):
                if int(marker_id) in ignored_marker_ids:
                    crop_detections_log.append(f"  Marker {marker_id}: ignored by config")
                    continue
                pts_zoom = marker_corners[0].astype(np.float32)
                area = float(cv2.contourArea(pts_zoom)) / (zoom_scale * zoom_scale)
                area_threshold = marker_area_threshold(
                    marker_id,
                    assigned_marker_ids,
                    assigned_min_marker_area,
                    unassigned_min_marker_area,
                )
                if area < area_threshold:
                    crop_detections_log.append(f"  Marker {marker_id}: area={area:.1f} < threshold={area_threshold} (FILTERED)")
                    continue
                pts_full = (pts_zoom / zoom_scale) + np.array([cx1, cy1], dtype=np.float32)
                if not marker_quality_ok(pts_full, person["bbox"], min_side_px=8.0):
                    crop_detections_log.append(f"  Marker {marker_id}: failed marker geometry/body-position check (FILTERED)")
                    continue
                marker_center = pts_full.mean(axis=0)
                if not expanded_contains(person["bbox"], marker_center, 0.0):
                    crop_detections_log.append(f"  Marker {marker_id}: center outside person box (FILTERED)")
                    continue
                inside_score = 1.0 if expanded_contains(person["bbox"], marker_center, 0.0) else 0.0
                candidates.append({
                    "marker_id": int(marker_id),
                    "corners": pts_full,
                    "person": person,
                    "person_index": person_index,
                    "score": (inside_score * 100000.0) + area - bbox_area(person["bbox"]) * 0.0001,
                })
                crop_detections_log.append(f"  Marker {marker_id}: area={area:.1f} (OK)")
        except Exception as e:
            crop_detections_log.append(f"Person {person_index}: exception {e}")
            continue

    if not candidates:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            full_corners, full_ids = detect_markers_with_variants(detector, gray)
            if full_ids is not None:
                for marker_id, marker_corners in zip(full_ids.flatten(), full_corners):
                    if int(marker_id) in ignored_marker_ids:
                        continue
                    pts_full = marker_corners[0].astype(np.float32)
                    area = marker_area(pts_full)
                    area_threshold = marker_area_threshold(
                        marker_id,
                        assigned_marker_ids,
                        assigned_min_marker_area,
                        unassigned_min_marker_area,
                    )
                    if area < area_threshold:
                        continue
                    marker_center = pts_full.mean(axis=0)
                    matched_person = find_person_for_marker(marker_center, person_boxes, 0.18)
                    if matched_person is None:
                        continue
                    if not expanded_contains(matched_person["bbox"], marker_center, 0.18):
                        continue
                    if not marker_quality_ok(pts_full, matched_person["bbox"], min_side_px=8.0):
                        continue
                    matched_index = next(
                        (idx for idx, person in enumerate(person_boxes) if person is matched_person),
                        -1,
                    )
                    candidates.append({
                        "marker_id": int(marker_id),
                        "corners": pts_full,
                        "person": matched_person,
                        "person_index": matched_index,
                        "score": 90000.0 + area - bbox_area(matched_person["bbox"]) * 0.0001,
                    })
                    crop_detections_log.append(
                        f"Full-frame assist: marker {int(marker_id)} matched inside person box area={area:.1f}"
                    )
        except Exception as e:
            crop_detections_log.append(f"Full-frame assist exception {e}")

    if debug and crop_detections_log:
        print("[ARUCO CROPS] " + " | ".join(crop_detections_log))

    if not candidates:
        return None, None, {}

    # Wide/zoomed person crops can see a neighboring marker. A single person
    # detection must not own multiple employee IDs, otherwise yellow can bind to
    # the wrong worker. Keep the strongest marker per person first, then ensure
    # each marker is used only once.
    best_by_person = {}
    for item in candidates:
        person_key = id(item["person"])
        previous = best_by_person.get(person_key)
        if previous is None or item["score"] > previous["score"]:
            best_by_person[person_key] = item

    best_by_marker = {}
    used_people = set()
    for item in sorted(best_by_person.values(), key=lambda value: value["score"], reverse=True):
        marker_id = item["marker_id"]
        person_key = id(item["person"])
        if person_key in used_people:
            continue
        previous = best_by_marker.get(marker_id)
        if previous is None or item["score"] > previous["score"]:
            best_by_marker[marker_id] = item
            used_people.add(person_key)

    ordered = sorted(best_by_marker.values(), key=lambda item: item["score"], reverse=True)
    corners = tuple(np.array([item["corners"]], dtype=np.float32) for item in ordered)
    ids = np.array([[item["marker_id"]] for item in ordered], dtype=np.int32)
    marker_person_matches = {idx: item["person"] for idx, item in enumerate(ordered)}

    # Update pos_hints with confirmed marker positions for next frame
    if pos_hints is not None:
        for item in ordered:
            tid = item["person"].get("track_id") or item["person"].get("global_id")
            if tid:
                mc_y = float(item["corners"].mean(axis=0)[1])
                by1, by2 = item["person"]["bbox"][1], item["person"]["bbox"][3]
                rel_y = (mc_y - by1) / max(1.0, by2 - by1)
                pos_hints[tid] = float(np.clip(rel_y, 0.05, 0.95))

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


def build_supervision_detections(person_boxes):
    if sv is None:
        return None
    if not person_boxes:
        return sv.Detections(xyxy=np.zeros((0, 4), dtype=np.float32), confidence=np.zeros((0,), dtype=np.float32), class_id=np.zeros((0,), dtype=np.int32))

    xyxy = np.array([person["bbox"] for person in person_boxes], dtype=np.float32)
    confidence = np.array([person.get("confidence", 0.0) for person in person_boxes], dtype=np.float32)
    class_id = np.zeros((len(person_boxes),), dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


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

def marker_quality_ok(corners, person_bbox=None, min_side_px=8.0):
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return False
    sides = np.array([
        np.linalg.norm(pts[(idx + 1) % 4] - pts[idx])
        for idx in range(4)
    ], dtype=np.float32)
    min_side = float(np.min(sides))
    max_side = float(np.max(sides))
    if min_side < min_side_px:
        return False
    if max_side / max(1.0, min_side) > 2.25:
        return False
    area = marker_area(pts)
    square_ratio = area / max(1.0, float(np.mean(sides) ** 2))
    if square_ratio < 0.30 or square_ratio > 1.55:
        return False
    if person_bbox is not None:
        x1, y1, x2, y2 = person_bbox
        marker_center = pts.mean(axis=0)
        person_w = max(1.0, x2 - x1)
        person_h = max(1.0, y2 - y1)
        rel_x = (marker_center[0] - x1) / person_w
        rel_y = (marker_center[1] - y1) / person_h
        if rel_x < -0.08 or rel_x > 1.08:
            return False
        if rel_y < -0.02 or rel_y > 0.93:
            return False
    return True

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
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    upper = upper_body_patch(frame, bbox)
    if upper is not None:
        hist_upper = cv2.calcHist([upper], [0], None, [32], [0, 256])
        hist_upper = cv2.normalize(hist_upper, None).flatten().astype(np.float32)
        patch = cv2.resize(upper, (16, 16), interpolation=cv2.INTER_AREA)
        patch_feat = (patch.flatten().astype(np.float32) / 255.0)
        feature = np.concatenate((hist, hist_upper, patch_feat))
    else:
        feature = hist
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
            sv_detections = build_supervision_detections(detections)
            if sv_detections is not None:
                tracked = byte_tracker.update_with_detections(sv_detections)
                matched_det_indices = set()
            else:
                tracked = None
                matched_det_indices = set()
        else:
            tracked = None
            matched_det_indices = set()

        if tracked is not None:
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
                    if det.get("feature") is not None and track.get("feature") is not None:
                        score += feature_similarity(track.get("feature"), det.get("feature")) * 12.0
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
                return det
        return None


class BoTSORTProductionTracker:
    """Camera-local Ultralytics BoT-SORT backend.

    Ultralytics tracker state is stored on the model/predictor, so production
    uses one model instance per camera. That prevents track IDs from mixing
    between RTSP streams while keeping the rest of the production ArUco/report
    pipeline unchanged.
    """

    def __init__(self, model_path, device, tracker_yaml="botsort.yaml", binding_ttl_seconds=12.0):
        self.model_path = model_path
        self.device = device
        self.tracker_yaml = tracker_yaml
        self.binding_ttl_seconds = float(binding_ttl_seconds)
        self.models_by_camera = {}
        self.employee_bindings = {}

    def _model_for_camera(self, cam_id):
        model = self.models_by_camera.get(cam_id)
        if model is None:
            model = YOLO(self.model_path)
            if hasattr(model, "to"):
                model.to(self.device)
            self.models_by_camera[cam_id] = model
        return model

    def update(self, cam_id, proc_frame, yolo_scale, render_frame, confidence, classes, imgsz):
        model = self._model_for_camera(cam_id)
        result = model.track(
            proc_frame,
            persist=True,
            tracker=self.tracker_yaml,
            conf=confidence,
            iou=0.70,
            classes=classes,
            verbose=False,
            device=self.device,
            imgsz=imgsz,
            half=(self.device != "cpu"),
        )[0]
        detections = scale_tracked_boxes(result, yolo_scale, render_frame)
        bindings = self.employee_bindings.setdefault(cam_id, {})
        now_ts = time.time()
        for track_id in list(bindings):
            entry = bindings[track_id]
            entry_ts = entry.get("ts", 0.0) if isinstance(entry, dict) else 0.0
            if now_ts - entry_ts > self.binding_ttl_seconds:
                bindings.pop(track_id, None)
        for det in detections:
            entry = bindings.get(det.get("track_id"))
            marker_id = entry.get("marker_id") if isinstance(entry, dict) else entry
            if marker_id is not None:
                det["employee_marker_id"] = int(marker_id)
                det["global_id"] = f"emp_{int(marker_id)}"
                if isinstance(entry, dict):
                    entry["ts"] = now_ts
        return detections

    def bind_employee(self, cam_id, track_id, marker_id):
        if track_id is None:
            return
        self.employee_bindings.setdefault(cam_id, {})[track_id] = {
            "marker_id": int(marker_id),
            "ts": time.time(),
        }

    def find_employee_detection(self, cam_id, marker_id, detections, used_box_ids):
        for idx, det in enumerate(detections):
            if idx in used_box_ids:
                continue
            if det.get("employee_marker_id") == int(marker_id):
                return det
        return None

def find_person_for_marker(marker_center, person_boxes, margin_ratio, existing_state=None):
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
    expected_track_id = None
    previous_feature = None
    if existing_state is not None:
        expected_track_id = existing_state.get("track_id")
        previous_feature = existing_state.get("last_feature")

    def person_score(person):
        center_dist = float(np.linalg.norm(bbox_center(person["bbox"]) - marker_point))
        iou = 0.0
        if existing_state is not None and existing_state.get("last_bbox") is not None:
            iou = bbox_iou(existing_state["last_bbox"], person["bbox"])
        feature_score = 0.0
        if previous_feature is not None and person.get("feature") is not None:
            feature_score = feature_similarity(previous_feature, person.get("feature"))
        same_track_bonus = -260.0 if expected_track_id is not None and person.get("track_id") == expected_track_id else 0.0
        inside_bonus = -130.0 if expanded_contains(person["bbox"], marker_point, 0.0) else 0.0
        # When YOLO/BoT-SORT returns duplicate nested boxes around one worker,
        # keep the marker on the stable full-body box instead of a small crop.
        full_body_bonus = -min(130.0, np.sqrt(max(1.0, bbox_area(person["bbox"]))) * 0.22)
        return (
            center_dist * 0.30
            - iou * 170.0
            - feature_score * 240.0
            + person.get("confidence", 0.0) * -90.0
            + inside_bonus
            + same_track_bonus
            + full_body_bonus
        )

    return min(containing, key=person_score)

def synthesize_person_from_marker(marker_corners, frame_shape, existing_state=None):
    """Fallback for bent/occluded workers when YOLO misses but ArUco is visible."""
    pts = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return None
    h, w = frame_shape[:2]
    center = pts.mean(axis=0)
    marker_w = max(
        float(np.linalg.norm(pts[0] - pts[1])),
        float(np.linalg.norm(pts[2] - pts[3])),
        12.0,
    )
    marker_h = max(
        float(np.linalg.norm(pts[1] - pts[2])),
        float(np.linalg.norm(pts[3] - pts[0])),
        12.0,
    )
    marker_size = max(marker_w, marker_h)

    if existing_state and existing_state.get("last_bbox"):
        x1, y1, x2, y2 = existing_state["last_bbox"]
        prev_w = max(40.0, float(x2 - x1))
        prev_h = max(60.0, float(y2 - y1))
        box_w = max(prev_w * 0.75, marker_size * 4.5)
        box_h = max(prev_h * 0.75, marker_size * 5.5)
    else:
        box_w = marker_size * 5.0
        box_h = marker_size * 6.0

    cx, cy = float(center[0]), float(center[1])
    x1 = max(0.0, cx - box_w * 0.5)
    x2 = min(float(w - 1), cx + box_w * 0.5)
    y1 = max(0.0, cy - box_h * 0.45)
    y2 = min(float(h - 1), cy + box_h * 0.65)
    if x2 - x1 < 20 or y2 - y1 < 30:
        return None
    return {
        "bbox": (x1, y1, x2, y2),
        "confidence": 0.99,
        "track_id": existing_state.get("track_id") if existing_state else None,
        "feature": existing_state.get("last_feature") if existing_state else None,
        "marker_only": True,
    }

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
    elif motion_ema >= gesture_threshold:
        status = "WORKING"
        reason = "small_hand_or_body_motion"
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
        floor_pos = person_floor_point(person)
        distance_px = float(np.linalg.norm(floor_pos - last_pos))
        iou = bbox_iou(last_bbox, person["bbox"]) if last_bbox else 0.0
        center_distance = float(np.linalg.norm(bbox_center(person["bbox"]) - last_center)) if last_center is not None else distance_px
        feature_score = feature_similarity(last_feature, person.get("feature"))
        same_track = last_track_id is not None and person.get("track_id") == last_track_id
        spatial_good = distance_px <= max_radius_px or center_distance <= max_radius_px
        strong_visual = feature_score >= 0.72
        weak_visual = feature_score >= 0.65

        if not same_track and not (spatial_good and strong_visual):
            continue

        aspect_penalty = 0.0
        if last_aspect:
            aspect_penalty = abs(np.log(bbox_aspect(person["bbox"]) / last_aspect))
        area_penalty = 0.0
        if last_bbox:
            prev_area = max(1.0, bbox_area(last_bbox))
            next_area = max(1.0, bbox_area(person["bbox"]))
            area_penalty = abs(np.log(next_area / prev_area))
            if not same_track and area_penalty > 0.85:
                continue
            if not same_track and aspect_penalty > 0.85:
                continue
            if not same_track and iou < max(min_iou, 0.08) and center_distance > max(70.0, max_radius_px * 0.35):
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
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1:
        best_person = candidates[0][2]
        second_person = candidates[1][2]
        best_same_track = last_track_id is not None and best_person.get("track_id") == last_track_id
        second_distance = float(np.linalg.norm(bbox_center(second_person["bbox"]) - last_center)) if last_center is not None else 9999.0
        if not best_same_track and second_distance <= max(120.0, max_radius_px * 0.55):
            return None
    _, idx, person = candidates[0]
    used_box_ids.add(idx)
    return person

def find_strict_employee_rebind(worker_state, person_boxes, used_box_ids):
    """Recover an employee binding only when the same body is still obvious.

    This is intentionally stricter than the normal continuation helper. It is
    used for the brief case where ByteTrack changes ID after a bend/turn, but
    we must not assign the employee to a nearby unknown worker.
    """
    last_bbox = worker_state.get("last_bbox")
    if not person_boxes or last_bbox is None:
        return None

    last_center = bbox_center(last_bbox)
    last_floor = np.array((last_center[0], last_bbox[3]), dtype=np.float32)
    last_area = max(1.0, bbox_area(last_bbox))
    last_aspect = bbox_aspect(last_bbox)
    last_feature = worker_state.get("last_feature")
    last_track_id = worker_state.get("track_id")
    marker_id = worker_state.get("opencv_marker_id")
    size_ref = max(1.0, np.sqrt(last_area))
    candidates = []

    for idx, person in enumerate(person_boxes):
        if idx in used_box_ids:
            continue
        person_marker = person.get("employee_marker_id")
        if person_marker is not None and marker_id is not None and int(person_marker) != int(marker_id):
            continue

        same_track = last_track_id is not None and person.get("track_id") == last_track_id
        iou = bbox_iou(last_bbox, person["bbox"])
        center_distance = float(np.linalg.norm(bbox_center(person["bbox"]) - last_center))
        floor_distance = float(np.linalg.norm(person_floor_point(person) - last_floor))
        feature_score = feature_similarity(last_feature, person.get("feature"))
        area_ratio = abs(np.log(max(1.0, bbox_area(person["bbox"])) / last_area))
        aspect_ratio = abs(np.log(bbox_aspect(person["bbox"]) / last_aspect)) if last_aspect else 0.0

        if same_track:
            candidates.append((3.0 + iou + feature_score, idx, person))
            continue

        shape_ok = area_ratio <= 0.55 and aspect_ratio <= 0.55
        overlap_ok = iou >= 0.35
        position_ok = center_distance <= max(55.0, size_ref * 0.28) and floor_distance <= max(85.0, size_ref * 0.42)
        appearance_ok = feature_score >= 0.68
        no_feature_strict = feature_score <= 0.0 and iou >= 0.58 and position_ok

        if not shape_ok:
            continue
        if not ((overlap_ok and appearance_ok) or (position_ok and appearance_ok) or no_feature_strict):
            continue

        score = (
            iou * 1.6
            + feature_score * 1.3
            - min(center_distance / max(1.0, size_ref), 2.0) * 0.35
            - area_ratio * 0.25
            - aspect_ratio * 0.20
        )
        candidates.append((score, idx, person))

    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.35:
        return None

    _, idx, person = candidates[0]
    used_box_ids.add(idx)
    return person

def find_safe_hidden_continuation(worker_state, person_boxes, used_box_ids, max_radius_px, min_iou):
    candidate = find_continuation_person(
        worker_state,
        person_boxes,
        used_box_ids,
        max_radius_px,
        min_iou,
    )
    if candidate is None:
        return None

    last_bbox = worker_state.get("last_bbox")
    last_feature = worker_state.get("last_feature")
    last_track_id = worker_state.get("track_id")
    same_track = last_track_id is not None and candidate.get("track_id") == last_track_id
    if same_track:
        return candidate

    if last_bbox is None:
        return None

    iou = bbox_iou(last_bbox, candidate["bbox"])
    feature_score = feature_similarity(last_feature, candidate.get("feature"))
    center_distance = float(np.linalg.norm(bbox_center(candidate["bbox"]) - bbox_center(last_bbox)))
    size_ref = max(1.0, np.sqrt(max(1.0, bbox_area(last_bbox))))
    if feature_score < 0.72:
        return None
    if iou < 0.18 and center_distance > max(95.0, size_ref * 0.65):
        return None

    close_unknowns = 0
    for person in person_boxes:
        if person is candidate:
            continue
        if person.get("employee_marker_id") is not None:
            continue
        other_distance = float(np.linalg.norm(bbox_center(person["bbox"]) - bbox_center(last_bbox)))
        if other_distance <= max(120.0, size_ref * 0.85):
            close_unknowns += 1
    if close_unknowns >= 1:
        return None
    return candidate


def bound_track_detection_is_plausible(worker_state, person, marker_age, runtime):
    if person is None:
        return False, "no bound person"
    confidence = float(person.get("confidence", 0.0))
    same_track = (
        worker_state.get("track_id") is not None
        and person.get("track_id") == worker_state.get("track_id")
    )
    if marker_age <= 4.0 and confidence >= runtime["tracker_confidence_threshold"]:
        return True, "fresh marker lock"
    min_confidence = (
        runtime["tracker_confidence_threshold"]
        if same_track and marker_age <= runtime["marker_hidden_max_seconds"]
        else runtime["identity_continuation_min_confidence"]
    )
    if confidence < min_confidence:
        return False, "bound track confidence below identity threshold"

    last_bbox = worker_state.get("last_bbox")
    if last_bbox is None:
        return marker_age <= 3.0, "no previous bbox for bound track"

    current_bbox = person["bbox"]
    iou = bbox_iou(last_bbox, current_bbox)
    last_center = bbox_center(last_bbox)
    current_center = bbox_center(current_bbox)
    center_distance = float(np.linalg.norm(current_center - last_center))
    floor_distance = float(np.linalg.norm(person_floor_point(person) - np.array((last_center[0], last_bbox[3]), dtype=np.float32)))
    last_area = max(1.0, bbox_area(last_bbox))
    current_area = max(1.0, bbox_area(current_bbox))
    area_ratio = abs(np.log(current_area / last_area))
    last_aspect = bbox_aspect(last_bbox)
    aspect_ratio = abs(np.log(bbox_aspect(current_bbox) / last_aspect)) if last_aspect else 0.0
    feature_score = feature_similarity(worker_state.get("last_feature"), person.get("feature"))
    size_ref = max(1.0, np.sqrt(last_area))

    if iou >= 0.18:
        return True, "bound track overlaps previous body"
    if center_distance <= max(115.0, size_ref * 0.70) and floor_distance <= max(150.0, size_ref * 0.90):
        return True, "bound track near previous body"
    if same_track and marker_age <= runtime["marker_hidden_max_seconds"]:
        if center_distance <= max(170.0, size_ref * 1.05) and floor_distance <= max(210.0, size_ref * 1.20):
            return True, "same bound track near previous body"
    # Appearance check BEFORE shape — overhead cameras change bbox size a lot when
    # workers move closer/farther; appearance similarity is more reliable than size.
    if feature_score >= 0.72 and center_distance <= max(170.0, size_ref * 1.10):
        return True, "bound track appearance match"
    # Shape guard: log ratio > 2.0 means ~7.4x size/aspect change — truly a different person.
    # (old threshold 1.05 = 2.86x was too tight for overhead camera depth variation)
    if area_ratio > 2.0 or aspect_ratio > 2.0:
        return False, "bound track body shape changed too much"
    return False, "bound track jumped away from previous body"


def find_single_person_stationary_continuation(worker_state, person_boxes, used_box_ids, runtime):
    available = [
        (idx, person)
        for idx, person in enumerate(person_boxes)
        if idx not in used_box_ids and float(person.get("confidence", 0.0)) >= runtime["identity_continuation_min_confidence"]
    ]
    if len(available) != 1:
        return None

    idx, person = available[0]
    last_bbox = worker_state.get("last_bbox")
    if last_bbox is None:
        return None

    current_bbox = person["bbox"]
    iou = bbox_iou(last_bbox, current_bbox)
    last_center = bbox_center(last_bbox)
    current_center = bbox_center(current_bbox)
    center_distance = float(np.linalg.norm(current_center - last_center))
    last_floor = np.array((last_center[0], last_bbox[3]), dtype=np.float32)
    floor_distance = float(np.linalg.norm(person_floor_point(person) - last_floor))
    last_area = max(1.0, bbox_area(last_bbox))
    current_area = max(1.0, bbox_area(current_bbox))
    area_ratio = abs(np.log(current_area / last_area))
    last_aspect = bbox_aspect(last_bbox)
    aspect_ratio = abs(np.log(bbox_aspect(current_bbox) / last_aspect)) if last_aspect else 0.0
    feature_score = feature_similarity(worker_state.get("last_feature"), person.get("feature"))
    size_ref = max(1.0, np.sqrt(last_area))

    if area_ratio > 1.20 or aspect_ratio > 1.20:
        return None
    if iou >= 0.10:
        used_box_ids.add(idx)
        return person
    if center_distance <= max(180.0, size_ref * 1.05) and floor_distance <= max(230.0, size_ref * 1.25):
        used_box_ids.add(idx)
        return person
    if feature_score >= 0.68 and center_distance <= max(230.0, size_ref * 1.35):
        used_box_ids.add(idx)
        return person
    return None


def scale_tracked_boxes(result, scale, frame):
    boxes = []
    if result is None or result.boxes is None or result.boxes.xyxy is None:
        return boxes
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(xyxy), dtype=int)
    for box, conf, track_id in zip(xyxy, confs, ids):
        x1, y1, x2, y2 = [float(v * scale) for v in box]
        bbox = (x1, y1, x2, y2)
        boxes.append({
            "bbox": bbox,
            "confidence": float(conf),
            "track_id": f"bs_{int(track_id)}",
            "global_id": f"bs_{int(track_id)}",
            "employee_marker_id": None,
            "feature": person_appearance_feature(frame, bbox),
        })
    return boxes

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
    delta = max(0.0, min(now_ts - last_account_ts, max(10.0, float(stale_after_seconds))))
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

def build_employee_record(marker_id, worker_state, now_ts, visible_grace_seconds=8.0):
    if worker_state.get("status") == "LOST":
        worker_state["status"] = "WORKING"
    if "status_times" in worker_state:
        worker_state["status_times"].pop("LOST", None)
    last_seen_age = now_ts - worker_state.get("last_seen_ts", now_ts)
    visible_now = bool(worker_state.get("visible_now", False))
    recently_visible = visible_now or last_seen_age <= visible_grace_seconds
    display_status = worker_state["status"] if recently_visible else "NOT_VISIBLE"
    tracking_mode = worker_state.get("tracking_mode", "unknown") if recently_visible else "waiting_for_reappearance"
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
        "proof_view": worker_state.get("proof_view", proof_recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"])),
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

def proof_recording_filename(marker_id, employee_name):
    return f"{marker_id}_{safe_filename(employee_name)}_proof.mp4"

def proof_recording_disk_path(marker_id, employee_name):
    return str(Path(recording_root()) / "proof" / proof_recording_filename(marker_id, employee_name))

def to_web_path(path):
    resolved = Path(path).resolve()
    try:
        return "/" + resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return "/" + resolved.as_posix().lstrip("/")

def recording_view_path(marker_id, employee_name, cam_id):
    return to_web_path(Path(recording_root()) / recording_filename(marker_id, employee_name, cam_id))

def proof_recording_view_path(marker_id, employee_name):
    return to_web_path(Path(recording_root()) / "proof" / proof_recording_filename(marker_id, employee_name))

def unassigned_marker_snapshot_path(marker_id, cam_id, now_ts):
    day = time.strftime("%Y-%m-%d", time.localtime(now_ts))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
    folder = Path("outputs") / "unassigned_marker_captures" / day
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / f"marker_{int(marker_id):02d}_{safe_filename(cam_id)}_{stamp}.jpg")

def write_text_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def write_jpeg_atomic(path, frame, quality):
    temp_path = path.replace(".jpg", "_tmp.jpg")
    cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    try:
        os.replace(temp_path, path)
    except OSError:
        pass

def append_tracking_debug(path, rows):
    if not rows:
        return
    debug_path = Path(path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not debug_path.exists()
    fieldnames = [
        "ts",
        "camera",
        "marker_id",
        "employee",
        "event",
        "reason",
        "marker_age_sec",
        "persons",
        "last_track_id",
        "matched_track_id",
        "person_conf",
        "tracking_mode",
    ]
    with debug_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

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

    Path("logs/factory_ai.stop").unlink(missing_ok=True)

    for path in Path(runtime["recording_dir"]).rglob("*.part.mp4"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in Path(runtime["recording_dir"]).rglob("*.mp4"):
        try:
            if path.stat().st_size <= 1024:
                path.unlink()
        except OSError:
            pass
class VideoRecorder:
    def __init__(self, fps, codec, min_frames=3, segment_seconds=60.0):
        self.fps = fps
        self.codec = codec
        self.min_frames = min_frames
        self.segment_seconds = segment_seconds
        self._writers = {}

    @staticmethod
    def _segment_paths(path):
        base_path = Path(path)
        stamp = time.strftime("%H%M%S")
        final_path = base_path.with_name(f"{base_path.stem}_{stamp}{base_path.suffix}")
        temp_path = final_path.with_name(f"{final_path.stem}.part{final_path.suffix}")
        return str(final_path), str(temp_path)

    def _finish_writer(self, path, writer_info):
        writer_info["writer"].release()
        temp_path = Path(writer_info.get("temp_path") or path)
        final_path = Path(writer_info.get("final_path") or path)
        frames = int(writer_info.get("frames", 0))
        try:
            if frames >= self.min_frames and temp_path.exists() and temp_path.stat().st_size > 1024:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.replace(final_path)
            else:
                temp_path.unlink(missing_ok=True)
                if final_path.exists() and final_path.stat().st_size <= 1024:
                    final_path.unlink(missing_ok=True)
        except OSError:
            pass

    def write(self, path, frame):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        height, width = frame.shape[:2]
        writer_info = self._writers.get(path)
        segment_expired = (
            writer_info is not None
            and time.time() - float(writer_info.get("started_at", time.time())) >= self.segment_seconds
        )
        if writer_info is None or writer_info["size"] != (width, height) or segment_expired:
            if writer_info is not None:
                self._finish_writer(path, writer_info)
            fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
            final_path, temp_path = self._segment_paths(path)
            Path(temp_path).unlink(missing_ok=True)
            writer = cv2.VideoWriter(temp_path, fourcc, self.fps, (width, height))
            if not writer.isOpened():
                return False
            self._writers[path] = {
                "writer": writer,
                "size": (width, height),
                "frames": 0,
                "temp_path": temp_path,
                "final_path": final_path,
                "started_at": time.time(),
            }
            writer_info = self._writers[path]
        writer_info["writer"].write(frame)
        writer_info["frames"] += 1
        return True

    def close(self):
        for path, writer_info in self._writers.items():
            self._finish_writer(path, writer_info)
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

def merge_recordings_on_shutdown(runtime):
    try:
        from tools.reporting.merge_recordings import merge_recordings

        results = merge_recordings(runtime["recording_dir"], runtime["path_video_codec"])
        if results:
            print(f"Merged {len(results)} employee/proof recording file(s).")
    except Exception as exc:
        print(f"Recording merge skipped: {exc}")

def redirect_decoder_stderr(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_handle = open(log_file, "a", buffering=1, encoding="utf-8", errors="replace")
    log_handle.write(f"\n--- decoder log started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    sys.stderr.flush()
    os.dup2(log_handle.fileno(), 2)
    return log_handle

def start_production_watchdog(runtime, heartbeat):
    if os.environ.get("FACTORY_AI_PRODUCTION") != "1":
        return None

    timeout = max(30.0, float(runtime.get("production_watchdog_timeout_seconds", 120.0)))
    interval = max(5.0, float(runtime.get("production_watchdog_check_seconds", 10.0)))
    log_path = Path(runtime["log_dir"]) / "watchdog.log"

    def _watch():
        while True:
            time.sleep(interval)
            age = time.time() - heartbeat.get("last_stats_write_ts", time.time())
            if age > timeout:
                message = (
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"watchdog_exit live_stats_stale_seconds={age:.1f} timeout={timeout:.1f}\n"
                )
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(message)
                    print(message.strip(), flush=True)
                finally:
                    os._exit(70)

    thread = threading.Thread(target=_watch, name="production-watchdog", daemon=True)
    thread.start()
    return thread

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
            
    # 3:00 - 3:15 Tea Break
    if hour == 15 and minute < 15:
        return "TEA BREAK IN PROGRESS"

    # Night shift dinner: 9:00 PM - 9:30 PM
    if hour == 21 and minute < 30:
        return "DINNER BREAK IN PROGRESS"

    # Night shift tea: 11:00 PM - 11:15 PM
    if hour == 23 and minute < 15:
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
                    payload = json.loads(Path(registry_path).read_text(encoding="utf-8-sig"))
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
    employee_map = {}
    for _, r in employee_df.iterrows():
        active = str(r.get("active", "yes")).strip().lower() != "no"
        name = str(r.get("name", "")).strip()
        opencv_marker_id = int(r.get("opencv_marker_id", r["marker_id"]))
        display_marker_id = int(r["marker_id"])
        if not active or not name or name.lower().startswith("marker_") or "unassigned" in name.lower():
            name = f"Marker {display_marker_id:02d} Unassigned"
            department = "Unassigned"
        else:
            department = r.get("department", "Production")
        employee_map[opencv_marker_id] = {
            "id": display_marker_id,
            "opencv_marker_id": opencv_marker_id,
            "name": name,
            "dept": department,
            "active": active,
            "assigned": active and department != "Unassigned",
        }
    print(f"[STARTUP] Loaded employee_map with IDs: {sorted(employee_map.keys())}")

    tracker_backend = runtime["tracker_backend"]
    inference_workers = [] if tracker_backend == "botsort" else load_models_for_devices(config["detection"]["model_path"], devices)
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
        f"tracker={tracker_backend}"
    )
    # BoT-SORT VRAM warning: one YOLO model per camera, each ~500MB VRAM
    if tracker_backend == "botsort" and acceleration["cuda_available"]:
        enabled_cam_count = sum(1 for c in config.get("cameras", []) if c.get("enabled", True))
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        estimated_vram_needed = enabled_cam_count * 500
        print(
            f"[GPU] GPU: {acceleration['cuda_device_name']}, VRAM: {vram_mb}MB. "
            f"BoT-SORT needs ~{estimated_vram_needed}MB for {enabled_cam_count} cameras "
            f"({enabled_cam_count} models x ~500MB)."
        )
        if estimated_vram_needed > vram_mb * 0.85:
            print(
                f"[GPU WARNING] Estimated VRAM needed ({estimated_vram_needed}MB) exceeds "
                f"85% of GPU VRAM ({vram_mb}MB). "
                f"GPU utilization will be LOW. Disable {enabled_cam_count - int(vram_mb * 0.85 / 500)} cameras "
                f"or switch to tracker_backend: bytetrack in settings.yaml."
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
    last_unassigned_snapshot_ts = {}
    aruco_pos_hints = {}   # track_id -> rel_y of last confirmed marker position within person bbox
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
    confidence = min(
        float(runtime["tracker_confidence_threshold"]),
        float(config["detection"].get("confidence_threshold", 0.25)),
    )
    display_confidence = float(runtime["display_confidence_threshold"])
    classes = config["detection"].get("classes", [0])
    if tracker_backend == "botsort":
        person_tracker = BoTSORTProductionTracker(
            config["detection"]["model_path"],
            devices[0] if devices else "cpu",
            runtime["botsort_tracker_yaml"],
            runtime["track_binding_ttl_seconds"],
        )
    else:
        person_tracker = ByteTrackStyleTracker(runtime)

    aruco_pool = concurrent.futures.ThreadPoolExecutor(max_workers=runtime["aruco_workers"])
    image_writer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=runtime["image_writer_workers"])
    pid_file = Path("logs/factory_ai.pid")
    stop_file = Path("logs/factory_ai.stop")
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    path_video_recorder = VideoRecorder(
        runtime["path_video_fps"],
        runtime["path_video_codec"],
        runtime["path_video_min_frames"],
        runtime["path_video_segment_seconds"],
    )
    tracking_debug_path = os.path.join(runtime["log_dir"], "tracking_debug.csv")
    tracking_debug_rows = []
    last_camera_sync_ts = 0.0
    last_stats_write_ts = 0.0
    stats_write_interval = 0.5  # Write live_stats.json only every 0.5 seconds (2x/sec instead of 8x/sec)
    watchdog_heartbeat = {"last_stats_write_ts": time.time()}
    start_production_watchdog(runtime, watchdog_heartbeat)

    try:
        while True:
            if stop_file.exists():
                print("Graceful stop requested; finalizing recordings and report...")
                break
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
                        "tracker_backend": runtime["tracker_backend"],
                        "target_fps": runtime["target_fps"],
                        "path_video_every_n_frames": runtime["path_video_every_n_frames"],
                        "path_video_fps": runtime["path_video_fps"],
                        "reconnect_after": runtime["reconnect_after"],
                        "reconnect_interval_seconds": runtime["reconnect_interval_seconds"],
                        "stale_camera_after_seconds": runtime["stale_camera_after_seconds"],
                        "distance_methods": distance_methods,
                        "marker_confirmations_required": runtime["marker_confirmations_required"],
                        "marker_stale_after_seconds": runtime["marker_stale_after_seconds"],
                        "marker_hidden_grace_seconds": runtime["marker_hidden_grace_seconds"],
                        "marker_hidden_match_radius_px": runtime["marker_hidden_match_radius_px"],
                        "tracker_max_age_seconds": runtime["tracker_max_age_seconds"],
                        "tracker_confidence_threshold": runtime["tracker_confidence_threshold"],
                        "display_confidence_threshold": runtime["display_confidence_threshold"],
                        "track_binding_ttl_seconds": runtime["track_binding_ttl_seconds"],
                        "identity_continuation_max_seconds": runtime["identity_continuation_max_seconds"],
                        "identity_continuation_min_confidence": runtime["identity_continuation_min_confidence"],
                        "single_person_continuation_max_seconds": runtime["single_person_continuation_max_seconds"],
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
                    frame, read_ts = cam.latest(copy=False)
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
                if tracker_backend == "botsort":
                    inference_tasks.append((prepared_frames, None, [None] * len(prepared_frames)))
                else:
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
                        if tracker_backend == "botsort":
                            person_boxes = person_tracker.update(
                                cam_id,
                                item["frame"],
                                item["yolo_scale"],
                                render_frame,
                                confidence,
                                classes,
                                runtime["inference_width"],
                            )
                        else:
                            person_boxes = scale_person_boxes(results, item["yolo_scale"])
                            person_boxes = person_tracker.update(cam_id, person_boxes, render_frame, now_ts)
                        used_person_box_ids = set()
                        marker_seen_workers = set()

                        person_count = len(person_boxes)
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
                                runtime.get("debug_mode", False),
                                set(employee_map.keys()),
                                runtime["assigned_min_marker_area"],
                                runtime["unassigned_min_marker_area"],
                                runtime["ignored_opencv_marker_ids"],
                                aruco_pos_hints,
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
                        if ids is not None and corners is not None:
                            for marker_index, marker_id_value in enumerate(ids.flatten()):
                                if marker_person_matches.get(marker_index) is not None:
                                    continue
                                marker_id_value = int(marker_id_value)
                                pts_full = corners[marker_index][0].astype(np.float32)
                                marker_center = pts_full.mean(axis=0)
                                matched_person = find_person_for_marker(
                                    marker_center,
                                    person_boxes,
                                    0.22,
                                    history.get(marker_id_value),
                                )
                                if matched_person is None:
                                    continue
                                if not expanded_contains(matched_person["bbox"], marker_center, 0.22):
                                    continue
                                if not marker_quality_ok(pts_full, matched_person["bbox"], min_side_px=8.0):
                                    continue
                                marker_person_matches[marker_index] = matched_person

                        num_markers = int(ids.size) if ids is not None else 0
                        num_crop_matches = sum(1 for match in marker_person_matches.values() if match is not None)
                        if runtime.get("debug_mode"):
                            print(
                                f"[DEBUG] cam={cam_id} frame={frame_index} persons={len(person_boxes)} "
                                f"markers={num_markers} crop_matches={num_crop_matches} "
                                f"pending_switches={len(pending_track_switches)}"
                            )

                        if ids is not None:
                            detected_ids = [
                                int(mid)
                                for mid in ids.flatten().tolist()
                                if int(mid) not in runtime["ignored_opencv_marker_ids"]
                            ]
                            unmapped_ids = [mid for mid in detected_ids if employee_map.get(mid) is None]
                            if unmapped_ids:
                                print(f"[MARKER UNKNOWN] cam={cam_id} frame={frame_index}: OpenCV IDs {unmapped_ids} not in employees.csv")
                            for i, marker_id in enumerate(ids.flatten()):
                                marker_id = int(marker_id)
                                if marker_id in runtime["ignored_opencv_marker_ids"]:
                                    continue
                                emp = employee_map.get(marker_id)
                                if not emp:
                                    continue

                                marker_corners = corners[i][0]
                                marker_area_minimum = (
                                    runtime["assigned_min_marker_area"]
                                    if emp.get("assigned", False)
                                    else runtime["unassigned_min_marker_area"]
                                )
                                if marker_area(marker_corners) < marker_area_minimum:
                                    continue

                                matched_person = marker_person_matches.get(i)
                                existing_state = history.get(marker_id)
                                if matched_person is None:
                                    marker_seen_workers.add(marker_id)
                                    for key in list(marker_confirmations):
                                        if key[0] == cam_id and key[1] == marker_id:
                                            marker_confirmations.pop(key, None)
                                    tracking_debug_rows.append({
                                        "ts": frame_data["timestamp"],
                                        "camera": cam_id,
                                        "marker_id": marker_id,
                                        "employee": emp.get("name", ""),
                                        "event": "marker_seen_no_person_match",
                                        "reason": "marker detected but not inside any YOLO person box",
                                        "marker_age_sec": 0,
                                        "persons": len(person_boxes),
                                        "last_track_id": existing_state.get("track_id") if existing_state else "",
                                        "matched_track_id": "",
                                        "person_conf": "",
                                        "tracking_mode": "aruco_reject",
                                    })
                                    if runtime.get("debug_mode"):
                                        print(
                                            f"[MARKER IGNORE] cam={cam_id} frame={frame_index}: "
                                            f"marker={marker_id} is not inside a detected person"
                                    )
                                    continue
                                matched_track_id = matched_person.get("track_id")
                                owner_marker_id = matched_person.get("employee_marker_id")
                                if owner_marker_id is not None and int(owner_marker_id) != marker_id:
                                    owner_marker_id = int(owner_marker_id)
                                    owner_state = history.get(owner_marker_id)
                                    owner_recent_same_camera = (
                                        owner_state is not None
                                        and owner_state.get("cam_id") == cam_id
                                        and (
                                            now_ts - owner_state.get("last_marker_seen_ts", 0.0)
                                        ) <= runtime["marker_stale_after_seconds"]
                                    )
                                    if owner_recent_same_camera:
                                        tracking_debug_rows.append({
                                            "ts": frame_data["timestamp"],
                                            "camera": cam_id,
                                            "marker_id": marker_id,
                                            "employee": emp.get("name", ""),
                                            "event": "marker_rejected_track_owned",
                                            "reason": f"person track recently belongs to marker {owner_marker_id}",
                                            "marker_age_sec": 0,
                                            "persons": len(person_boxes),
                                            "last_track_id": existing_state.get("track_id") if existing_state else "",
                                            "matched_track_id": matched_track_id or "",
                                            "person_conf": matched_person.get("confidence", ""),
                                            "tracking_mode": "aruco_reject",
                                        })
                                        continue

                                    # BoT-SORT can carry a stale employee binding after occlusion or ID reuse.
                                    # A fresh marker inside the current YOLO body must be allowed to correct it.
                                    if (
                                        owner_state is not None
                                        and owner_state.get("cam_id") == cam_id
                                        and owner_state.get("track_id") == matched_track_id
                                    ):
                                        owner_state["track_id"] = None
                                        owner_state["tracking_mode"] = "waiting_for_reappearance"
                                    matched_person["employee_marker_id"] = None
                                    matched_person["global_id"] = matched_person.get("global_id")
                                    tracking_debug_rows.append({
                                        "ts": frame_data["timestamp"],
                                        "camera": cam_id,
                                        "marker_id": marker_id,
                                        "employee": emp.get("name", ""),
                                        "event": "marker_allowed_stale_track_owner",
                                        "reason": f"visible marker overrides stale track owner {owner_marker_id}",
                                        "marker_age_sec": 0,
                                        "persons": len(person_boxes),
                                        "last_track_id": existing_state.get("track_id") if existing_state else "",
                                        "matched_track_id": matched_track_id or "",
                                        "person_conf": matched_person.get("confidence", ""),
                                        "tracking_mode": "aruco_rebind",
                                    })
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

                                marker_side = float(max(
                                    np.linalg.norm(marker_corners[1] - marker_corners[0]),
                                    np.linalg.norm(marker_corners[2] - marker_corners[1]),
                                    np.linalg.norm(marker_corners[3] - marker_corners[2]),
                                    np.linalg.norm(marker_corners[0] - marker_corners[3]),
                                ))
                                required_confirmations = runtime["marker_confirmations_required"]
                                if not emp.get("assigned", False):
                                    required_confirmations = runtime["unassigned_marker_confirmations_required"]
                                elif marker_id in history or marker_side >= 14.0:
                                    required_confirmations = 1

                                confirmation_key = (cam_id, marker_id, matched_track_id or id(matched_person))
                                marker_confirmations[confirmation_key] = marker_confirmations.get(confirmation_key, 0) + 1
                                if (
                                    marker_id not in history
                                    and marker_confirmations[confirmation_key] < required_confirmations
                                ):
                                    continue

                                for key in list(marker_confirmations):
                                    if key[0] == cam_id and key[1] == marker_id and key != confirmation_key:
                                        marker_confirmations.pop(key, None)

                                try:
                                    used_person_box_ids.add(person_boxes.index(matched_person))
                                except ValueError:
                                    pass
                                if matched_person.get("track_id") is not None:
                                    person_tracker.bind_employee(cam_id, matched_person["track_id"], marker_id)
                                    matched_person["employee_marker_id"] = marker_id
                                    matched_person["global_id"] = f"emp_{marker_id}"

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
                                        "proof_view": proof_recording_view_path(emp.get("id", marker_id), emp["name"]),
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
                                    worker_state["proof_view"] = proof_recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"])
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
                                worker_state["proof_view"] = proof_recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"])
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
                                if not emp.get("assigned", False):
                                    snapshot_key = (cam_id, marker_id)
                                    if now_ts - last_unassigned_snapshot_ts.get(snapshot_key, 0.0) >= 20.0:
                                        snapshot_path = unassigned_marker_snapshot_path(emp.get("id", marker_id), cam_id, now_ts)
                                        image_writer_pool.submit(
                                            write_jpeg_atomic,
                                            snapshot_path,
                                            render_frame.copy(),
                                            82,
                                        )
                                        last_unassigned_snapshot_ts[snapshot_key] = now_ts

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
                            if matched_person is not None:
                                # Proximity guard: if another unowned person overlaps this
                                # detection by >30% IoU, the track may have swapped during
                                # close proximity. Require a fresh ArUco to re-confirm.
                                if marker_age > 4.0:
                                    m_bbox = matched_person.get("bbox")
                                    for other in person_boxes:
                                        if other is matched_person:
                                            continue
                                        if other.get("employee_marker_id") is not None:
                                            continue  # already owned by someone
                                        if m_bbox and bbox_iou(m_bbox, other.get("bbox", (0,0,0,0))) > 0.30:
                                            matched_person = None
                                            reason = "proximity swap guard — overlapping unknown worker"
                                            break
                            if matched_person is not None:
                                plausible, reason = bound_track_detection_is_plausible(
                                    worker_state,
                                    matched_person,
                                    marker_age,
                                    runtime,
                                )
                                if not plausible:
                                    tracking_debug_rows.append({
                                        "ts": frame_data["timestamp"],
                                        "camera": cam_id,
                                        "marker_id": marker_id,
                                        "employee": worker_state.get("name", ""),
                                        "event": "yellow_not_continued",
                                        "reason": reason,
                                        "marker_age_sec": round(marker_age, 2),
                                        "persons": len(person_boxes),
                                        "last_track_id": worker_state.get("track_id", ""),
                                        "matched_track_id": matched_person.get("track_id", ""),
                                        "person_conf": matched_person.get("confidence", ""),
                                        "tracking_mode": tracking_mode,
                                    })
                                    matched_person = None
                            if (
                                matched_person is None
                                and marker_age <= runtime["identity_continuation_max_seconds"]
                            ):
                                matched_person = find_strict_employee_rebind(
                                    worker_state,
                                    person_boxes,
                                    used_person_box_ids,
                                )
                                tracking_mode = "strict_rebind"
                            if (
                                matched_person is None
                                and marker_age <= min(
                                    runtime["marker_hidden_max_seconds"],
                                    runtime["identity_continuation_max_seconds"],
                                )
                            ):
                                matched_person = find_safe_hidden_continuation(
                                    worker_state,
                                    person_boxes,
                                    used_person_box_ids,
                                    runtime["marker_hidden_match_radius_px"],
                                    runtime["marker_hidden_min_iou"],
                                )
                                tracking_mode = "safe_hidden"
                            if (
                                matched_person is None
                                and marker_age <= runtime["single_person_continuation_max_seconds"]
                            ):
                                matched_person = find_single_person_stationary_continuation(
                                    worker_state,
                                    person_boxes,
                                    used_person_box_ids,
                                    runtime,
                                )
                                tracking_mode = "single_person_hold"
                            if (
                                matched_person is not None
                                and float(matched_person.get("confidence", 0.0)) < runtime["identity_continuation_min_confidence"]
                            ):
                                tracking_debug_rows.append({
                                    "ts": frame_data["timestamp"],
                                    "camera": cam_id,
                                    "marker_id": marker_id,
                                    "employee": worker_state.get("name", ""),
                                    "event": "yellow_not_continued",
                                    "reason": "continuation person confidence below identity threshold",
                                    "marker_age_sec": round(marker_age, 2),
                                    "persons": len(person_boxes),
                                    "last_track_id": worker_state.get("track_id", ""),
                                    "matched_track_id": matched_person.get("track_id", ""),
                                    "person_conf": matched_person.get("confidence", ""),
                                    "tracking_mode": tracking_mode,
                                })
                                matched_person = None
                            if matched_person is None:
                                tracking_debug_rows.append({
                                    "ts": frame_data["timestamp"],
                                    "camera": cam_id,
                                    "marker_id": marker_id,
                                    "employee": worker_state.get("name", ""),
                                    "event": "yellow_not_continued",
                                    "reason": "no current person body matched to locked employee",
                                    "marker_age_sec": round(marker_age, 2),
                                    "persons": len(person_boxes),
                                    "last_track_id": worker_state.get("track_id", ""),
                                    "matched_track_id": "",
                                    "person_conf": worker_state.get("person_conf", ""),
                                    "tracking_mode": "lost",
                                })
                                continue

                            matched_person["employee_marker_id"] = marker_id
                            matched_person["global_id"] = f"emp_{marker_id}"
                            if matched_person.get("track_id") is not None:
                                person_tracker.bind_employee(cam_id, matched_person["track_id"], marker_id)
                            try:
                                used_person_box_ids.add(person_boxes.index(matched_person))
                            except ValueError:
                                pass

                            floor_pos = person_floor_point(matched_person)
                            calibration = distance_calibrations.get(cam_id, {})
                            map_pos = frame_point_to_map(floor_pos, render_frame.shape, calibration)
                            worker_zone = resolve_worker_zone(map_pos, cam_id, floor_model)

                            worker_state["last_seen_ts"] = now_ts
                            worker_state["person_conf"] = matched_person["confidence"]
                            worker_state["track_id"] = matched_person.get("track_id")
                            worker_state["path_view"] = recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"], cam_id)
                            worker_state["proof_view"] = proof_recording_view_path(worker_state.get("display_marker_id", marker_id), worker_state["name"])
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
                            if float(person.get("confidence", 0.0)) < display_confidence:
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
                    account_worker_time(
                        state,
                        now_ts,
                        min(runtime["marker_hidden_max_seconds"], runtime["marker_hidden_grace_seconds"]),
                    )
                    current_cam = state.get("cam_id", "unknown")
                    visible_this_loop = state.get("last_visual_frame") == loop_counter
                    state["visible_now"] = visible_this_loop
                    state["path"] = state.get("camera_paths", {}).get(current_cam, state.get("path", []))
                    display_marker_id = state.get("display_marker_id", mid)
                    state["path_view"] = recording_view_path(display_marker_id, state["name"], current_cam)
                    employee_record = build_employee_record(
                        mid,
                        state,
                        now_ts,
                        min(runtime["marker_hidden_max_seconds"], runtime["marker_hidden_grace_seconds"]),
                    )
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
                        path_video_recorder.write(proof_recording_disk_path(display_marker_id, state["name"]), path_frame)

                    frame_data["stats"].append(employee_record)

                frame_data["total_person_count"] = total_detected_people

                # Throttle all JSON writes to reduce I/O contention
                if now_ts - last_stats_write_ts >= stats_write_interval:
                    # employee_records — moved here from per-frame write (was blocking 15x/sec)
                    _emp_payload = json.dumps({"timestamp": frame_data["timestamp"], "employees": employee_records})
                    image_writer_pool.submit(write_text_file, runtime["employee_records_file"], _emp_payload)
                    append_tracking_debug(tracking_debug_path, tracking_debug_rows)
                    tracking_debug_rows.clear()
                    for live_stats_path in {runtime["live_stats_file"], runtime["dashboard_live_stats_file"]}:
                        with open(live_stats_path, "w") as f:
                            json.dump(frame_data, f, indent=2)
                    last_stats_write_ts = now_ts
                    watchdog_heartbeat["last_stats_write_ts"] = time.time()

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
        merge_recordings_on_shutdown(runtime)
        export_employee_report_on_shutdown()
        pid_file.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)
        aruco_pool.shutdown(wait=False, cancel_futures=True)
        image_writer_pool.shutdown(wait=False, cancel_futures=True)
        for cam in active_cameras.values():
            cam.stop()
        if decoder_log_handle is not None:
            decoder_log_handle.close()

if __name__ == "__main__":
    main()
