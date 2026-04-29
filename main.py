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
from datetime import datetime
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
from collections import deque
from ultralytics import YOLO
from src.runtime.camera_capture import ThreadedCamera
from src.runtime.performance import PerformanceMonitor

try:
    import torch
except Exception:
    torch = None

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run Factory AI real-time tracking")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--camera-profile", default=None, help="Camera stream profile, e.g. main_stream or sub_stream")
    parser.add_argument("--inference-width", type=int, default=None, help="Resize width before AI inference")
    parser.add_argument("--target-fps", type=float, default=None, help="Main processing loop FPS cap")
    parser.add_argument("--batch-size", type=int, default=None, help="Max camera frames per YOLO batch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    return parser.parse_args()

def get_runtime_config(config, args):
    runtime = config.get("runtime", {})
    profile_name = args.camera_profile or runtime.get("camera_profile", "main_stream")
    profile = config.get("camera_profiles", {}).get(profile_name, {})
    return {
        "camera_profile": profile_name,
        "camera_profile_settings": profile,
        "inference_width": max(160, int(args.inference_width or runtime.get("inference_width", 1024))),
        "target_fps": max(1.0, float(args.target_fps or config.get("system", {}).get("fps_target", 15))),
        "batch_size": max(1, int(args.batch_size or runtime.get("batch_size", 8))),
        "aruco_max_width": max(160, int(runtime.get("aruco_max_width", 1920))),
        "aruco_every_n_frames": max(1, int(runtime.get("aruco_every_n_frames", 3))),
        "aruco_workers": max(1, int(runtime.get("aruco_workers", 2))),
        "dashboard_image_width": max(320, int(runtime.get("dashboard_image_width", 960))),
        "dashboard_fps": max(1.0, float(runtime.get("dashboard_fps", 5.0))),
        "image_writer_workers": max(1, int(runtime.get("image_writer_workers", 2))),
        "path_video_every_n_frames": max(1, int(runtime.get("path_video_every_n_frames", 10))),
        "path_video_fps": max(1.0, float(runtime.get("path_video_fps", 5.0))),
        "path_video_codec": str(runtime.get("path_video_codec", "mp4v")),
        "redirect_decoder_logs": bool(runtime.get("redirect_decoder_logs", True)),
        "decoder_log_file": runtime.get("decoder_log_file", "logs/decoder_ffmpeg.log"),
        "monitor_interval_seconds": runtime.get("monitor_interval_seconds", 2.0),
        "jpeg_quality_live": runtime.get("jpeg_quality_live", 75),
        "reconnect_after": runtime.get("reconnect_after", 10),
        "reconnect_interval_seconds": float(runtime.get("reconnect_interval_seconds", 5.0)),
        "stale_camera_after_seconds": float(runtime.get("stale_camera_after_seconds", 3.0)),
        "marker_confirmations_required": max(1, int(runtime.get("marker_confirmations_required", 2))),
        "marker_stale_after_seconds": float(runtime.get("marker_stale_after_seconds", 8.0)),
        "person_box_margin": float(runtime.get("person_box_margin", 0.15)),
        "min_marker_area": float(runtime.get("min_marker_area", 80.0)),
        "employee_records_file": runtime.get("employee_records_file", "logs/employee_records.json"),
    }

def select_device(requested_device):
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

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
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if corners is not None and aruco_scale != 1.0:
        corners = tuple(corner * aruco_scale for corner in corners)
    return corners, ids

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

def expanded_contains(box, point, margin_ratio):
    x1, y1, x2, y2 = box
    px, py = point
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    return x1 - margin_x <= px <= x2 + margin_x and y1 - margin_y <= py <= y2 + margin_y

def marker_area(corners):
    return float(cv2.contourArea(corners.astype(np.float32)))

def find_person_for_marker(marker_center, person_boxes, margin_ratio):
    containing = [
        person for person in person_boxes
        if expanded_contains(person["bbox"], marker_center, margin_ratio)
    ]
    if not containing:
        return None
    px, py = marker_center
    return min(
        containing,
        key=lambda person: abs(((person["bbox"][0] + person["bbox"][2]) / 2) - px)
        + abs(((person["bbox"][1] + person["bbox"][3]) / 2) - py),
    )

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
    return {
        "id": marker_id,
        "name": worker_state["name"],
        "dept": worker_state["dept"],
        "status": worker_state["status"],
        "current_camera": worker_state.get("cam_id"),
        "total_distance_ft": float(round(worker_state["dist"], 2)),
        "last_seen_age": round(now_ts - worker_state.get("last_seen_ts", now_ts), 2),
        "person_conf": round(worker_state.get("person_conf", 0.0), 3),
        "camera_times_sec": {k: round(v, 2) for k, v in worker_state.get("camera_times", {}).items()},
        "status_times_sec": {k: round(v, 2) for k, v in worker_state.get("status_times", {}).items()},
        "path_view": worker_state.get("path_view", f"/outputs/recordings/{marker_id}.mp4"),
    }

def write_jpeg_atomic(path, frame, quality):
    temp_path = path.replace(".jpg", "_tmp.jpg")
    cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    try:
        os.replace(temp_path, path)
    except OSError:
        pass

class VideoRecorder:
    def __init__(self, fps, codec):
        self.fps = fps
        self.codec = codec
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
            self._writers[path] = {"writer": writer, "size": (width, height)}
            writer_info = self._writers[path]
        writer_info["writer"].write(frame)
        return True

    def close(self):
        for writer_info in self._writers.values():
            writer_info["writer"].release()
        self._writers.clear()

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

def start_server(port=8000):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args): return
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            super().end_headers()
    class ReusableThreadingTCPServer(ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    while port < 8010:
        try:
            with ReusableThreadingTCPServer(("", port), QuietHandler) as httpd:
                with open("logs/current_port.txt", "w") as f: f.write(str(port))
                httpd.serve_forever()
        except OSError: port += 1

def main():
    args = parse_args()
    config = load_config(args.config)
    runtime = get_runtime_config(config, args)
    device = select_device(args.device)
    acceleration = get_acceleration_info(args.device, device)
    decoder_log_handle = None
    if runtime["redirect_decoder_logs"]:
        decoder_log_handle = redirect_decoder_stderr(runtime["decoder_log_file"])

    employee_df = pd.read_csv("data/employees.csv")
    employee_map = {
        int(r["marker_id"]): {"name": r["name"], "dept": r.get("department", "Production")}
        for _, r in employee_df.iterrows()
        if not str(r["name"]).startswith("Employee_")
    }

    model = YOLO(config["detection"]["model_path"])
    if hasattr(model, "to"):
        model.to(device)
    if torch is not None and device == "cuda":
        torch.backends.cudnn.benchmark = True
    if acceleration["cuda_warning"]:
        print(acceleration["cuda_warning"])
    print(
        f"Runtime: device={device}, batch={runtime['batch_size']}, "
        f"inference_width={runtime['inference_width']}, aruco_every={runtime['aruco_every_n_frames']}, "
        f"dashboard={runtime['dashboard_image_width']}px/{runtime['dashboard_fps']}fps"
    )
    if runtime["redirect_decoder_logs"]:
        print(f"Decoder warnings redirected to {runtime['decoder_log_file']}")

    profile = runtime["camera_profile_settings"]
    cap_width, cap_height = parse_resolution(profile.get("resolution"))
    cameras = []
    for cam in [c for c in config["cameras"] if c["enabled"]]:
        source = resolve_camera_source(cam, runtime["camera_profile"])
        if not source:
            continue
        camera = ThreadedCamera(
            camera_id=cam["id"],
            video_path=source,
            width=cap_width,
            height=cap_height,
            fps=profile.get("fps"),
            reconnect_after=runtime["reconnect_after"],
            reconnect_interval_seconds=runtime["reconnect_interval_seconds"],
            stale_after_seconds=runtime["stale_camera_after_seconds"],
        )
        camera.start()
        cameras.append(camera)

    if not cameras:
        return

    history = {}
    marker_confirmations = {}
    last_render_frames = {}
    last_aruco_frame_index = {}
    last_dashboard_save_ts = {}
    pending_live_writes = {}
    loop_counter = 0
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs/live", exist_ok=True)
    os.makedirs("outputs/recordings", exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["aruco"]["dictionary_type"]))
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)

    session_start = time.time()
    scale = config["mapping"].get("scale_factor", 50.0)
    monitor = PerformanceMonitor(runtime["monitor_interval_seconds"])
    loop_interval = 1.0 / max(float(runtime["target_fps"]), 1.0)
    confidence = config["detection"].get("confidence_threshold", 0.25)
    classes = config["detection"].get("classes", [0])

    aruco_pool = concurrent.futures.ThreadPoolExecutor(max_workers=runtime["aruco_workers"])
    image_writer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=runtime["image_writer_workers"])
    path_video_recorder = VideoRecorder(runtime["path_video_fps"], runtime["path_video_codec"])

    try:
        while True:
            loop_counter += 1
            loop_started = time.time()
            try:
                now_ts = time.time()
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
                        "inference_width": runtime["inference_width"],
                        "aruco_max_width": runtime["aruco_max_width"],
                        "aruco_every_n_frames": runtime["aruco_every_n_frames"],
                        "aruco_workers": runtime["aruco_workers"],
                        "dashboard_image_width": runtime["dashboard_image_width"],
                        "dashboard_fps": runtime["dashboard_fps"],
                        "batch_size": runtime["batch_size"],
                        "device": device,
                        "target_fps": runtime["target_fps"],
                        "path_video_every_n_frames": runtime["path_video_every_n_frames"],
                        "path_video_fps": runtime["path_video_fps"],
                        "marker_confirmations_required": runtime["marker_confirmations_required"],
                        "marker_stale_after_seconds": runtime["marker_stale_after_seconds"],
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
                    item = {
                        "id": cam.id,
                        "frame": proc_frame,
                        "render_frame": render_frame,
                        "yolo_scale": yolo_scale,
                        "read_age_ms": int((now_ts - read_ts) * 1000) if read_ts else None,
                        "stream_status": stream_status,
                    }
                    prepared_frames.append(item)

                    frame_index = stream_status.get("frames_read", 0)
                    last_index = last_aruco_frame_index.get(cam.id, 0)
                    if frame_index - last_index >= runtime["aruco_every_n_frames"]:
                        last_aruco_frame_index[cam.id] = frame_index
                        aruco_futures[cam.id] = aruco_pool.submit(
                            detect_aruco_markers,
                            aruco_dict,
                            render_frame,
                            runtime["aruco_max_width"],
                        )

                total_detected_people = 0
                for frame_batch in chunked(prepared_frames, int(runtime["batch_size"])):
                    if not frame_batch:
                        continue

                    results_batch = model(
                        [item["frame"] for item in frame_batch],
                        conf=confidence,
                        classes=classes,
                        verbose=False,
                        device=device,
                    )

                    for item, results in zip(frame_batch, results_batch):
                        cam_id = item["id"]
                        render_frame = item["render_frame"]
                        person_boxes = scale_person_boxes(results, item["yolo_scale"])

                        person_count = len(results.boxes)
                        total_detected_people += person_count
                        if cam_id in aruco_futures:
                            try:
                                corners, ids = aruco_futures[cam_id].result(timeout=1.0)
                            except concurrent.futures.TimeoutError:
                                corners, ids = None, None
                        else:
                            corners, ids = None, None

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
                                matched_person = find_person_for_marker(
                                    raw_pos,
                                    person_boxes,
                                    runtime["person_box_margin"],
                                )
                                if matched_person is None:
                                    marker_confirmations.pop((cam_id, marker_id), None)
                                    continue

                                confirmation_key = (cam_id, marker_id)
                                marker_confirmations[confirmation_key] = marker_confirmations.get(confirmation_key, 0) + 1
                                if (
                                    marker_id not in history
                                    and marker_confirmations[confirmation_key] < runtime["marker_confirmations_required"]
                                ):
                                    continue

                                if marker_id not in history:
                                    history[marker_id] = {
                                        "name": emp["name"],
                                        "dept": emp["dept"],
                                        "pos_buffer": deque(maxlen=5),
                                        "path": [raw_pos.tolist()],
                                        "camera_paths": {cam_id: [raw_pos.tolist()]},
                                        "dist": 0.0,
                                        "status": "WORKING",
                                        "last_move_ts": now_ts,
                                        "last_seen_ts": now_ts,
                                        "last_account_ts": now_ts,
                                        "person_conf": matched_person["confidence"],
                                        "camera_times": {},
                                        "status_times": {"WALKING": 0.0, "WORKING": 0.0},
                                        "path_view": f"/outputs/recordings/{marker_id}_{cam_id}.mp4",
                                        "cam_id": cam_id,
                                    }

                                worker_state = history[marker_id]
                                if worker_state["cam_id"] != cam_id:
                                    # Keep total distance cumulative, but start a fresh visual path per camera.
                                    worker_state["cam_id"] = cam_id
                                    worker_state["pos_buffer"].clear()
                                    worker_state.setdefault("camera_paths", {})
                                    worker_state["camera_paths"].setdefault(cam_id, [raw_pos.tolist()])
                                    worker_state["path"] = worker_state["camera_paths"][cam_id]
                                    worker_state["last_move_ts"] = now_ts
                                    worker_state["last_seen_ts"] = now_ts
                                    worker_state["person_conf"] = matched_person["confidence"]
                                    worker_state["status"] = "WORKING"
                                    worker_state["path_view"] = f"/outputs/recordings/{marker_id}_{cam_id}.mp4"
                                    continue

                                worker_state["last_seen_ts"] = now_ts
                                worker_state["person_conf"] = matched_person["confidence"]
                                worker_state["path_view"] = f"/outputs/recordings/{marker_id}_{cam_id}.mp4"
                                worker_state.setdefault("camera_paths", {}).setdefault(cam_id, worker_state.get("path", [raw_pos.tolist()]))
                                worker_state["path"] = worker_state["camera_paths"][cam_id]
                                worker_state["pos_buffer"].append(raw_pos)
                                smooth_pos = np.mean(worker_state["pos_buffer"], axis=0)
                                last_pt = np.array(worker_state["path"][-1])
                                d_pixels = float(np.sqrt(np.sum((smooth_pos - last_pt) ** 2)))

                                if d_pixels > 8:
                                    dist_meters = d_pixels / scale
                                    dist_feet = dist_meters * 3.28084
                                    if dist_feet > 20 and len(worker_state["path"]) > 1:
                                        worker_state["pos_buffer"].clear()
                                        continue

                                    worker_state["dist"] += dist_feet
                                    worker_state["path"].append(smooth_pos.tolist())
                                    worker_state["camera_paths"][cam_id] = worker_state["path"]
                                    worker_state["status"] = "WALKING"
                                    worker_state["last_move_ts"] = now_ts
                                    worker_state["cam_id"] = cam_id
                                elif now_ts - worker_state["last_move_ts"] > 5:
                                    worker_state["status"] = "WORKING"

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
                    state["path"] = state.get("camera_paths", {}).get(current_cam, state.get("path", []))
                    state["path_view"] = f"/outputs/recordings/{mid}_{current_cam}.mp4"
                    employee_record = build_employee_record(mid, state, now_ts)
                    employee_records.append(employee_record)

                    if loop_counter % runtime["path_video_every_n_frames"] == 0:
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
                            f"{state['name']} | {current_cam} | {state['status']} | {state['dist']:.2f} ft",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
                        path_frame, _ = resize_to_width(path_frame, runtime["dashboard_image_width"])
                        path_video_recorder.write(f"outputs/recordings/{mid}_{current_cam}.mp4", path_frame)

                    frame_data["stats"].append(employee_record)

                frame_data["total_person_count"] = total_detected_people
                with open(runtime["employee_records_file"], "w") as f:
                    json.dump({
                        "timestamp": frame_data["timestamp"],
                        "employees": employee_records,
                    }, f, indent=2)
                with open("logs/live_stats.json", "w") as f:
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
        aruco_pool.shutdown(wait=False, cancel_futures=True)
        image_writer_pool.shutdown(wait=False, cancel_futures=True)
        for cam in cameras:
            cam.stop()
        if decoder_log_handle is not None:
            decoder_log_handle.close()

if __name__ == "__main__":
    main()
