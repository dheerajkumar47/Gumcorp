import argparse
import json
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import cv2
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_camera_source(cam, profile_name):
    stream_profiles = cam.get("stream_profiles", {})
    if profile_name in stream_profiles:
        return stream_profiles[profile_name]
    return cam.get("video_path")


def redact_url(url):
    parsed = urlsplit(url)
    if "@" not in parsed.netloc:
        return url
    host = parsed.netloc.rsplit("@", 1)[1]
    return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, parsed.fragment))


def probe_camera(camera_id, url):
    started = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    opened = cap.isOpened()
    ret, frame = cap.read() if opened else (False, None)
    shape = list(frame.shape[:2]) if ret and frame is not None else None
    cap.release()
    return {
        "id": camera_id,
        "opened": bool(opened),
        "read_frame": bool(ret and frame is not None),
        "shape": shape,
        "seconds": round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe configured RTSP cameras")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--camera-profile", default=None)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--probe-id", default=None)
    parser.add_argument("--probe-url", default=None)
    args = parser.parse_args()

    if args.probe_id and args.probe_url:
        result = probe_camera(args.probe_id, args.probe_url)
        print(json.dumps(result))
        return

    config = load_config(args.config)
    profile = args.camera_profile or config.get("runtime", {}).get("camera_profile", "main_stream")

    for cam in [c for c in config["cameras"] if c.get("enabled", False)]:
        camera_id = cam["id"]
        url = resolve_camera_source(cam, profile)
        command = [sys.executable, __file__, "--probe-id", camera_id, "--probe-url", url]

        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"{camera_id}: TIMEOUT after {args.timeout}s | {redact_url(url)}")
            continue

        lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
        if not lines:
            print(f"{camera_id}: FAILED no result | {redact_url(url)}")
            continue

        result = json.loads(lines[-1])
        status = "OK" if result["opened"] and result["read_frame"] else "FAILED"
        print(
            f"{camera_id}: {status} "
            f"opened={result['opened']} read_frame={result['read_frame']} "
            f"shape={result['shape']} seconds={result['seconds']} | {redact_url(url)}"
        )


if __name__ == "__main__":
    main()
