import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "data" / "videos"
OUT_DIR = ROOT / "data" / "video_replay"
REPORT_DIR = ROOT / "outputs" / "video_replay"

CLIP_RE = re.compile(r"^D(?P<channel>\d+)_S(?P<start>\d{14})_E(?P<end>\d{14}|0)\.mp4$", re.IGNORECASE)

CHANNEL_CAMERA_MAP = {
    18: "production_outside",
    27: "main_production",
    29: "coolingroom2",
    30: "palletline",
    31: "sheetline",
    32: "mixer1",
    33: "mixer2",
    34: "mixer3",
}

CAMERA_LABELS = {
    "production_outside": "Production Outside",
    "production2": "GC Production 2",
    "main_production": "Main Production",
    "coolingroom2": "Cooling Room 2",
    "palletline": "Pallet Line",
    "sheetline": "Sheet Line",
    "mixer1": "Mixer 1",
    "mixer2": "Mixer 2",
    "mixer3": "Mixer 3",
    "mixer4": "Mixer 4",
}

CAMERA_POSITIONS = {
    "sheetline": [3137, 660],
    "palletline": [2994, 1654],
    "coolingroom2": [2668, 661],
    "mixer1": [3617, 1140],
    "mixer2": [3620, 1169],
    "mixer3": [3617, 1657],
    "mixer4": [3617, 840],
    "production2": [2374, 1657],
    "production_outside": [2966, 1660],
    "main_production": [2996, 1169],
}


def parse_time(value):
    if value == "0":
        return None
    return datetime.strptime(value, "%Y%m%d%H%M%S")


def rel(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def inspect_video(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"opened": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "opened": True,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}" if width and height else "",
    }


def sample_frame(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 30:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, max(10, frame_count // 4)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def make_contact_sheet(groups, output_path):
    tiles = []
    for channel, clips in sorted(groups.items()):
        first = clips[0]
        frame = sample_frame(first["path"])
        if frame is None:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (420, 236))
        camera_id = CHANNEL_CAMERA_MAP.get(channel, "unknown")
        title = f"D{channel} -> {camera_id} | {len(clips)} clips"
        cv2.rectangle(frame, (0, 0), (420, 34), (0, 0, 0), -1)
        cv2.putText(frame, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(frame)
    if not tiles:
        return
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    blank = np.full_like(tiles[0], 245)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    row_images = [np.hstack(tiles[idx:idx + cols]) for idx in range(0, len(tiles), cols)]
    sheet = np.vstack(row_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def scan_videos(video_dir):
    groups = {}
    for path in video_dir.glob("*.mp4"):
        match = CLIP_RE.match(path.name)
        if not match:
            continue
        channel = int(match.group("channel"))
        start = parse_time(match.group("start"))
        end = parse_time(match.group("end"))
        groups.setdefault(channel, []).append(
            {
                "path": path,
                "file": rel(path),
                "channel": channel,
                "start": start,
                "end": end,
                "start_text": start.isoformat(sep=" ") if start else "",
                "end_text": end.isoformat(sep=" ") if end else "",
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        )
    for clips in groups.values():
        clips.sort(key=lambda item: item["start"] or datetime.min)
    return groups


def build_registry(groups):
    cameras = []
    for channel, clips in sorted(groups.items()):
        camera_id = CHANNEL_CAMERA_MAP.get(channel)
        if not camera_id:
            continue
        playlist = [clip["file"] for clip in clips]
        cameras.append(
            {
                "id": camera_id,
                "label": CAMERA_LABELS.get(camera_id, camera_id),
                "zone": "",
                "role": camera_id,
                "enabled": True,
                "replay": True,
                "video_path": playlist[0],
                "video_playlist": playlist,
                "video_start_offset_seconds": round(float(clips[0].get("start_offset_seconds", 0.0)), 2),
                "stream_profiles": {
                    "main_stream": playlist[0],
                    "sub_stream": playlist[0],
                },
                "position": CAMERA_POSITIONS.get(camera_id, [0, 0]),
                "height_ft": None,
                "notes": f"Replay from NVR channel D{channel}",
                "calibration_points": {"camera_points": [], "map_points": []},
            }
        )
    return {
        "version": 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "video_replay",
        "cameras": cameras,
    }


def build_manifest(groups):
    channels = []
    for channel, clips in sorted(groups.items()):
        inspected = inspect_video(clips[0]["path"]) if clips else {}
        total_seconds = 0.0
        for clip in clips:
            if clip["start"] and clip["end"]:
                total_seconds += (clip["end"] - clip["start"]).total_seconds()
        channels.append(
            {
                "channel": channel,
                "camera_id": CHANNEL_CAMERA_MAP.get(channel, "unknown"),
                "clip_count": len(clips),
                "first_start": clips[0]["start_text"] if clips else "",
                "last_end": clips[-1]["end_text"] if clips else "",
                "total_hours_from_names": round(total_seconds / 3600, 3),
                "sample_resolution": inspected.get("resolution", ""),
                "sample_fps": inspected.get("fps", 0),
                "first_clip_offset_seconds": round(float(clips[0].get("start_offset_seconds", 0.0)), 2) if clips else 0.0,
                "clips": [
                    {
                        "file": clip["file"],
                        "start": clip["start_text"],
                        "end": clip["end_text"],
                        "size_mb": clip["size_mb"],
                    }
                    for clip in clips
                ],
            }
        )
    return {"generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "channels": channels}


def write_report(manifest, path):
    lines = ["# Video Replay Inventory", ""]
    for channel in manifest["channels"]:
        lines.append(
            f"- D{channel['channel']} -> {channel['camera_id']}: "
            f"{channel['clip_count']} clips, {channel['total_hours_from_names']} hours, "
            f"{channel['first_start']} to {channel['last_end']}, "
            f"{channel['sample_resolution']} @ {channel['sample_fps']} fps, "
            f"first offset {channel['first_clip_offset_seconds']} sec"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_datetime_arg(value):
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Invalid datetime: {value}")


def apply_time_window(groups, shift_start=None, shift_end=None, excluded_channels=None):
    excluded_channels = set(excluded_channels or [])
    filtered = {}
    for channel, clips in groups.items():
        if channel in excluded_channels:
            continue
        selected = []
        for clip in clips:
            clip_start = clip["start"]
            clip_end = clip["end"]
            if shift_start and clip_end and clip_end <= shift_start:
                continue
            if shift_end and clip_start and clip_start >= shift_end:
                continue
            item = dict(clip)
            item["start_offset_seconds"] = 0.0
            if shift_start and clip_start and clip_end and clip_start <= shift_start < clip_end:
                item["start_offset_seconds"] = (shift_start - clip_start).total_seconds()
            selected.append(item)
        if selected:
            filtered[channel] = selected
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Build local video replay profile from NVR clips")
    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="Folder containing Dxx_S..._E... mp4 files")
    parser.add_argument("--shift-start", type=parse_datetime_arg, default=None, help="Replay start time, e.g. 2026-04-29 08:00:00")
    parser.add_argument("--shift-end", type=parse_datetime_arg, default=None, help="Replay end time, e.g. 2026-04-29 19:00:00")
    parser.add_argument("--exclude-channel", type=int, action="append", default=[], help="NVR channel to exclude, e.g. 29")
    args = parser.parse_args()

    groups = scan_videos(Path(args.video_dir))
    groups = apply_time_window(groups, args.shift_start, args.shift_end, args.exclude_channel)
    if not groups:
        raise SystemExit(f"No NVR mp4 clips found in {args.video_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(groups)
    registry = build_registry(groups)
    make_contact_sheet(groups, REPORT_DIR / "channel_contact_sheet.jpg")
    write_report(manifest, REPORT_DIR / "video_replay_inventory.md")

    (OUT_DIR / "replay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT_DIR / "cameras_replay.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Channels found: {', '.join('D' + str(ch) for ch in sorted(groups))}")
    print(f"Replay cameras: {', '.join(cam['id'] for cam in registry['cameras'])}")
    print(f"Registry: {rel(OUT_DIR / 'cameras_replay.json')}")
    print(f"Inventory: {rel(REPORT_DIR / 'video_replay_inventory.md')}")
    print(f"Contact sheet: {rel(REPORT_DIR / 'channel_contact_sheet.jpg')}")


if __name__ == "__main__":
    main()
