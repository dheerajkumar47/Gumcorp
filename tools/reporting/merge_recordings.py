import argparse
import re
from pathlib import Path

import cv2


SEGMENT_RE = re.compile(r"^(?P<base>.+)_(?P<stamp>\d{6})\.mp4$", re.IGNORECASE)


def is_segment(path: Path) -> bool:
    return bool(SEGMENT_RE.match(path.name))


def base_output_path(path: Path, recording_root: Path) -> Path:
    match = SEGMENT_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a segment file: {path}")
    base_name = f"{match.group('base')}.mp4"
    if path.parent.name.lower() == "proof":
        return path.parent / base_name
    return recording_root / base_name


def grouped_segments(recording_root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for path in recording_root.rglob("*.mp4"):
        if ".part." in path.name.lower() or not is_segment(path):
            continue
        output_path = base_output_path(path, recording_root)
        if path.resolve() == output_path.resolve():
            continue
        groups.setdefault(output_path, []).append(path)
    for output_path in groups:
        groups[output_path].sort(key=lambda item: item.name)
    return groups


def open_writer(output_path: Path, fps: float, frame_shape, codec: str):
    height, width = frame_shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.mergepart{output_path.suffix}")
    temp_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*codec[:4]),
        max(0.5, float(fps) if fps else 2.5),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {temp_path}")
    return writer, temp_path


def merge_group(output_path: Path, segments: list[Path], codec: str) -> tuple[int, int]:
    writer = None
    temp_path = None
    frame_count = 0
    source_count = 0
    target_size = None
    target_fps = 2.5

    try:
        for segment in segments:
            cap = cv2.VideoCapture(str(segment))
            if not cap.isOpened():
                cap.release()
                continue
            source_count += 1
            fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    target_fps = fps
                    target_size = frame.shape[:2]
                    writer, temp_path = open_writer(output_path, target_fps, frame.shape, codec)
                elif target_size is not None and frame.shape[:2] != target_size:
                    frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)
                frame_count += 1
            cap.release()
    finally:
        if writer is not None:
            writer.release()

    if frame_count > 0 and temp_path is not None and temp_path.exists() and temp_path.stat().st_size > 1024:
        temp_path.replace(output_path)
    elif temp_path is not None:
        temp_path.unlink(missing_ok=True)
    return source_count, frame_count


def merge_recordings(recording_root: Path | str, codec: str = "mp4v") -> list[dict]:
    recording_root = Path(recording_root)
    if not recording_root.exists():
        return []
    results = []
    for output_path, segments in grouped_segments(recording_root).items():
        source_count, frame_count = merge_group(output_path, segments, codec)
        results.append(
            {
                "output": str(output_path),
                "segments": source_count,
                "frames": frame_count,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Factory AI segmented employee/proof recordings")
    parser.add_argument("--recording-root", required=True)
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()
    results = merge_recordings(args.recording_root, args.codec)
    for result in results:
        print(f"{result['output']} | segments={result['segments']} frames={result['frames']}")


if __name__ == "__main__":
    main()
