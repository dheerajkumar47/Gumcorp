import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np


_OPEN_CAPTURE_SEMAPHORE = threading.Semaphore(1)


class ThreadedCamera:
    """Continuously read the latest frame from one camera without blocking inference."""

    def __init__(
        self,
        camera_id: str,
        video_path: str,
        video_playlist: Optional[Sequence[str]] = None,
        video_start_offset_seconds: float = 0.0,
        replay_speed: float = 1.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        reconnect_after: int = 10,
        reconnect_interval_seconds: float = 5.0,
        stale_after_seconds: float = 3.0,
    ):
        self.id = camera_id
        self.video_path = video_path
        self.video_playlist = [str(path) for path in (video_playlist or []) if str(path).strip()]
        if not self.video_playlist and video_path:
            self.video_playlist = [str(video_path)]
        self._source_index = 0
        self._start_offset_seconds = max(0.0, float(video_start_offset_seconds or 0.0))
        self.replay_speed = max(0.1, float(replay_speed or 1.0))
        self._replay_stride = max(1, int(round(self.replay_speed)))
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_after = reconnect_after
        self.reconnect_interval_seconds = reconnect_interval_seconds
        self.stale_after_seconds = stale_after_seconds

        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._read_ts = 0.0
        self._err_count = 0
        self._frames_read = 0
        self._reconnects = 0
        self._last_error = ""
        self._last_open_attempt_ts = 0.0
        self._opened_ts = 0.0
        self._running = False
        self._thread = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, self._read_ts
            return self._frame.copy(), self._read_ts

    def status(self) -> dict:
        with self._lock:
            return {
                "frames_read": self._frames_read,
                "reconnects": self._reconnects,
                "err_count": self._err_count,
                "last_error": self._last_error,
                "has_frame": self._frame is not None,
                "seconds_since_frame": round(time.time() - self._read_ts, 2) if self._read_ts else None,
                "source": self._current_source(),
                "playlist_index": self._source_index if len(self.video_playlist) > 1 else None,
                "playlist_size": len(self.video_playlist) if len(self.video_playlist) > 1 else None,
                "replay_speed": self.replay_speed if self._is_file_source(self._current_source()) else None,
                "replay_stride": self._replay_stride if self._is_file_source(self._current_source()) else None,
            }

    def _current_source(self) -> str:
        if self.video_playlist:
            return self.video_playlist[self._source_index % len(self.video_playlist)]
        return self.video_path

    def _is_file_source(self, source: str) -> bool:
        parsed = urlparse(str(source))
        if parsed.scheme and parsed.scheme.lower() not in ("file",):
            return False
        return Path(str(source)).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".m4v"}

    def _advance_playlist(self) -> bool:
        if len(self.video_playlist) <= 1:
            return False
        self._source_index = (self._source_index + 1) % len(self.video_playlist)
        self._open_capture()
        return True

    def _open_capture(self) -> None:
        self._last_open_attempt_ts = time.time()
        if self._cap is not None:
            self._cap.release()

        self.video_path = self._current_source()
        with _OPEN_CAPTURE_SEMAPHORE:
            self._cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self._source_index == 0 and self._start_offset_seconds:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, self._start_offset_seconds * 1000.0)
        self._err_count = 0
        self._reconnects += 1
        self._opened_ts = time.time()
        time.sleep(0.35)

    def _reader(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._last_error = "capture_not_open"
                if time.time() - self._last_open_attempt_ts < self.reconnect_interval_seconds:
                    time.sleep(0.2)
                    continue
                self._open_capture()
                time.sleep(0.2)
                continue

            now = time.time()
            if (
                self._read_ts
                and now - self._read_ts > self.stale_after_seconds
                and now - self._opened_ts > self.stale_after_seconds
            ):
                self._last_error = "stale_frame_reconnect"
                self._open_capture()
                time.sleep(self.reconnect_interval_seconds)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self._is_file_source(self.video_path) and self._advance_playlist():
                    self._last_error = "playlist_advanced"
                    time.sleep(0.05)
                    continue
                self._err_count += 1
                self._last_error = "read_failed"
                if self._err_count >= self.reconnect_after:
                    self._open_capture()
                time.sleep(0.05)
                continue

            mean_pixel = float(np.mean(frame))
            if mean_pixel < 5 or mean_pixel > 250:
                self._err_count += 1
                self._last_error = "invalid_brightness"
                if self._err_count >= self.reconnect_after:
                    self._open_capture()
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame
                self._read_ts = time.time()
                self._frames_read += 1
                self._last_error = ""
            self._err_count = 0
            if self._is_file_source(self.video_path) and self.fps:
                if self._replay_stride > 1 and self._cap is not None:
                    next_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES) + self._replay_stride - 1
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                time.sleep(max(0.001, 1.0 / float(self.fps)))
