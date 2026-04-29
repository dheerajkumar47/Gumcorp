import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class ThreadedCamera:
    """Continuously read the latest frame from one camera without blocking inference."""

    def __init__(
        self,
        camera_id: str,
        video_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        reconnect_after: int = 10,
        reconnect_interval_seconds: float = 5.0,
        stale_after_seconds: float = 3.0,
    ):
        self.id = camera_id
        self.video_path = video_path
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
            }

    def _open_capture(self) -> None:
        self._last_open_attempt_ts = time.time()
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._err_count = 0
        self._reconnects += 1

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

            if self._read_ts and time.time() - self._read_ts > self.stale_after_seconds:
                self._last_error = "stale_frame_reconnect"
                self._open_capture()
                time.sleep(0.2)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
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
