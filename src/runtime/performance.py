import subprocess
import time
from typing import Dict

import psutil

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable during lightweight checks
    torch = None


class PerformanceMonitor:
    def __init__(self, interval_seconds: float = 2.0):
        self.interval_seconds = interval_seconds
        self._last_sample_ts = 0.0
        self._last = {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "gpu_memory_percent": 0.0,
            "gpu_util_percent": 0.0,
        }

    def sample(self) -> Dict[str, float]:
        now = time.time()
        if now - self._last_sample_ts < self.interval_seconds:
            return self._last

        gpu_memory_percent = 0.0
        if torch is not None and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            gpu_memory_percent = (used_bytes / total_bytes) * 100 if total_bytes else 0.0

        self._last = {
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_percent": float(psutil.virtual_memory().percent),
            "gpu_memory_percent": float(round(gpu_memory_percent, 2)),
            "gpu_util_percent": self._nvidia_gpu_util(),
        }
        self._last_sample_ts = now
        return self._last

    @staticmethod
    def _nvidia_gpu_util() -> float:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            first_value = result.stdout.strip().splitlines()[0]
            return float(first_value)
        except Exception:
            return 0.0
