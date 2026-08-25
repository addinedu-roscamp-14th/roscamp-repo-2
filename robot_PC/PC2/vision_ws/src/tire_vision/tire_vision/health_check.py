from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    mock_mode: bool
    model_available: bool
    left_camera_available: bool
    right_camera_available: bool
    uptime_sec: float

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mock_mode": self.mock_mode,
            "model_available": self.model_available,
            "left_camera_available": self.left_camera_available,
            "right_camera_available": self.right_camera_available,
            "uptime_sec": round(self.uptime_sec, 3),
        }


class HealthCheck:
    def __init__(self, config: dict, camera_manager=None, detector=None, started_at: float | None = None):
        self.config = config
        self.started_at = time.monotonic() if started_at is None else float(started_at)
        self.camera_manager = camera_manager
        self.detector = detector

    def collect(self) -> HealthStatus:
        mock_mode = bool(self.config.get("mock_mode", True))
        model_path = self.config.get("model_path")
        model_available = bool(model_path and Path(str(model_path)).is_file())
        camera_available = lambda side: bool(
            mock_mode or (self.camera_manager and self.camera_manager.is_available(side))
        )
        return HealthStatus(
            ok=mock_mode or model_available,
            mock_mode=mock_mode,
            model_available=model_available,
            left_camera_available=camera_available("left"),
            right_camera_available=camera_available("right"),
            uptime_sec=time.monotonic() - self.started_at,
        )
