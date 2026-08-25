from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    mock_mode: bool = True


class Camera(Protocol):
    def read(self):
        ...

    def release(self) -> None:
        ...


class FakeCamera:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = int(width)
        self.height = int(height)
        self.released = False

    def read(self):
        try:
            import numpy as np

            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        except Exception:
            frame = {"width": self.width, "height": self.height, "mock": True}
        return True, frame

    def release(self) -> None:
        self.released = True


class OpenCvCamera:
    def __init__(self, config: CameraConfig):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(config.index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera open failed: index={config.index}")

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


class CameraManager:
    def __init__(self, configs: dict[str, CameraConfig]):
        self.configs = configs
        self._cameras: dict[str, Camera] = {}
        self._reconnect_counts = {side: 0 for side in configs}
        self.open_all()

    @classmethod
    def from_config(cls, config: dict) -> "CameraManager":
        mock_mode = bool(config.get("mock_mode", True))
        configs = {}
        for side, default_index in (("left", 0), ("right", 1)):
            values = config.get(f"{side}_camera", {}) or {}
            configs[side] = CameraConfig(
                index=int(values.get("index", config.get("camera_index", default_index))),
                width=int(values.get("width", config.get("camera_width", 640))),
                height=int(values.get("height", config.get("camera_height", 480))),
                fps=int(values.get("fps", config.get("camera_fps", 15))),
                mock_mode=mock_mode,
            )
        return cls(configs)

    def _open(self, side: str) -> Camera:
        config = self.configs[side]
        if config.mock_mode:
            return FakeCamera(config.width, config.height)
        return OpenCvCamera(config)

    def open_all(self) -> None:
        for side in ("left", "right"):
            if side not in self._cameras:
                self._cameras[side] = self._open(side)

    def is_available(self, side: str) -> bool:
        return side in self._cameras

    def capture_frame(self, side: str = "left"):
        side = str(side).lower().strip()
        if side not in self.configs:
            raise ValueError(f"unknown camera side: {side}")
        camera = self._cameras.get(side)
        if camera is None:
            camera = self._open(side)
            self._cameras[side] = camera
        ok, frame = camera.read()
        if ok:
            return frame

        camera.release()
        self._cameras.pop(side, None)
        camera = self._open(side)
        self._reconnect_counts[side] += 1
        self._cameras[side] = camera
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError(f"{side} camera frame read failed after reconnect")
        return frame

    def reconnect_count(self, side: str) -> int:
        return self._reconnect_counts.get(str(side).lower().strip(), 0)

    def release(self) -> None:
        for camera in self._cameras.values():
            camera.release()
        self._cameras.clear()
