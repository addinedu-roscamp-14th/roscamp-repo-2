from __future__ import annotations

import threading
import time

from tire_vision.camera_manager import CameraManager
from tire_vision.detection_stabilizer import DetectionStabilizer, StableResult
from tire_vision.health_check import HealthCheck
from tire_vision.protocol import (
    ProtocolError,
    make_error,
    make_response,
    request_id_from,
    version_from,
)
from tire_vision.tire_detector import create_detector


class VisionService:
    def __init__(
        self,
        config: dict,
        camera_manager: CameraManager | None = None,
        detector=None,
        health_check: HealthCheck | None = None,
    ):
        self.config = config
        self.camera_manager = camera_manager or CameraManager.from_config(config)
        self.detector = detector or create_detector(config)
        self.health_check = health_check or HealthCheck(config, self.camera_manager, self.detector)
        self._detect_lock = threading.Lock()
        self._stabilizers = {
            "left": self._create_stabilizer(),
            "right": self._create_stabilizer(),
        }

    def close(self) -> None:
        release = getattr(self.camera_manager, "release", None)
        if release:
            release()

    def _create_stabilizer(self, required_count: int | None = None) -> DetectionStabilizer:
        return DetectionStabilizer(
            required_count=required_count or int(self.config.get("required_stable_count", 3)),
            bad_class_name=str(self.config.get("bad_class_name", "bad_tire")),
            good_class_name=str(self.config.get("good_class_name", "good_tire")),
            window_size=int(self.config.get("detection_window_size", 3)),
            minimum_average_confidence=float(
                self.config.get("minimum_average_confidence", 0.0)
            ),
        )

    def handle_request(self, request: dict) -> dict:
        request_type = request.get("type")
        if request_type == "health":
            return self.health(request)
        if request_type == "detect_tires":
            return self.detect_tires(request)
        raise ProtocolError("UNKNOWN_REQUEST", f"unknown request type: {request_type}")

    def health(self, request: dict | None = None) -> dict:
        return make_response(
            "ok",
            version=version_from(request),
            type="health",
            request_id=request_id_from(request),
            health=self.health_check.collect().to_dict(),
        )

    def detect_tires(self, request: dict | None = None) -> dict:
        request = request or {}
        if not self._detect_lock.acquire(blocking=False):
            return make_error("BUSY", "Detection is already running", request)

        try:
            if bool(request.get("reset_cycle", False)):
                self.reset_detection_cycle()
            if request.get("side"):
                return self._detect_single_side(request)
            return self._detect_both_sides(request)
        finally:
            self._detect_lock.release()

    def _detect_single_side(self, request: dict) -> dict:
        side = self._normalize_side(request.get("side", "left"))
        side_result = self._detect_side(side, request)
        return make_response(
            "ok",
            version=version_from(request),
            type="detect_tires",
            request_id=request_id_from(request),
            side=side,
            result=side_result["class_name"],
            confidence=side_result["confidence"],
            stable=side_result["stable"],
            stable_count=side_result["votes"],
            frames=side_result["frames"],
            detections=side_result["detections"],
            detection_status=side_result["status"],
        )

    def _detect_both_sides(self, request: dict) -> dict:
        left = self._detect_side("left", request)
        right = self._detect_side("right", request)
        return make_response(
            "ok",
            version=version_from(request),
            type="detection_result",
            request_id=request_id_from(request),
            left=left,
            right=right,
        )

    def _detect_side(self, side: str, request: dict) -> dict:
        timeout_sec = float(request.get("timeout_sec", self.config.get("detect_timeout_sec", 5.0)))
        required_count = int(
            request.get("required_stable_count", self.config.get("required_stable_count", 3))
        )
        stabilizer = self._stabilizers[side]
        if stabilizer.required_count != max(1, required_count):
            stabilizer = self._create_stabilizer(required_count)
            self._stabilizers[side] = stabilizer
        stabilizer.reset()

        deadline = time.monotonic() + max(0.1, timeout_sec)
        loop_sleep = float(self.config.get("detect_loop_sleep_sec", 0.01))
        frames = 0
        last_detections = []
        stable = StableResult("none", 0.0, 0, False)

        while time.monotonic() < deadline:
            frame = self._capture_frame(side)
            frames += 1
            last_detections = self._run_detector(frame, side)
            stable = stabilizer.update(last_detections)
            if stable.stable:
                break
            time.sleep(loop_sleep)

        return {
            "status": self._status_for_side(side, stable),
            "class_name": stable.label if stable.stable else "none",
            "confidence": stable.confidence if stable.stable else 0.0,
            "stable": stable.stable,
            "votes": stable.stable_count,
            "frames": frames,
            "detections": [] if bool(self.config.get("mock_mode", True)) else last_detections,
        }

    def reset_detection_cycle(self):
        reset_histories = getattr(self.detector, "reset_histories", None)
        if reset_histories:
            reset_histories()
        for stabilizer in self._stabilizers.values():
            stabilizer.reset()

    def _status_for_side(self, side, stable):
        if stable.stable:
            return "GOOD" if stable.label == "good_tire" else "BAD"
        status_for_side = getattr(self.detector, "status_for_side", None)
        status = status_for_side(side) if status_for_side else "RECHECK"
        return status if status in {"GOOD", "BAD", "RECHECK", "ERROR"} else "ERROR"

    def _capture_frame(self, side: str):
        try:
            return self.camera_manager.capture_frame(side)
        except TypeError:
            return self.camera_manager.capture_frame()

    def _run_detector(self, frame, side: str) -> list[dict]:
        try:
            return self.detector.detect(frame, side=side)
        except TypeError:
            return self.detector.detect(frame)

    @staticmethod
    def _normalize_side(value) -> str:
        side = str(value).lower().strip()
        if side not in ("left", "right"):
            raise ProtocolError("INVALID_SIDE", f"unknown tire side: {side}")
        return side
