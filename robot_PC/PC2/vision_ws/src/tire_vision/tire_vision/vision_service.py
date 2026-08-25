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
        self.health_check = health_check or HealthCheck(
            config, self.camera_manager, self.detector
        )
        self._detect_lock = threading.Lock()
        self._stabilizers = {
            "left": self._create_stabilizer(),
            "right": self._create_stabilizer(),
        }
        self._camera_reconnect_counts = {
            side: self._camera_reconnect_count(side)
            for side in ("left", "right")
        }
        self._camera_reconnected_during_cycle = False

    def close(self) -> None:
        release = getattr(self.camera_manager, "release", None)
        if release:
            release()

    def _create_stabilizer(self, required_count: int | None = None) -> DetectionStabilizer:
        return DetectionStabilizer(
            required_count=required_count or int(
                self.config.get("required_stable_count", 3)
            ),
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
            # Every request is a new physical inspection. This also resets both
            # sides when central retries because either side returned RECHECK.
            self._camera_reconnected_during_cycle = False
            self.reset_detection_cycle()
            if request.get("side"):
                return self._detect_single_side(request)
            return self._detect_both_sides(request)
        finally:
            self._detect_lock.release()

    def _detect_single_side(self, request: dict) -> dict:
        side = self._normalize_side(request.get("side", "left"))
        side_result = self._detect_side(side, request)
        fields = {
            "version": version_from(request),
            "type": "detect_tires",
            "request_id": request_id_from(request),
            "side": side,
            # Existing compatibility fields.
            "result": side_result["class_name"],
            "confidence": side_result["confidence"],
            "stable": side_result["stable"],
            "stable_count": side_result["votes"],
            "frames": side_result["frames"],
            "detections": side_result["detections"],
            "detection_status": side_result["status"],
            # Additive state details.
            "median_ratio": side_result["median_ratio"],
            "std_ratio": side_result["std_ratio"],
            "valid_frames": side_result["valid_frames"],
            "session_id": side_result["session_id"],
            "marker_state": side_result["marker_state"],
        }
        if side_result.get("error_code"):
            fields["error_code"] = side_result["error_code"]
            fields["message"] = side_result.get("message")
        return make_response("ok", **fields)

    def _detect_both_sides(self, request: dict) -> dict:
        max_retries = max(
            0,
            int(
                request.get(
                    "max_retries",
                    self.config.get("max_detection_retries", 3),
                )
            ),
        )
        retry_count = 0
        retries_exhausted = False

        while True:
            self._camera_reconnected_during_cycle = False
            left = self._detect_side("left", request)
            right = self._detect_side("right", request)
            overall = self._overall_status(left, right)
            if self._camera_reconnected_during_cycle:
                overall = "RECHECK"
            if overall != "RECHECK":
                break
            if retry_count >= max_retries:
                retries_exhausted = True
                overall = "ERROR"
                break
            retry_count += 1
            # A retry is a new paired inspection. Never carry either side's
            # result/history into the next attempt.
            self.reset_detection_cycle()

        fields = {
            "version": version_from(request),
            "type": "detection_result",
            "request_id": request_id_from(request),
            "left": left,
            "right": right,
            "detection_status": overall,
            "retry_count": retry_count,
            "attempts": retry_count + 1,
        }
        if retries_exhausted:
            fields["error_code"] = "DETECTION_ERROR"
            fields["message"] = "paired tire detection retry limit exceeded"
            fields["reason"] = "tire or marker result is not stable"
        elif overall == "ERROR":
            failed = next(
                side for side in (left, right)
                if side["status"] in {"NO_TIRE", "ERROR"}
            )
            fields["error_code"] = failed.get("error_code") or "DETECTION_ERROR"
            fields["message"] = failed.get("message") or "tire detection failed"
        return make_response("ok", **fields)

    @staticmethod
    def _overall_status(left, right):
        statuses = {left["status"], right["status"]}
        if statuses & {"NO_TIRE", "ERROR"}:
            return "ERROR"
        if statuses & {"COLLECTING", "RECHECK"}:
            return "RECHECK"
        return "OK"

    def _detect_side(self, side: str, request: dict) -> dict:
        timeout_sec = float(
            request.get("timeout_sec", self.config.get("detect_timeout_sec", 5.0))
        )
        required_count = int(
            request.get(
                "required_stable_count",
                self.config.get("required_stable_count", 3),
            )
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
        uses_decision_state = bool(
            getattr(self.detector, "uses_decision_state", False)
            and callable(getattr(self.detector, "decision_state", None))
        )

        while time.monotonic() < deadline:
            try:
                frame = self._capture_frame(side)
            except Exception as error:
                recorder = getattr(self.detector, "record_error", None)
                if recorder:
                    recorder(side, f"camera capture failed: {error}", "CAMERA_ERROR")
                break
            frames += 1
            last_detections = self._run_detector(frame, side)
            if uses_decision_state:
                state = self._decision_state(side)
                status = state.get("result", "ERROR")
                if status in {"GOOD", "BAD"}:
                    expected = "good_tire" if status == "GOOD" else "bad_tire"
                    selected = next(
                        (
                            item
                            for item in last_detections
                            if item.get("class_name") == expected
                        ),
                        None,
                    )
                    if selected is not None:
                        stable = StableResult(
                            expected,
                            float(selected.get("confidence", 0.0)),
                            int(state.get("valid_count", 0)),
                            True,
                        )
                        break
                if status in {"NO_TIRE", "ERROR"}:
                    break
            else:
                stable = stabilizer.update(last_detections)
                status = self._detector_status(side, stable)
                if stable.stable or status in {"NO_TIRE", "ERROR"}:
                    break
            time.sleep(loop_sleep)

        state = self._decision_state(side)
        status = self._detector_status(side, stable)
        return {
            "status": status,
            # Additive state-machine field; class_name remains for protocol
            # compatibility with existing central-control clients.
            "result": status,
            "class_name": stable.label if stable.stable else "none",
            "confidence": stable.confidence if stable.stable else 0.0,
            "stable": stable.stable,
            "votes": stable.stable_count,
            "frames": frames,
            "detections": (
                [] if bool(self.config.get("mock_mode", True)) else last_detections
            ),
            "median_ratio": state.get("median_ratio"),
            "std_ratio": state.get("std_ratio"),
            "valid_frames": state.get("valid_count", 0),
            "session_id": state.get("session_id"),
            "error_code": state.get("error_code"),
            "message": state.get("detail", ""),
            "marker_state": state,
        }

    def reset_detection_cycle(self):
        reset_sessions = getattr(self.detector, "reset_sessions", None)
        reset_histories = getattr(self.detector, "reset_histories", None)
        if reset_sessions:
            reset_sessions()
        elif reset_histories:
            reset_histories()
        for stabilizer in self._stabilizers.values():
            stabilizer.reset()

    def _decision_state(self, side):
        getter = getattr(self.detector, "decision_state", None)
        if getter:
            return getter(side)
        marker_diagnostics = getattr(self.detector, "marker_diagnostics", None)
        return marker_diagnostics(side) if marker_diagnostics else {}

    def _detector_status(self, side, stable):
        if stable.stable:
            return "GOOD" if stable.label == "good_tire" else "BAD"
        status_for_side = getattr(self.detector, "status_for_side", None)
        status = status_for_side(side) if status_for_side else "RECHECK"
        allowed = {"NO_TIRE", "COLLECTING", "GOOD", "BAD", "RECHECK", "ERROR"}
        return status if status in allowed else "ERROR"

    def _camera_reconnect_count(self, side: str) -> int:
        getter = getattr(self.camera_manager, "reconnect_count", None)
        return int(getter(side)) if getter else 0

    def _capture_frame(self, side: str):
        previous = self._camera_reconnect_counts.get(side, 0)
        try:
            frame = self.camera_manager.capture_frame(side)
        except TypeError:
            frame = self.camera_manager.capture_frame()
        current = self._camera_reconnect_count(side)
        if current > previous:
            self._camera_reconnect_counts[side] = current
            self._camera_reconnected_during_cycle = True
            # Reconnection invalidates both cameras' pending paired result.
            self.reset_detection_cycle()
        return frame

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
