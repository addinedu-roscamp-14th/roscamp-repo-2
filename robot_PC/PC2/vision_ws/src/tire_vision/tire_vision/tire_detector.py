from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tire_vision.marker_detector import detect_yellow_marker
from tire_vision.wear_measurement import (
    GeometryRecheckError,
    MarkerNotFoundError,
    MeasurementError,
    ModelInferenceError,
    TireDecisionState,
    TireNotFoundError,
    decision_state_args_from_config,
    estimate_tire_geometry_for_side,
    extract_tire_detection,
    measure_radial_gap,
)


@dataclass(frozen=True)
class TireDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class Detector(Protocol):
    def detect(self, frame, side: str | None = None) -> list[dict]:
        ...


class MockTireDetector:
    def __init__(
        self,
        label: str = "bad_tire",
        confidence: float = 0.91,
        side_results: dict[str, tuple[str, float]] | None = None,
    ):
        self.label = label
        self.confidence = float(confidence)
        self.side_results = side_results or {
            "left": ("bad_tire", 0.91),
            "right": ("good_tire", 0.88),
        }

    def detect(self, frame, side: str | None = None) -> list[dict]:
        label = self.label
        confidence = self.confidence
        if side is not None:
            label, confidence = self.side_results.get(
                side.lower().strip(), (label, confidence)
            )
        if label == "none":
            return []
        return [TireDetection(
            class_id=0 if label == "bad_tire" else 1,
            class_name=label,
            confidence=float(confidence),
            bbox=[120.0, 90.0, 320.0, 280.0],
        ).to_dict()]


class YoloTireDetector:
    """Run measurements through the shared per-camera decision state."""

    uses_decision_state = True

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.5,
        device="cpu",
        yellow_lower=(15, 40, 60),
        yellow_upper=(42, 255, 255),
        min_marker_area: float = 5,
        max_marker_area: float = 500,
        marker_roi_scale: float = 1.15,
        no_tire_reset_frames: int = 3,
        marker_hold_frames: int = 3,
        marker_recheck_frames: int = 5,
        ratio_window_size: int = 15,
        ratio_min_valid_frames: int = 10,
        bad_max_ratio: float = 0.28,
        good_min_ratio: float = 0.32,
        max_ratio_std: float = 0.02,
        side_decision_args: dict | None = None,
        model=None,
    ):
        self.model_path = str(model_path)
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.yellow_lower = tuple(int(value) for value in yellow_lower)
        self.yellow_upper = tuple(int(value) for value in yellow_upper)
        self.min_marker_area = float(min_marker_area)
        self.max_marker_area = float(max_marker_area)
        self.marker_roi_scale = float(marker_roi_scale)
        self._model = model

        state_args = {
            "window_size": ratio_window_size,
            "min_valid_frames": ratio_min_valid_frames,
            "bad_max_ratio": bad_max_ratio,
            "good_min_ratio": good_min_ratio,
            "max_ratio_std": max_ratio_std,
            "no_tire_reset_frames": no_tire_reset_frames,
            "marker_hold_frames": marker_hold_frames,
            "marker_recheck_frames": marker_recheck_frames,
        }
        side_decision_args = side_decision_args or {}
        keys = ("left", "right", "default")
        self._decision_states = {
            key: TireDecisionState(
                **{
                    **state_args,
                    **side_decision_args.get(key, {}),
                }
            )
            for key in keys
        }
        self._ratio_stabilizers = {
            key: state.ratio_stabilizer
            for key, state in self._decision_states.items()
        }
        self._last_stable_detections = {key: None for key in keys}
        self._load_model()

    def _load_model(self):
        if self._model is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def _key(self, side: str | None) -> str:
        key = str(side).lower().strip() if side is not None else "default"
        return key if key in self._decision_states else "default"

    def detect(self, frame, side: str | None = None) -> list[dict]:
        key = self._key(side)
        state = self._decision_states[key]
        try:
            detection, measurement = self._measure_frame(frame, key)
        except TireNotFoundError as error:
            snapshot = state.update_tire_detection(False, str(error))
            if snapshot["result"] == "NO_TIRE":
                self._last_stable_detections[key] = None
            return []
        except MarkerNotFoundError as error:
            state.update_tire_detection(True)
            snapshot = state.update_marker_detection(False, detail=str(error))
            held = self._last_stable_detections[key]
            if snapshot["result"] in {"GOOD", "BAD"} and held is not None:
                return [dict(held)]
            return []
        except GeometryRecheckError as error:
            state.update_recheck(str(error))
            return []
        except ModelInferenceError as error:
            state.update_error(str(error), "MODEL_ERROR")
            return []
        except MeasurementError as error:
            state.update_error(str(error))
            return []

        state.update_tire_detection(True)
        snapshot = state.update_marker_detection(
            True,
            ratio=measurement["gap_ratio"],
            gap_px=measurement.get("gap_px"),
            radius_px=measurement.get("radius_px"),
        )
        status = snapshot["result"]
        if status not in {"GOOD", "BAD"}:
            return []

        class_name = "good_tire" if status == "GOOD" else "bad_tire"
        current = TireDetection(
            class_id=1 if class_name == "good_tire" else 0,
            class_name=class_name,
            confidence=float(detection["confidence"]),
            bbox=list(detection["bbox"]),
        ).to_dict()
        self._last_stable_detections[key] = current
        return [current]

    def reset_histories(self):
        self.reset_sessions()

    def reset_sessions(self):
        for key, state in self._decision_states.items():
            state.reset_session()
            self._last_stable_detections[key] = None

    def status_for_side(self, side):
        return self._decision_states[self._key(side)].current_result

    def decision_state(self, side):
        return self._decision_states[self._key(side)].snapshot()

    def marker_diagnostics(self, side):
        return self.decision_state(side)

    def record_error(self, side, detail, error_code="DETECTION_ERROR"):
        self._decision_states[self._key(side)].update_error(detail, error_code)

    def _measure_frame(self, frame, side="default"):
        try:
            results = self._load_model()(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as error:
            raise ModelInferenceError(f"model inference failed: {error}") from error
        if not results:
            raise TireNotFoundError("tire not detected")
        detection = extract_tire_detection(
            results[0], frame.shape, self.conf_threshold
        )
        geometry = estimate_tire_geometry_for_side(detection["mask"], side)
        geometry_mask = geometry.get("mask", detection["mask"])
        marker = detect_yellow_marker(
            frame,
            geometry_mask,
            geometry["center"],
            geometry["radius"],
            self.yellow_lower,
            self.yellow_upper,
            self.min_marker_area,
            self.max_marker_area,
            self.marker_roi_scale,
        )
        measurement = measure_radial_gap(
            geometry["center"],
            geometry["radius"],
            marker["contour"],
            marker["center"],
        )
        return detection, measurement


def create_detector(config: dict) -> Detector:
    if bool(config.get("mock_mode", True)):
        side_results = {
            "left": (
                str(config.get("mock_left_class_name", "bad_tire")),
                float(config.get("mock_left_confidence", 0.91)),
            ),
            "right": (
                str(config.get("mock_right_class_name", "good_tire")),
                float(config.get("mock_right_confidence", 0.88)),
            ),
        }
        return MockTireDetector(
            label=str(config.get("mock_detection_label", "bad_tire")),
            confidence=float(config.get("mock_detection_confidence", 0.91)),
            side_results=side_results,
        )

    model_path = config.get("model_path")
    if not model_path:
        raise RuntimeError("model_path is required when mock_mode=false")
    return YoloTireDetector(
        model_path=str(model_path),
        imgsz=int(config.get("imgsz", 640)),
        conf_threshold=float(config.get("conf_threshold", 0.5)),
        device=config.get("device", "cpu"),
        yellow_lower=config.get("yellow_lower", [15, 40, 60]),
        yellow_upper=config.get("yellow_upper", [42, 255, 255]),
        min_marker_area=float(config.get("min_marker_area", 5)),
        max_marker_area=float(config.get("max_marker_area", 500)),
        marker_roi_scale=float(config.get("marker_roi_scale", 1.15)),
        no_tire_reset_frames=int(config.get("no_tire_reset_frames", 3)),
        marker_hold_frames=int(config.get("marker_hold_frames", 3)),
        marker_recheck_frames=int(config.get("marker_recheck_frames", 5)),
        ratio_window_size=int(config.get("ratio_window_size", 15)),
        ratio_min_valid_frames=int(config.get("ratio_min_valid_frames", 10)),
        bad_max_ratio=float(config.get("bad_max_ratio", 0.28)),
        good_min_ratio=float(config.get("good_min_ratio", 0.32)),
        max_ratio_std=float(config.get("max_ratio_std", 0.02)),
        side_decision_args={
            "left": decision_state_args_from_config(config, "left"),
            "right": decision_state_args_from_config(config, "right"),
        },
    )
