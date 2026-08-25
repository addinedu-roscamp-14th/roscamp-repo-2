from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
                side.lower().strip(),
                (label, confidence),
            )
        if label == "none":
            return []
        return [
            TireDetection(
                class_id=0 if label == "bad_tire" else 1,
                class_name=label,
                confidence=float(confidence),
                bbox=[120.0, 90.0, 320.0, 280.0],
            ).to_dict()
        ]


class YoloTireDetector:
    """Convert tire segmentation and marker wear measurements to legacy labels."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.5,
        device="cpu",
        yellow_lower=(15, 80, 80),
        yellow_upper=(40, 255, 255),
        min_marker_area: float = 5,
        max_marker_area: float = 500,
        ratio_window_size: int = 15,
        ratio_min_valid_frames: int = 10,
        bad_max_ratio: float = 0.23,
        good_min_ratio: float = 0.30,
        max_ratio_std: float = 0.01,
        model=None,
    ):
        from tire_vision.marker_detector import detect_yellow_marker
        from tire_vision.wear_measurement import (
            MeasurementError,
            RatioStabilizer,
            estimate_tire_geometry,
            extract_tire_detection,
            measure_radial_gap,
        )

        self.model_path = str(model_path)
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.yellow_lower = tuple(int(value) for value in yellow_lower)
        self.yellow_upper = tuple(int(value) for value in yellow_upper)
        self.min_marker_area = float(min_marker_area)
        self.max_marker_area = float(max_marker_area)
        self._measurement_error = MeasurementError
        self._extract_tire_detection = extract_tire_detection
        self._estimate_tire_geometry = estimate_tire_geometry
        self._detect_yellow_marker = detect_yellow_marker
        self._measure_radial_gap = measure_radial_gap
        self._model = model
        stabilizer_args = {
            "window_size": ratio_window_size,
            "min_valid_frames": ratio_min_valid_frames,
            "bad_max_ratio": bad_max_ratio,
            "good_min_ratio": good_min_ratio,
            "max_ratio_std": max_ratio_std,
        }
        self._ratio_stabilizers = {
            "left": RatioStabilizer(**stabilizer_args),
            "right": RatioStabilizer(**stabilizer_args),
            "default": RatioStabilizer(**stabilizer_args),
        }
        self._last_status = {"left": "RECHECK", "right": "RECHECK", "default": "RECHECK"}
        self._load_model()

    def _load_model(self):
        if self._model is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame, side: str | None = None) -> list[dict]:
        history_key = str(side).lower().strip() if side is not None else "default"
        if history_key not in self._ratio_stabilizers:
            history_key = "default"
        try:
            detection, measurement = self._measure_frame(frame)
        except self._measurement_error:
            self._last_status[history_key] = "ERROR"
            return []
        stabilizer = self._ratio_stabilizers[history_key]
        summary = stabilizer.update(measurement["gap_ratio"])
        self._last_status[history_key] = summary["result"]
        if summary["result"] == "RECHECK":
            return []
        class_name = "good_tire" if summary["result"] == "GOOD" else "bad_tire"
        return [TireDetection(
            class_id=1 if class_name == "good_tire" else 0,
            class_name=class_name,
            confidence=float(detection["confidence"]),
            bbox=list(detection["bbox"]),
        ).to_dict()]

    def reset_histories(self):
        for stabilizer in self._ratio_stabilizers.values():
            stabilizer.reset()
        for side in self._last_status:
            self._last_status[side] = "RECHECK"

    def status_for_side(self, side):
        key = str(side).lower().strip()
        return self._last_status.get(key, "RECHECK")

    def _measure_frame(self, frame):
        model = self._load_model()
        results = model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            raise MeasurementError("YOLO returned no result")
        detection = self._extract_tire_detection(
            results[0], frame.shape, self.conf_threshold)
        geometry = self._estimate_tire_geometry(detection["mask"])
        marker = self._detect_yellow_marker(
            frame,
            detection["mask"],
            geometry["center"],
            geometry["radius"],
            self.yellow_lower,
            self.yellow_upper,
            self.min_marker_area,
            self.max_marker_area,
        )
        measurement = self._measure_radial_gap(
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
        yellow_lower=config.get("yellow_lower", [15, 80, 80]),
        yellow_upper=config.get("yellow_upper", [40, 255, 255]),
        min_marker_area=float(config.get("min_marker_area", 5)),
        max_marker_area=float(config.get("max_marker_area", 500)),
        ratio_window_size=int(config.get("ratio_window_size", 15)),
        ratio_min_valid_frames=int(config.get("ratio_min_valid_frames", 10)),
        bad_max_ratio=float(config.get("bad_max_ratio", 0.23)),
        good_min_ratio=float(config.get("good_min_ratio", 0.30)),
        max_ratio_std=float(config.get("max_ratio_std", 0.01)),
    )
