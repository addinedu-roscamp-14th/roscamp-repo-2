from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VehicleDetection:
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


class MockVehicleDetector:
    def __init__(self, detections: list[dict] | None = None):
        self.detections = detections or [
            VehicleDetection(
                class_id=0,
                class_name="Car_B",
                confidence=0.95,
                bbox=[285.0, 295.0, 325.0, 350.0],
            ).to_dict()
        ]

    def detect(self, frame):
        return [dict(detection) for detection in self.detections], frame


class VehicleDetector:
    """Vehicle YOLO wrapper ported from tire_ws detector.VehicleDetector."""

    def __init__(
        self,
        model_path="yolo11s.pt",
        imgsz=640,
        conf_threshold=0.4,
        vehicle_class_names=None,
        vehicle_class_ids=None,
        device=None,
    ):
        self.model_path = str(model_path)
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.vehicle_class_names = set(vehicle_class_names or ["car"])
        self.vehicle_class_ids = (
            {int(item) for item in vehicle_class_ids}
            if vehicle_class_ids is not None
            else None
        )
        self.device = device
        self.model = None
        self.names = {}
        self._load_model_once()

    def _load_model_once(self):
        if self.model is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"vehicle YOLO model not found: {self.model_path}")
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        self.names = self.model.names

    def detect(self, frame):
        kwargs = {
            "imgsz": self.imgsz,
            "conf": self.conf_threshold,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model(frame, **kwargs)
        detections = []
        if len(results) == 0:
            return detections, frame

        result = results[0]
        annotated_frame = result.plot()
        if result.boxes is None:
            return detections, annotated_frame

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.names.get(class_id, str(class_id))
            if self.vehicle_class_ids is not None:
                if class_id not in self.vehicle_class_ids:
                    continue
            elif class_name not in self.vehicle_class_names:
                continue
            detections.append(
                VehicleDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=box.xyxy[0].tolist(),
                ).to_dict()
            )
        return detections, annotated_frame


def create_vehicle_detector(config: dict):
    if bool(config.get("mock_vehicle_detector", True)):
        return MockVehicleDetector(config.get("mock_detections"))
    model_path = config.get("model_path")
    if not model_path:
        raise RuntimeError("model_path is required when mock_vehicle_detector=false")
    return VehicleDetector(
        model_path=model_path,
        imgsz=int(config.get("imgsz", 640)),
        conf_threshold=float(config.get("confidence", 0.4)),
        vehicle_class_names=config.get("vehicle_class_names"),
        vehicle_class_ids=config.get("vehicle_class_ids"),
        device=config.get("device"),
    )
