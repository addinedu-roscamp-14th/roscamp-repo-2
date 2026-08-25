"""Reusable tire segmentation geometry, radial-gap, and ratio stabilization."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MeasurementError(RuntimeError):
    """An expected per-frame measurement failure."""


def _scalar(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def extract_tire_detection(result, frame_shape, conf_threshold):
    """Select the best tire instance and return its original-size binary mask."""
    if result.boxes is None or len(result.boxes) == 0:
        raise MeasurementError("tire not detected")
    choices = []
    for index, box in enumerate(result.boxes):
        class_id = int(_scalar(box.cls))
        names = result.names
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        confidence = _scalar(box.conf)
        if name == "tire" and confidence >= conf_threshold:
            choices.append((confidence, index, box))
    if not choices:
        raise MeasurementError("tire not detected")
    if result.masks is None or getattr(result.masks, "data", None) is None:
        raise MeasurementError("result.masks is missing")

    confidence, index, box = max(choices, key=lambda item: item[0])
    if index >= len(result.masks.data):
        raise MeasurementError("mask index does not match detected tire")
    mask = result.masks.data[index]
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise MeasurementError("invalid tire mask dimensions")
    height, width = frame_shape[:2]
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (resized > 0.5).astype(np.uint8) * 255

    xyxy = box.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    bbox = [float(value) for value in np.asarray(xyxy).reshape(-1)[:4]]
    return {"confidence": confidence, "bbox": bbox, "mask": binary}


def estimate_tire_geometry(tire_mask, min_contour_area=100.0, min_radius=5.0):
    """Estimate center/radius from the largest external mask contour."""
    mask = np.asarray(tire_mask)
    if mask.ndim != 2 or mask.size == 0:
        raise MeasurementError("invalid tire mask")
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise MeasurementError("tire contour not found")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_contour_area:
        raise MeasurementError(f"tire contour is too small ({area:.1f} px^2)")
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if not np.isfinite([cx, cy, radius]).all() or radius < min_radius:
        raise MeasurementError(f"tire radius is too small ({radius:.1f} px)")
    return {
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "contour": contour,
        "area": area,
    }


def measure_radial_gap(center, radius, marker_contour, marker_center=None):
    """Measure marker outer edge to tire outer edge along the radial direction."""
    if radius <= 0 or not np.isfinite(radius):
        raise MeasurementError("invalid tire radius")
    points = np.asarray(marker_contour, np.float64).reshape(-1, 2)
    if not len(points):
        raise MeasurementError("marker contour is empty")
    tire_center = np.asarray(center, np.float64)
    if marker_center is None:
        moments = cv2.moments(points.astype(np.float32).reshape(-1, 1, 2))
        if moments["m00"] <= 0:
            raise MeasurementError("marker center cannot be calculated")
        marker_center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
    marker_center_array = np.asarray(marker_center, np.float64)
    direction = marker_center_array - tire_center
    direction_length = float(np.linalg.norm(direction))
    if direction_length < 1e-6:
        raise MeasurementError("marker center overlaps tire center")
    distances = np.linalg.norm(points - tire_center, axis=1)
    marker_outer = points[int(np.argmax(distances))]
    if float(np.max(distances)) > radius + 1e-3:
        raise MeasurementError("marker outer boundary is outside tire radius")
    tire_outer = tire_center + radius * direction / direction_length
    gap_px = float(np.linalg.norm(tire_outer - marker_outer))
    if not np.isfinite(gap_px) or gap_px < 0 or gap_px > radius:
        raise MeasurementError("radial gap is outside the valid range")
    return {
        "gap_px": gap_px,
        "radius_px": float(radius),
        "gap_ratio": gap_px / float(radius),
        "marker_center": marker_center_array.tolist(),
        "marker_outer_point": marker_outer.tolist(),
        "tire_outer_point": tire_outer.tolist(),
    }


def classify_history(history, min_valid_frames, bad_max_ratio,
                     good_min_ratio, max_ratio_std):
    values = np.asarray(list(history), np.float64)
    median = float(np.median(values)) if values.size else None
    std = float(np.std(values)) if values.size else None
    result = "RECHECK"
    if values.size >= min_valid_frames and std <= max_ratio_std:
        if median >= good_min_ratio:
            result = "GOOD"
        elif median <= bad_max_ratio:
            result = "BAD"
    return {"result": result, "median_ratio": median, "std_ratio": std}


class RatioStabilizer:
    """Keep a bounded history of valid normalized gap measurements."""

    def __init__(self, window_size=15, min_valid_frames=10,
                 bad_max_ratio=0.23, good_min_ratio=0.30,
                 max_ratio_std=0.01):
        self.window_size = max(1, int(window_size))
        self.min_valid_frames = max(1, int(min_valid_frames))
        if self.min_valid_frames > self.window_size:
            raise ValueError("min_valid_frames cannot exceed window_size")
        self.bad_max_ratio = float(bad_max_ratio)
        self.good_min_ratio = float(good_min_ratio)
        if self.bad_max_ratio >= self.good_min_ratio:
            raise ValueError("bad_max_ratio must be less than good_min_ratio")
        self.max_ratio_std = float(max_ratio_std)
        self.history = deque(maxlen=self.window_size)

    def reset(self):
        self.history.clear()

    def update(self, gap_ratio):
        ratio = float(gap_ratio)
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError("gap_ratio must be finite and non-negative")
        self.history.append(ratio)
        return self.summary()

    def summary(self):
        return classify_history(
            self.history,
            self.min_valid_frames,
            self.bad_max_ratio,
            self.good_min_ratio,
            self.max_ratio_std,
        )

