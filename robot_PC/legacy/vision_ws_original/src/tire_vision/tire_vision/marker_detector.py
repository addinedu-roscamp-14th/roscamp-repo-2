"""Yellow marker detection constrained to the upper portion of a tire mask."""
from __future__ import annotations

import cv2
import numpy as np

try:
    from tire_vision.wear_measurement import MeasurementError
except ImportError:  # Direct execution support for tire_marker_test.py.
    from wear_measurement import MeasurementError


def build_marker_search_mask(tire_mask, center, radius):
    mask = np.asarray(tire_mask)
    if mask.ndim != 2 or radius <= 0:
        raise MeasurementError("invalid tire geometry for marker search")
    height, width = mask.shape
    cx, cy = center
    x1 = max(0, int(cx - 0.8 * radius))
    x2 = min(width, int(np.ceil(cx + 0.8 * radius)))
    y1 = max(0, int(cy - radius))
    y2 = min(height, int(np.ceil(cy - 0.1 * radius)))
    search = np.zeros_like(mask, dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        search[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(search, (mask > 0).astype(np.uint8) * 255)


def detect_yellow_marker(
    frame,
    tire_mask,
    center,
    radius,
    yellow_lower=(15, 80, 80),
    yellow_upper=(40, 255, 255),
    min_marker_area=5,
    max_marker_area=500,
):
    if frame is None or np.asarray(frame).ndim != 3:
        raise MeasurementError("invalid frame for marker detection")
    search = build_marker_search_mask(tire_mask, center, radius)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.asarray(yellow_lower, dtype=np.uint8),
        np.asarray(yellow_upper, dtype=np.uint8),
    )
    yellow = cv2.bitwise_and(yellow, search)
    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if min_marker_area <= area <= max_marker_area and moments["m00"] > 0:
            marker_center = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
            if marker_center[1] < center[1]:
                choices.append((area, contour, marker_center))
    if not choices:
        raise MeasurementError("yellow marker not found")
    area, contour, marker_center = max(choices, key=lambda item: item[0])
    return {
        "contour": contour,
        "center": marker_center,
        "area": area,
        "binary_mask": yellow,
        "search_mask": search,
    }

