"""Yellow marker detection in an expanded upper tire region."""
from __future__ import annotations

import cv2
import numpy as np

try:
    from tire_vision.wear_measurement import MeasurementError, MarkerNotFoundError
except ImportError:  # Direct execution support for tire_marker_test.py.
    from wear_measurement import MeasurementError, MarkerNotFoundError


def build_marker_search_mask(tire_mask, center, radius, marker_roi_scale=1.15):
    """Expand the tire mask around its edge, then keep its upper marker region."""
    mask = np.asarray(tire_mask)
    scale = float(marker_roi_scale)
    if mask.ndim != 2 or radius <= 0 or scale < 1.0:
        raise MeasurementError("invalid tire geometry for marker search")

    height, width = mask.shape
    cx, cy = center
    expanded_radius = scale * float(radius)
    x1 = max(0, int(cx - 0.8 * expanded_radius))
    x2 = min(width, int(np.ceil(cx + 0.8 * expanded_radius)))
    y1 = max(0, int(cy - expanded_radius))
    y2 = min(height, int(np.ceil(cy - 0.1 * radius)))

    upper_roi = np.zeros_like(mask, dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        upper_roi[y1:y2, x1:x2] = 255

    tire_binary = (mask > 0).astype(np.uint8) * 255
    margin_px = max(0, int(np.ceil((scale - 1.0) * float(radius))))
    if margin_px:
        kernel_size = 2 * margin_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        tire_binary = cv2.dilate(tire_binary, kernel)

    return cv2.bitwise_and(upper_roi, tire_binary)


def detect_yellow_marker(
    frame,
    tire_mask,
    center,
    radius,
    yellow_lower=(15, 40, 60),
    yellow_upper=(42, 255, 255),
    min_marker_area=5,
    max_marker_area=500,
    marker_roi_scale=1.15,
):
    if frame is None or np.asarray(frame).ndim != 3:
        raise MeasurementError("invalid frame for marker detection")
    search = build_marker_search_mask(
        tire_mask, center, radius, marker_roi_scale=marker_roi_scale
    )
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
    contours, _ = cv2.findContours(
        yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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
        raise MarkerNotFoundError("yellow marker not found")
    area, contour, marker_center = max(choices, key=lambda item: item[0])
    return {
        "contour": contour,
        "center": marker_center,
        "area": area,
        "binary_mask": yellow,
        "search_mask": search,
    }
