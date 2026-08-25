"""Reusable tire segmentation geometry, radial-gap, and ratio stabilization."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MeasurementError(RuntimeError):
    """An expected per-frame measurement failure."""


class TireNotFoundError(MeasurementError):
    """No tire instance was present in the current frame."""


class MarkerNotFoundError(MeasurementError):
    """The tire was found but its yellow marker was absent in this frame."""


class ModelInferenceError(MeasurementError):
    """The model could not produce a usable result for the current frame."""


class GeometryRecheckError(MeasurementError):
    """The tire geometry is unsuitable for a reliable ratio decision."""


class MarkerMissTracker:
    """Hold a stable result briefly, then degrade to RECHECK and ERROR."""

    def __init__(self, hold_frames=3, recheck_frames=5):
        self.hold_frames = int(hold_frames)
        self.recheck_frames = int(recheck_frames)
        if self.hold_frames < 0 or self.recheck_frames < self.hold_frames:
            raise ValueError("marker miss thresholds must satisfy 0 <= hold <= recheck")
        self.reset()

    def reset(self):
        self.miss_count = 0
        self.last_stable_result = None

    def marker_found(self, result):
        self.miss_count = 0
        if result in {"GOOD", "BAD"}:
            self.last_stable_result = result
        return result

    def marker_missing(self):
        self.miss_count += 1
        if self.miss_count <= self.hold_frames and self.last_stable_result:
            return self.last_stable_result
        if self.miss_count <= self.recheck_frames:
            return "RECHECK"
        return "ERROR"


def _scalar(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def extract_tire_detection(result, frame_shape, conf_threshold):
    """Select the best tire instance and return its original-size binary mask."""
    if result.boxes is None or len(result.boxes) == 0:
        raise TireNotFoundError("tire not detected")
    choices = []
    for index, box in enumerate(result.boxes):
        class_id = int(_scalar(box.cls))
        names = result.names
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        confidence = _scalar(box.conf)
        if name == "tire" and confidence >= conf_threshold:
            choices.append((confidence, index, box))
    if not choices:
        raise TireNotFoundError("tire not detected")
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


def estimate_tire_geometry_from_bbox(
    tire_mask,
    min_component_area=100.0,
    min_radius=5.0,
    min_aspect_ratio=0.70,
    max_aspect_ratio=1.30,
):
    """Estimate right-camera geometry from the largest mask component bbox."""
    mask = np.asarray(tire_mask)
    if mask.ndim != 2 or mask.size == 0:
        raise MeasurementError("invalid tire mask")
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise MeasurementError("tire component not found")

    component_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[component_label, cv2.CC_STAT_AREA])
    if area < min_component_area:
        raise MeasurementError(f"tire component is too small ({area:.1f} px^2)")

    component = (labels == component_label).astype(np.uint8) * 255
    x, y, width, height = cv2.boundingRect(component)
    if width <= 0 or height <= 0:
        raise MeasurementError("invalid tire bounding rectangle")
    aspect_ratio = float(width) / float(height)
    if not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
        raise GeometryRecheckError(
            f"right tire bbox aspect ratio out of range: {aspect_ratio:.3f}"
        )

    center = (float(x + width / 2.0), float(y + height / 2.0))
    radius = float(width + height) / 4.0
    if not np.isfinite([*center, radius]).all() or radius < min_radius:
        raise MeasurementError(f"tire radius is too small ({radius:.1f} px)")

    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    return {
        "center": center,
        "radius": radius,
        "contour": contour,
        "area": area,
        "mask": component,
        "bbox": (int(x), int(y), int(x + width), int(y + height)),
        "width": int(width),
        "height": int(height),
        "aspect_ratio": aspect_ratio,
        "method": "bounding_rect",
    }


def estimate_tire_geometry_for_side(tire_mask, side):
    """Keep LEFT unchanged; use bbox geometry only for RIGHT."""
    if str(side).lower().strip() == "right":
        return estimate_tire_geometry_from_bbox(tire_mask)
    return estimate_tire_geometry(tire_mask)


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


def decision_state_args_from_config(config, side=None):
    """Build one camera's state arguments; camera keys take precedence."""
    args = {
        "window_size": int(config.get("ratio_window_size", 15)),
        "min_valid_frames": int(config.get("ratio_min_valid_frames", 10)),
        "bad_max_ratio": float(config.get("bad_max_ratio", 0.28)),
        "good_min_ratio": float(config.get("good_min_ratio", 0.32)),
        "bad_max_gap_px": config.get("bad_max_gap_px"),
        "good_min_gap_px": config.get("good_min_gap_px"),
        "max_ratio_std": float(config.get("max_ratio_std", 0.02)),
        "no_tire_reset_frames": int(config.get("no_tire_reset_frames", 3)),
        "marker_hold_frames": int(config.get("marker_hold_frames", 3)),
        "marker_recheck_frames": int(config.get("marker_recheck_frames", 5)),
    }
    thresholds = config.get("decision_thresholds", {})
    if isinstance(thresholds, dict) and side in {"left", "right"}:
        side_values = thresholds.get(side, {})
        if isinstance(side_values, dict):
            for key in ("bad_max_ratio", "good_min_ratio"):
                if key in side_values:
                    args[key] = float(side_values[key])

    camera_values = config.get(f"{side}_camera", {}) if side else {}
    if isinstance(camera_values, dict):
        integer_keys = {
            "window_size",
            "min_valid_frames",
            "no_tire_reset_frames",
            "marker_hold_frames",
            "marker_recheck_frames",
        }
        float_keys = {
            "bad_max_ratio",
            "good_min_ratio",
            "bad_max_gap_px",
            "good_min_gap_px",
            "max_ratio_std",
        }
        for key in integer_keys:
            if key in camera_values:
                args[key] = int(camera_values[key])
        for key in float_keys:
            if key in camera_values:
                args[key] = float(camera_values[key])

    camera = side or "default"
    if args["window_size"] <= 0 or args["min_valid_frames"] <= 0:
        raise ValueError(f"{camera} window/min_valid_frames must be positive")
    if args["min_valid_frames"] > args["window_size"]:
        raise ValueError(f"{camera} min_valid_frames cannot exceed window_size")
    if args["bad_max_ratio"] >= args["good_min_ratio"]:
        raise ValueError(
            f"{camera} bad_max_ratio must be less than good_min_ratio"
        )
    if args["max_ratio_std"] < 0:
        raise ValueError(f"{camera} max_ratio_std must be non-negative")
    gap_thresholds = (args["bad_max_gap_px"], args["good_min_gap_px"])
    if (gap_thresholds[0] is None) != (gap_thresholds[1] is None):
        raise ValueError(f"{camera} gap thresholds must be configured together")
    if (
        gap_thresholds[0] is not None
        and gap_thresholds[0] >= gap_thresholds[1]
    ):
        raise ValueError(
            f"{camera} bad_max_gap_px must be less than good_min_gap_px"
        )

    if (
        args["marker_hold_frames"] < 0
        or args["marker_recheck_frames"] < args["marker_hold_frames"]
    ):
        raise ValueError(f"{camera} marker miss thresholds are invalid")
    return args


class RatioStabilizer:
    """Keep a bounded history of valid normalized gap measurements."""

    def __init__(self, window_size=15, min_valid_frames=10,
                 bad_max_ratio=0.28, good_min_ratio=0.32,
                 max_ratio_std=0.02):
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



class TireDecisionState:
    """Shared per-camera tire decision session for server and debug UI."""

    STATES = {"NO_TIRE", "COLLECTING", "GOOD", "BAD", "RECHECK", "ERROR"}

    def __init__(
        self,
        window_size=15,
        min_valid_frames=10,
        bad_max_ratio=0.28,
        good_min_ratio=0.32,
        bad_max_gap_px=None,
        good_min_gap_px=None,
        max_ratio_std=0.02,
        no_tire_reset_frames=3,
        marker_hold_frames=3,
        marker_recheck_frames=5,
    ):
        self.no_tire_reset_frames = max(1, int(no_tire_reset_frames))
        self.ratio_stabilizer = RatioStabilizer(
            window_size=window_size,
            min_valid_frames=min_valid_frames,
            bad_max_ratio=bad_max_ratio,
            good_min_ratio=good_min_ratio,
            max_ratio_std=max_ratio_std,
        )
        self.bad_max_gap_px = (
            None if bad_max_gap_px is None else float(bad_max_gap_px)
        )
        self.good_min_gap_px = (
            None if good_min_gap_px is None else float(good_min_gap_px)
        )
        if (self.bad_max_gap_px is None) != (self.good_min_gap_px is None):
            raise ValueError("gap thresholds must be configured together")
        if (
            self.bad_max_gap_px is not None
            and self.bad_max_gap_px >= self.good_min_gap_px
        ):
            raise ValueError("bad_max_gap_px must be less than good_min_gap_px")
        self.gap_history = deque(maxlen=self.ratio_stabilizer.window_size)

        self.marker_tracker = MarkerMissTracker(
            hold_frames=marker_hold_frames,
            recheck_frames=marker_recheck_frames,
        )
        self.session_id = 0
        self.reset_session()

    @property
    def ratio_history(self):
        return self.ratio_stabilizer.history

    @property
    def marker_miss_count(self):
        return self.marker_tracker.miss_count

    @property
    def last_stable_result(self):
        return self.marker_tracker.last_stable_result

    @property
    def stable_result(self):
        """Compatibility name used by the overlay and service diagnostics."""
        return self.last_stable_result

    def _clear_decision_buffers(self):
        self.ratio_stabilizer.reset()
        self.gap_history.clear()
        self.marker_tracker.reset()
        self.current_gap = None
        self.current_radius = None
        self.current_ratio = None

    def reset_session(self):
        """Clear every value that must not cross a detection request."""
        self.session_id += 1
        self._clear_decision_buffers()
        self.no_tire_count = 0
        self.retry_count = 0
        self.current_tire = "UNKNOWN"
        self.current_marker = "UNKNOWN"
        self.current_result = "NO_TIRE"
        self.detail = "detection session reset"
        self.error_code = None
        self._no_tire_latched = False
        return self.snapshot()

    def reset_for_no_tire(self, detail="tire not detected"):
        """End the old tire session and discard all measurements."""
        self._clear_decision_buffers()
        self.no_tire_count = self.no_tire_reset_frames
        self.current_tire = "LOST"
        self.current_marker = "UNKNOWN"
        self.current_result = "NO_TIRE"
        self.detail = str(detail)
        self.error_code = "NO_TIRE"
        self._no_tire_latched = True
        return self.snapshot()

    def _start_tire_session(self):
        # A request/manual reset already allocated a fresh session. A tire
        # appearing after a latched NO_TIRE state starts the next one here.
        if self._no_tire_latched:
            self.session_id += 1
        self._clear_decision_buffers()
        self.no_tire_count = 0
        self.retry_count = 0
        self.current_tire = "FOUND"
        self.current_marker = "UNKNOWN"
        self.current_result = "COLLECTING"
        self.detail = "collecting tire measurements"
        self.error_code = None
        self._no_tire_latched = False

    def update_tire_detection(self, tire_found, detail=""):
        if tire_found:
            if self._no_tire_latched or self.current_result == "NO_TIRE":
                self._start_tire_session()
            self.no_tire_count = 0
            self.current_tire = "FOUND"
            self.error_code = None
            return self.snapshot()

        self.current_tire = "LOST"
        self.current_marker = "UNKNOWN"
        self.current_gap = None
        self.current_radius = None
        self.current_ratio = None
        self.detail = str(detail or "tire not detected")
        if self._no_tire_latched:
            self.no_tire_count = self.no_tire_reset_frames
            self.current_result = "NO_TIRE"
            self.error_code = "NO_TIRE"
            return self.snapshot()

        self.no_tire_count += 1
        if self.no_tire_count >= self.no_tire_reset_frames:
            return self.reset_for_no_tire(self.detail)
        self.current_result = self.last_stable_result or "RECHECK"
        self.error_code = None
        return self.snapshot()

    @property
    def uses_gap_thresholds(self):
        return self.bad_max_gap_px is not None

    def _classify_gap_ratio(self, summary):
        median_gap = float(np.median(self.gap_history))
        median_ratio = summary["median_ratio"]
        if summary["std_ratio"] > self.ratio_stabilizer.max_ratio_std:
            return "RECHECK"
        if (
            median_gap <= self.bad_max_gap_px
            or median_ratio <= self.ratio_stabilizer.bad_max_ratio
        ):
            return "BAD"
        if (
            median_gap >= self.good_min_gap_px
            and median_ratio >= self.ratio_stabilizer.good_min_ratio
        ):
            return "GOOD"
        return "RECHECK"

    def update_marker_detection(
        self,
        marker_found,
        ratio=None,
        gap_px=None,
        radius_px=None,
        detail="",
    ):
        self.current_tire = "FOUND"
        self.no_tire_count = 0
        if marker_found:
            self.current_marker = "FOUND"
            self.current_gap = float(gap_px) if gap_px is not None else None
            self.current_radius = float(radius_px) if radius_px is not None else None
            self.current_ratio = float(ratio) if ratio is not None else None
            if self.uses_gap_thresholds and self.current_gap is None:
                self.marker_tracker.marker_found("RECHECK")
                self.current_result = "RECHECK"
                self.detail = "gap measurement unavailable"
                self.error_code = None
                return self.snapshot()

            if (
                self.uses_gap_thresholds
                and (not np.isfinite(self.current_gap) or self.current_gap < 0)
            ):
                self.marker_tracker.marker_found("RECHECK")
                self.current_result = "RECHECK"
                self.detail = "invalid gap measurement"
                self.error_code = None
                return self.snapshot()

            summary = self.ratio_stabilizer.update(self.current_ratio)
            if self.uses_gap_thresholds:
                self.gap_history.append(self.current_gap)
            if len(self.ratio_history) < self.ratio_stabilizer.min_valid_frames:
                result = "COLLECTING"
            elif self.uses_gap_thresholds:
                result = self._classify_gap_ratio(summary)
            else:
                result = summary["result"]
            self.marker_tracker.marker_found(result)
            self.current_result = result
            self.detail = str(detail or (
                "collecting tire measurements" if result == "COLLECTING" else ""
            ))
            self.error_code = None
            return self.snapshot()

        self.current_marker = "LOST"
        self.current_gap = None
        self.current_radius = None
        self.current_ratio = None
        self.current_result = self.marker_tracker.marker_missing()
        self.detail = str(detail or "yellow marker not found")
        self.error_code = (
            "MARKER_LOST" if self.current_result == "ERROR" else None
        )
        return self.snapshot()

    def update_recheck(self, detail, error_code="GEOMETRY_RECHECK"):
        """Reject the current frame without mutating valid ratio history."""
        self.current_tire = "FOUND"
        self.current_marker = "UNKNOWN"
        self.no_tire_count = 0
        self.current_gap = None
        self.current_radius = None
        self.current_ratio = None
        self.current_result = "RECHECK"
        self.detail = str(detail)
        self.error_code = str(error_code)
        return self.snapshot()

    def update_error(self, detail, error_code="DETECTION_ERROR"):
        self.current_gap = None
        self.current_radius = None
        self.current_ratio = None
        self.current_result = "ERROR"
        self.retry_count += 1
        self.detail = str(detail)
        self.error_code = str(error_code)
        return self.snapshot()

    def snapshot(self):
        summary = self.ratio_stabilizer.summary()
        median_gap = (
            float(np.median(self.gap_history)) if self.gap_history else None
        )
        return {
            "session_id": self.session_id,
            "result": self.current_result,
            "current_tire": self.current_tire,
            "current_marker": self.current_marker,
            "stable_result": self.last_stable_result,
            "no_tire_count": self.no_tire_count,
            "no_tire_reset_frames": self.no_tire_reset_frames,
            "marker_miss_count": self.marker_miss_count,
            # Kept for callers that used the old diagnostics spelling.
            "miss_count": self.marker_miss_count,
            "marker_hold_frames": self.marker_tracker.hold_frames,
            "marker_recheck_frames": self.marker_tracker.recheck_frames,
            "valid_count": len(self.ratio_history),
            "window_size": self.ratio_stabilizer.window_size,
            "min_valid_frames": self.ratio_stabilizer.min_valid_frames,
            "bad_max_ratio": self.ratio_stabilizer.bad_max_ratio,
            "good_min_ratio": self.ratio_stabilizer.good_min_ratio,
            "bad_max_gap_px": self.bad_max_gap_px,
            "good_min_gap_px": self.good_min_gap_px,
            "median_gap_px": median_gap,
            "max_ratio_std": self.ratio_stabilizer.max_ratio_std,
            "gap_px": self.current_gap,
            "radius_px": self.current_radius,
            "gap_ratio": self.current_ratio,
            "median_ratio": summary["median_ratio"],
            "std_ratio": summary["std_ratio"],
            "detail": self.detail,
            "error_code": self.error_code,
            "retry_count": self.retry_count,
        }
