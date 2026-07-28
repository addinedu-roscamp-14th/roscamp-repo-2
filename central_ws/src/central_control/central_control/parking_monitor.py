from __future__ import annotations

import time
from enum import Enum

import cv2
import numpy as np

from central_control.stop_detector import StopDetector


class ParkingState(str, Enum):
    WAITING = "WAITING"
    ENTERING = "ENTERING"
    STOP_CHECK = "STOP_CHECK"
    PARKED = "PARKED"
    TRIGGER = "TRIGGER"


class ParkingMonitor:
    """Track robust vehicle motion until it remains stopped inside the ROI."""

    def __init__(
        self,
        parking_roi,
        stop_threshold=4.0,
        stop_hold_time=3.0,
        center_history=10,
        clock=time.monotonic,
        median_samples=5,
        motion_window_sec=1.0,
        stop_jitter_threshold=6.0,
        move_jitter_threshold=10.0,
        stop_speed_threshold=2.0,
        move_speed_threshold=5.0,
        motion_grace_sec=0.4,
        detection_lost_grace_sec=0.3,
        roi_exit_grace_frames=3,
    ):
        parking_rois = np.asarray(parking_roi, dtype=np.int32)
        if parking_rois.ndim == 2:
            self.parking_rois = [parking_rois]
        elif parking_rois.ndim == 3:
            self.parking_rois = [roi for roi in parking_rois]
        else:
            raise ValueError(
                "parking_roi must be a polygon or a list of polygons"
            )
        self.parking_roi = self.parking_rois[0]
        self.stop_hold_time = float(stop_hold_time)
        self.stop_detector = StopDetector(
            stop_hold_time=stop_hold_time,
            median_samples=median_samples,
            motion_window_sec=motion_window_sec,
            stop_jitter_threshold=stop_jitter_threshold,
            move_jitter_threshold=move_jitter_threshold,
            stop_speed_threshold=stop_speed_threshold,
            move_speed_threshold=move_speed_threshold,
            motion_grace_sec=motion_grace_sec,
        )
        self.median_samples = self.stop_detector.median_samples
        self.motion_window_sec = self.stop_detector.motion_window_sec
        self.stop_jitter_threshold = self.stop_detector.stop_jitter_threshold
        self.move_jitter_threshold = self.stop_detector.move_jitter_threshold
        self.stop_speed_threshold = self.stop_detector.stop_speed_threshold
        self.move_speed_threshold = self.stop_detector.move_speed_threshold
        self.motion_grace_sec = self.stop_detector.motion_grace_sec
        self.detection_lost_grace_sec = max(
            0.0, float(detection_lost_grace_sec)
        )
        self.roi_exit_grace_frames = max(0, int(roi_exit_grace_frames))

        # Compatibility for existing callers that inspect legacy attributes.
        self.stop_threshold = self.move_jitter_threshold
        self.center_history = max(2, int(center_history))
        self.clock = clock
        self.state = ParkingState.WAITING
        self.lost_start_time = None
        self.roi_exit_frames = 0
        self.entry_confirmed = False
        self.center = None
        self.entry_points = None
        self.inside = False
        self.fully_inside = False
        self.triggered = False
        self._sync_stop_attrs()

    def _sync_stop_attrs(self):
        detector = self.stop_detector
        self.raw_centers = detector.raw_centers
        self.motion_samples = detector.motion_samples
        self.stop_start_time = detector.stop_start_time
        self.stop_elapsed = detector.stop_elapsed
        self.last_stationary_time = detector.last_stationary_time
        self.motion_violation_start_time = detector.motion_violation_start_time
        self.is_stationary = detector.is_stationary
        self.is_moving = detector.is_moving
        self.movement = detector.movement
        self.jitter_radius = detector.jitter_radius
        self.speed = detector.speed
        self.filtered_center = detector.filtered_center
        self.stationary_reference = detector.stationary_reference

    def contains(self, point):
        return any(
            cv2.pointPolygonTest(roi, point, False) >= 0
            for roi in self.parking_rois
        )

    def contains_all(self, points):
        """Return true when every point is inside the same parking ROI."""
        return any(
            all(
                cv2.pointPolygonTest(roi, point, False) >= 0
                for point in points
            )
            for roi in self.parking_rois
        )

    def reset_motion(self):
        self.stop_detector.reset()
        self.lost_start_time = None
        self.roi_exit_frames = 0
        self.entry_confirmed = False
        self._sync_stop_attrs()

    def update(self, center, now=None, entry_points=None):
        now = self.clock() if now is None else float(now)
        if self.triggered:
            self.state = ParkingState.TRIGGER
            return self.state
        if center is None:
            self.center = None
            self.entry_points = None
            self.inside = False
            self.fully_inside = False
            self.stop_detector.freeze()
            if not self.entry_confirmed:
                self._sync_stop_attrs()
                self.state = ParkingState.WAITING
                return self.state
            if self.lost_start_time is None:
                self.lost_start_time = now
            if now - self.lost_start_time <= self.detection_lost_grace_sec:
                self._sync_stop_attrs()
                self.state = ParkingState.STOP_CHECK
                return self.state
            self.reset_motion()
            self.stop_detector.is_moving = True
            self._sync_stop_attrs()
            self.state = ParkingState.WAITING
            return self.state

        self.lost_start_time = None
        self.center = (int(center[0]), int(center[1]))
        self.inside = self.contains(self.center)
        if entry_points is None:
            self.entry_points = (self.center,)
        else:
            self.entry_points = tuple(
                (int(point[0]), int(point[1])) for point in entry_points
            )
        self.fully_inside = self.contains_all(self.entry_points)
        if not self.fully_inside:
            self.stop_detector.freeze()
            if not self.entry_confirmed:
                self.reset_motion()
                self.state = ParkingState.ENTERING
                return self.state
            self.roi_exit_frames += 1
            if self.roi_exit_frames <= self.roi_exit_grace_frames:
                self._sync_stop_attrs()
                self.state = ParkingState.STOP_CHECK
                return self.state
            self.reset_motion()
            self.stop_detector.is_moving = True
            self._sync_stop_attrs()
            self.state = ParkingState.ENTERING
            return self.state

        self.entry_confirmed = True
        self.roi_exit_frames = 0
        self.stop_detector.update(self.center, now)
        self._sync_stop_attrs()
        self.state = ParkingState.STOP_CHECK

        if self.is_stationary and self.stop_elapsed >= self.stop_hold_time:
            self.state = ParkingState.PARKED
        return self.state

    def consume_parked(self):
        if self.state != ParkingState.PARKED:
            return False
        self.triggered = True
        self.state = ParkingState.TRIGGER
        return True

    def stopped_seconds(self, now=None):
        del now
        return self.stop_elapsed


def vehicle_entry_points(bbox, inset_ratio=0.1):
    """Return front/rear points on the vertical vehicle axis in BEV pixels."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    ratio = min(0.49, max(0.0, float(inset_ratio)))
    center_x = (x1 + x2) / 2.0
    inset = (y2 - y1) * ratio
    front = (int(round(center_x)), int(round(y1 + inset)))
    rear = (int(round(center_x)), int(round(y2 - inset)))
    return front, rear
