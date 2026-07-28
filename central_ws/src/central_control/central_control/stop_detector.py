from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StopDetectorResult:
    window_ready: bool
    is_stationary: bool
    is_moving: bool
    stop_started: bool
    stop_elapsed: float
    stop_start_time: float | None
    jitter_radius: float | None
    speed: float | None
    movement: float | None
    filtered_center: tuple[float, float] | None


class StopDetector:
    """Robust stopped-vehicle detector ported from tire_ws ParkingMonitor."""

    def __init__(
        self,
        stop_hold_time=3.0,
        median_samples=5,
        motion_window_sec=1.0,
        stop_jitter_threshold=6.0,
        move_jitter_threshold=10.0,
        stop_speed_threshold=2.0,
        move_speed_threshold=5.0,
        motion_grace_sec=0.4,
    ):
        self.stop_hold_time = float(stop_hold_time)
        self.median_samples = max(1, int(median_samples))
        self.motion_window_sec = max(0.1, float(motion_window_sec))
        self.stop_jitter_threshold = float(stop_jitter_threshold)
        self.move_jitter_threshold = max(
            self.stop_jitter_threshold, float(move_jitter_threshold)
        )
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.move_speed_threshold = max(
            self.stop_speed_threshold, float(move_speed_threshold)
        )
        self.motion_grace_sec = max(0.0, float(motion_grace_sec))
        self.raw_centers = deque(maxlen=self.median_samples)
        self.motion_samples = deque()
        self.stop_start_time = None
        self.stop_elapsed = 0.0
        self.last_stationary_time = None
        self.motion_violation_start_time = None
        self.is_stationary = False
        self.is_moving = False
        self.movement = None
        self.jitter_radius = None
        self.speed = None
        self.filtered_center = None
        self.stationary_reference = None

    def reset(self):
        self.raw_centers.clear()
        self.motion_samples.clear()
        self.stop_start_time = None
        self.stop_elapsed = 0.0
        self.last_stationary_time = None
        self.motion_violation_start_time = None
        self.is_stationary = False
        self.is_moving = False
        self.movement = None
        self.jitter_radius = None
        self.speed = None
        self.filtered_center = None
        self.stationary_reference = None

    def freeze(self):
        self.last_stationary_time = None

    def mark_motion_reset(self):
        self.stop_start_time = None
        self.stop_elapsed = 0.0
        self.last_stationary_time = None
        self.motion_violation_start_time = None
        self.is_stationary = False
        self.is_moving = True
        self.stationary_reference = None

    def update(self, center, now: float) -> StopDetectorResult:
        self._append_motion_sample(center, now)
        window_ready = self._update_motion_metrics(now)
        self._update_stationary_state(now, window_ready)
        return self.result(window_ready)

    def result(self, window_ready=False) -> StopDetectorResult:
        return StopDetectorResult(
            window_ready=window_ready,
            is_stationary=self.is_stationary,
            is_moving=self.is_moving,
            stop_started=self.stop_start_time is not None,
            stop_elapsed=self.stop_elapsed,
            stop_start_time=self.stop_start_time,
            jitter_radius=self.jitter_radius,
            speed=self.speed,
            movement=self.movement,
            filtered_center=self.filtered_center,
        )

    def _append_motion_sample(self, center, now):
        self.raw_centers.append((int(center[0]), int(center[1])))
        raw = np.asarray(self.raw_centers, dtype=np.float64)
        filtered = np.median(raw, axis=0)
        self.filtered_center = (float(filtered[0]), float(filtered[1]))
        self.motion_samples.append(
            (now, self.filtered_center[0], self.filtered_center[1])
        )

        cutoff = now - self.motion_window_sec
        while (
            len(self.motion_samples) > 2
            and self.motion_samples[1][0] <= cutoff
        ):
            self.motion_samples.popleft()

    def _update_motion_metrics(self, now):
        del now
        if len(self.motion_samples) < 2:
            self.jitter_radius = None
            self.speed = None
            self.movement = None
            return False

        samples = np.asarray(self.motion_samples, dtype=np.float64)
        duration = float(samples[-1, 0] - samples[0, 0])
        points = samples[:, 1:3]
        median = np.median(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        self.jitter_radius = float(np.percentile(distances, 90))
        self.movement = self.jitter_radius

        times = samples[:, 0] - np.mean(samples[:, 0])
        denominator = float(np.dot(times, times))
        if denominator <= 1e-9:
            self.speed = None
        else:
            centered_points = points - np.mean(points, axis=0)
            slopes = np.dot(times, centered_points) / denominator
            self.speed = float(np.linalg.norm(slopes))
        return duration >= self.motion_window_sec

    def _accumulate_stop_time(self, now):
        if self.last_stationary_time is not None:
            self.stop_elapsed += max(0.0, now - self.last_stationary_time)
        self.last_stationary_time = now

    def _update_stationary_state(self, now, window_ready):
        can_enter_stop = (
            window_ready
            and self.jitter_radius is not None
            and self.speed is not None
            and self.jitter_radius <= self.stop_jitter_threshold
            and self.speed <= self.stop_speed_threshold
        )
        reference_displacement = 0.0
        if self.stationary_reference is not None:
            reference_displacement = float(
                np.linalg.norm(
                    np.asarray(self.filtered_center)
                    - np.asarray(self.stationary_reference)
                )
            )
        moving_sample = (
            window_ready
            and self.jitter_radius is not None
            and self.speed is not None
            and (
                self.jitter_radius >= self.move_jitter_threshold
                or self.speed >= self.move_speed_threshold
                or reference_displacement >= self.move_jitter_threshold
            )
        )

        if not self.is_stationary:
            self.is_moving = bool(moving_sample)
            if not can_enter_stop:
                return
            self.is_stationary = True
            self.is_moving = False
            self.stop_start_time = now
            self.stop_elapsed = 0.0
            self.last_stationary_time = now
            self.motion_violation_start_time = None
            self.stationary_reference = self.filtered_center
            return

        if moving_sample:
            self.last_stationary_time = None
            if self.motion_violation_start_time is None:
                self.motion_violation_start_time = now
            if now - self.motion_violation_start_time >= self.motion_grace_sec:
                self.mark_motion_reset()
            return

        self.motion_violation_start_time = None
        if not can_enter_stop:
            self.last_stationary_time = None
            return

        self.is_moving = False
        self._accumulate_stop_time(now)
