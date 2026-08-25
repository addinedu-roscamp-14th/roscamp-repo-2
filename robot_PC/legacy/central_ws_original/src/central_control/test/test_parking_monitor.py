import pytest

from central_control.parking_monitor import (
    ParkingMonitor,
    ParkingState,
    vehicle_entry_points,
)
from central_control.manual_central_node import CentralParkingStateMachine
from central_control.work_state_machine import WorkStateMachine, WorkState


PARKING_ROI = [[0, 0], [100, 0], [100, 100], [0, 100]]
ENTRY_POINTS = ((50, 20), (50, 80))


def build_monitor(**overrides):
    options = {
        "parking_roi": PARKING_ROI,
        "stop_hold_time": 1.0,
        "median_samples": 1,
        "motion_window_sec": 0.5,
        "stop_jitter_threshold": 2.0,
        "move_jitter_threshold": 6.0,
        "stop_speed_threshold": 2.0,
        "move_speed_threshold": 5.0,
        "motion_grace_sec": 0.3,
        "detection_lost_grace_sec": 0.3,
        "roi_exit_grace_frames": 2,
    }
    options.update(overrides)
    return ParkingMonitor(**options)


def feed(monitor, centers, start=0.0, step=0.1):
    state = None
    for index, center in enumerate(centers):
        state = monitor.update(
            center,
            entry_points=ENTRY_POINTS,
            now=start + index * step,
        )
    return state


def test_entry_points_use_vertical_bbox_axis():
    assert vehicle_entry_points([20, 10, 60, 90], 0.1) == ((40, 18), (40, 82))


def test_roi_outside_does_not_park():
    monitor = build_monitor()
    state = monitor.update((50, 50), entry_points=((50, -1), (50, 80)), now=0.0)
    assert state == ParkingState.ENTERING
    assert monitor.fully_inside is False
    assert monitor.stop_start_time is None


def test_points_must_be_inside_the_same_roi():
    monitor = build_monitor(
        parking_roi=[
            [[0, 0], [40, 0], [40, 100], [0, 100]],
            [[60, 0], [100, 0], [100, 100], [60, 100]],
        ]
    )
    state = monitor.update((50, 50), entry_points=((20, 50), (80, 50)), now=0.0)
    assert state == ParkingState.ENTERING
    assert monitor.fully_inside is False


def test_roi_inside_but_moving_does_not_park():
    monitor = build_monitor(stop_speed_threshold=2.0, move_speed_threshold=5.0)
    centers = [(50 + int(index * 0.3), 50) for index in range(20)]
    state = feed(monitor, centers)
    assert state == ParkingState.STOP_CHECK
    assert monitor.speed > monitor.stop_speed_threshold
    assert monitor.stop_start_time is None


def test_roi_inside_stationary_parks_once_with_latch():
    parking_config = {
        "parking_detection_roi": PARKING_ROI,
        "parking_roi": PARKING_ROI,
        "parking_rois": [PARKING_ROI],
        "parking_entry_point_inset_ratio": 0.1,
        "parking_median_samples": 1,
        "parking_motion_window_sec": 0.5,
        "parking_stop_jitter_px": 2.0,
        "parking_move_jitter_px": 6.0,
        "parking_stop_speed_px_sec": 2.0,
        "parking_move_speed_px_sec": 5.0,
        "parking_motion_grace_sec": 0.3,
        "parking_detection_lost_grace_sec": 0.3,
        "parking_roi_exit_grace_frames": 2,
        "parking_stop_seconds": 0.8,
    }
    parking = CentralParkingStateMachine(parking_config)
    work = WorkStateMachine()
    detection = {"class_name": "Car_B", "confidence": 0.9, "bbox": [20, 10, 80, 90]}
    events = 0
    for index in range(16):
        result = parking.process_detections([detection], now=index * 0.1)
        if result.parked_event:
            work.parking_completed()
            events += 1
    assert events == 1
    assert work.state == WorkState.WAIT_SET_CAMERA_CONFIRM
    assert parking.parked_latched is True


def test_detection_loss_returns_waiting_after_grace():
    monitor = build_monitor(motion_window_sec=0.3, stop_hold_time=2.0, detection_lost_grace_sec=0.1)
    feed(monitor, [(50, 50)] * 9)
    monitor.update(None, now=0.9)
    state = monitor.update(None, now=1.2)
    assert state == ParkingState.WAITING
    assert monitor.entry_confirmed is False


def test_reset_allows_reparking():
    parking_config = {
        "parking_detection_roi": PARKING_ROI,
        "parking_roi": PARKING_ROI,
        "parking_rois": [PARKING_ROI],
        "parking_entry_point_inset_ratio": 0.1,
        "parking_median_samples": 1,
        "parking_motion_window_sec": 0.5,
        "parking_stop_jitter_px": 2.0,
        "parking_move_jitter_px": 6.0,
        "parking_stop_speed_px_sec": 2.0,
        "parking_move_speed_px_sec": 5.0,
        "parking_motion_grace_sec": 0.3,
        "parking_detection_lost_grace_sec": 0.3,
        "parking_roi_exit_grace_frames": 2,
        "parking_stop_seconds": 0.8,
    }
    parking = CentralParkingStateMachine(parking_config)
    detection = {"class_name": "Car_B", "confidence": 0.9, "bbox": [20, 10, 80, 90]}
    for index in range(16):
        parking.process_detections([detection], now=index * 0.1)
    assert parking.parked_latched
    parking.reset()
    assert not parking.parked_latched
    events = 0
    for index in range(16):
        result = parking.process_detections([detection], now=2.0 + index * 0.1)
        events += int(result.parked_event)
    assert events == 1
