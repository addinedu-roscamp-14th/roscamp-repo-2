import time

from central_control.manual_central_node import ManualCentralRuntime
from central_control.work_state_machine import WorkState


PARKING_CONFIG = {
    "mock_camera": True,
    "top_camera_process_width": 600,
    "top_camera_process_height": 400,
    "parking_detection_roi": [[255, 274], [354, 277], [349, 355], [261, 352]],
    "parking_roi": [[280, 287], [330, 288], [330, 358], [282, 358]],
    "parking_rois": [[[280, 287], [330, 288], [330, 358], [282, 358]]],
    "parking_entry_point_inset_ratio": 0.10,
    "parking_median_samples": 1,
    "parking_motion_window_sec": 0.1,
    "parking_stop_jitter_px": 6.0,
    "parking_move_jitter_px": 10.0,
    "parking_stop_speed_px_sec": 2.0,
    "parking_move_speed_px_sec": 5.0,
    "parking_motion_grace_sec": 0.1,
    "parking_detection_lost_grace_sec": 0.1,
    "parking_roi_exit_grace_frames": 1,
    "parking_stop_seconds": 0.1,
}
YOLO_CONFIG = {
    "mock_vehicle_detector": True,
    "show_window": False,
    "mock_detections": [
        {"class_id": 0, "class_name": "Car_B", "confidence": 0.95, "bbox": [285.0, 295.0, 325.0, 350.0]}
    ],
}
NETWORK_CONFIG = {
    "left_arm": {"host": "127.0.0.1", "port": 1, "connect_timeout_sec": 0.1, "command_timeout_sec": 0.1},
    "right_arm": {"host": "127.0.0.1", "port": 2, "connect_timeout_sec": 0.1, "command_timeout_sec": 0.1},
    "vision_pc": {"host": "127.0.0.1", "port": 3, "connect_timeout_sec": 0.1, "request_timeout_sec": 0.1},
}
OPERATION_CONFIG = {
    "operation": {"manual_step": True, "mock_arms": True, "mock_vision": True},
    "commands": {
        "set_camera": "SET_CAMERA",
        "remove_bad_tire": "REMOVE_BAD_TIRE",
        "set_aruco": "SET_ARUCO",
        "home": "HOME",
        "stop": "STOP",
    },
    "timeouts": {"worker_shutdown_sec": 0.1},
}


def wait_worker(runtime):
    for _ in range(20):
        runtime._apply_worker_result_if_ready()
        if runtime.worker_future is None:
            return
        time.sleep(0.02)
    raise AssertionError("worker did not finish")


def test_mock_full_flow_from_bypass_to_completed():
    runtime = ManualCentralRuntime(PARKING_CONFIG, YOLO_CONFIG, NETWORK_CONFIG, OPERATION_CONFIG)
    try:
        runtime.handle_key(ord("n"))
        assert runtime.work_state_machine.state == WorkState.WAIT_DETECTION_CONFIRM
        runtime.handle_key(32)
        wait_worker(runtime)
        assert runtime.work_state_machine.state == WorkState.WAIT_ACTION_CONFIRM
        assert runtime.work_state_machine.target_arms == {"left"}
        runtime.handle_key(32)
        wait_worker(runtime)
        assert runtime.work_state_machine.state == WorkState.WAIT_ARUCO_CONFIRM
        runtime.handle_key(32)
        wait_worker(runtime)
        assert runtime.work_state_machine.state == WorkState.WAIT_HOME_CONFIRM
        runtime.handle_key(32)
        wait_worker(runtime)
        assert runtime.work_state_machine.state == WorkState.COMPLETED
    finally:
        runtime.close()
