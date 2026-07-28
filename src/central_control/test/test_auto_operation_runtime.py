from copy import deepcopy
import threading
import time
from types import SimpleNamespace

import pytest

from central_control.arm_client import ArmCommandResult
from central_control.auto_operation_node import AutoOperationRuntime
from central_control.operation_controller import OperationState
from central_control.vision_client import VisionClientResult


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
AUTO_CONFIG = {
    "operation": {"auto_start": False, "mock_arms": True, "mock_vision": True},
    "delays": {"after_parking_sec": 0.0, "after_set_camera_sec": 0.0, "after_detection_sec": 0.0, "after_right_help_sec": 0.0},
    "commands": {"set_camera": "SET_CAMERA", "right_help": "HELP", "left_remove_bad_tire": "REMOVE_BAD_TIRE", "home": "HOME", "stop": "STOP"},
    "timeouts": {"worker_shutdown_sec": 0.1},
    "safety": {
        "vehicle_motion_monitor_enabled": True,
        "emergency_jitter_px": 12.0,
        "emergency_speed_px_sec": 6.0,
        "emergency_reference_displacement_px": 12.0,
        "emergency_hold_sec": 0.35,
        "detection_lost_emergency_sec": 1.0,
        "send_stop_before_home": True,
        "stop_to_home_delay_sec": 0.0,
        "home_on_vehicle_motion": True,
        "require_manual_reset": True,
    },
}
DETECTION = {
    "left": {"class_name": "bad_tire", "confidence": 0.91, "stable": True, "votes": 3, "frames": 3, "detections": []},
    "right": {"class_name": "good_tire", "confidence": 0.88, "stable": True, "votes": 3, "frames": 3, "detections": []},
}


class FakeArmManager:
    def __init__(self, fail=None):
        self.fail = set(fail or [])
        self.commands = []

    def _result(self, side, command):
        self.commands.append((side, command))
        success = (side, command) not in self.fail and command not in self.fail
        return ArmCommandResult(side, command, success, response=("OK" if success else "ERR"))

    def send_both(self, command):
        return {"left": self._result("left", command), "right": self._result("right", command)}

    def send_right(self, command):
        return {"right": self._result("right", command)}

    def send_left(self, command):
        return {"left": self._result("left", command)}

    @staticmethod
    def all_success(results):
        return bool(results) and all(r.success for r in results.values())


class FakeVisionClient:
    def __init__(self, response=None, error=None):
        self.response = response or DETECTION
        self.error = error
        self.calls = 0

    def request_detection(self):
        self.calls += 1
        if self.error:
            return VisionClientResult(False, error=self.error, request_id="detect-test")
        return VisionClientResult(True, response=self.response, request_id="detect-test")


def make_runtime(arm=None, vision=None, auto_config=None):
    return AutoOperationRuntime(
        deepcopy(PARKING_CONFIG),
        deepcopy(YOLO_CONFIG),
        deepcopy(NETWORK_CONFIG),
        deepcopy(auto_config or AUTO_CONFIG),
        arm_manager=arm or FakeArmManager(),
        vision_client=vision or FakeVisionClient(),
    )


def pump(runtime, limit=100):
    for _ in range(limit):
        runtime.step()
        if runtime.worker_future is None and not runtime.controller.busy:
            if runtime.controller.state in (OperationState.COMPLETED, OperationState.ERROR, OperationState.STOPPED, OperationState.PARKING_STABLE, OperationState.IDLE, OperationState.ARMED, OperationState.WAITING_FOR_VEHICLE):
                return
        time.sleep(0.01)


def test_no_set_camera_before_arm_even_if_parking_complete():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("n"))
        pump(runtime)
        assert arm.commands == []
        assert runtime.controller.state == OperationState.IDLE
    finally:
        runtime.close()


def test_a_input_arms_system():
    runtime = make_runtime()
    try:
        runtime.handle_key(ord("a"))
        assert runtime.controller.armed
        assert runtime.controller.state == OperationState.WAITING_FOR_VEHICLE
    finally:
        runtime.close()


def test_parking_complete_after_arm_starts_set_camera():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n")); pump(runtime)
        assert ("left", "SET_CAMERA") in arm.commands
        assert ("right", "SET_CAMERA") in arm.commands
    finally:
        runtime.close()


def test_full_bad_left_flow_completed():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(100):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.COMPLETED:
                break
        assert runtime.controller.state == OperationState.COMPLETED
        assert arm.commands == [("left", "SET_CAMERA"), ("right", "SET_CAMERA"), ("right", "HELP"), ("left", "REMOVE_BAD_TIRE")]
    finally:
        runtime.close()


def test_set_camera_one_side_failure_error():
    runtime = make_runtime(arm=FakeArmManager(fail={("right", "SET_CAMERA")}))
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(50):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.ERROR:
                break
        assert runtime.controller.state == OperationState.ERROR
    finally:
        runtime.close()


def test_detection_timeout_error():
    runtime = make_runtime(vision=FakeVisionClient(error="vision response timeout"))
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(80):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.ERROR:
                break
        assert runtime.controller.state == OperationState.ERROR
        assert "timeout" in runtime.controller.last_error
    finally:
        runtime.close()


def test_right_help_failure_blocks_left_remove():
    arm = FakeArmManager(fail={("right", "HELP")})
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(100):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.ERROR:
                break
        assert runtime.controller.state == OperationState.ERROR
        assert ("left", "REMOVE_BAD_TIRE") not in arm.commands
    finally:
        runtime.close()


def test_left_good_skips_help_and_remove():
    response = {
        "left": {"class_name": "good_tire", "confidence": 0.91, "stable": True},
        "right": {"class_name": "bad_tire", "confidence": 0.88, "stable": True},
    }
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm, vision=FakeVisionClient(response=response))
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(100):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.COMPLETED:
                break
        assert runtime.controller.state == OperationState.COMPLETED
        assert ("right", "HELP") not in arm.commands
        assert ("left", "REMOVE_BAD_TIRE") not in arm.commands
    finally:
        runtime.close()


def test_completed_requires_r_before_reexecution():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(100):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.COMPLETED:
                break
        count = len(arm.commands)
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n")); pump(runtime)
        assert len(arm.commands) == count
        runtime.handle_key(ord("r"))
        assert runtime.controller.state == OperationState.IDLE
    finally:
        runtime.close()


def test_stop_priority():
    runtime = make_runtime()
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n")); runtime.handle_key(ord("s"))
        assert runtime.controller.state == OperationState.STOPPED
    finally:
        runtime.close()

def prime_safety(runtime, state):
    runtime.controller.state = state
    runtime.controller.armed = True
    runtime.controller.cycle_latched = True
    runtime.safety_reference_center = (100.0, 100.0)


def wait_emergency(runtime, limit=100):
    for _ in range(limit):
        runtime._apply_emergency_result_if_ready()
        runtime._apply_worker_result_if_ready()
        if runtime.controller.state == OperationState.EMERGENCY:
            return
        time.sleep(0.005)
    raise AssertionError("emergency sequence did not finish")


def evaluate_motion(runtime, now, center=(120.0, 100.0), jitter=13.0, speed=7.0):
    return runtime._evaluate_safety_metrics(
        current_center=center,
        jitter=jitter,
        speed=speed,
        vehicle_detected=True,
        now=now,
    )


def test_motion_before_work_does_not_start_emergency_home():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.controller.state = OperationState.WAITING_FOR_VEHICLE
        runtime.safety_reference_center = (100.0, 100.0)
        assert not evaluate_motion(runtime, 0.0)
        assert not evaluate_motion(runtime, 1.0)
        assert not runtime.controller.emergency_latched
        assert arm.commands == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "state",
    [
        OperationState.SETTING_CAMERA,
        OperationState.DETECTING_TIRES,
        OperationState.CHECKING_LEFT_TIRE,
        OperationState.EXECUTING_RIGHT_HELP,
        OperationState.EXECUTING_LEFT_REMOVE,
    ],
)
def test_sustained_motion_triggers_emergency_in_each_work_state(state):
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        prime_safety(runtime, state)
        assert not evaluate_motion(runtime, 0.0)
        assert evaluate_motion(runtime, 0.36)
        wait_emergency(runtime)
        assert runtime.controller.state == OperationState.EMERGENCY
        assert runtime.controller.emergency_latched
        assert ("left", "STOP") in arm.commands
        assert ("right", "STOP") in arm.commands
        assert ("left", "HOME") in arm.commands
        assert ("right", "HOME") in arm.commands
    finally:
        runtime.close()


def test_single_frame_bbox_jump_is_ignored():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.SETTING_CAMERA)
        assert not evaluate_motion(runtime, 0.0)
        assert not evaluate_motion(
            runtime, 0.1, center=(100.0, 100.0), jitter=1.0, speed=1.0
        )
        assert not evaluate_motion(
            runtime, 0.6, center=(100.0, 100.0), jitter=1.0, speed=1.0
        )
        assert not runtime.controller.emergency_latched
    finally:
        runtime.close()


def test_motion_shorter_than_hold_is_ignored():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.DETECTING_TIRES)
        assert not evaluate_motion(runtime, 0.0)
        assert not evaluate_motion(runtime, 0.34)
        assert not runtime.controller.emergency_latched
    finally:
        runtime.close()


def test_reference_displacement_alone_triggers_emergency():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.EXECUTING_RIGHT_HELP)
        assert not evaluate_motion(
            runtime, 0.0, center=(113.0, 100.0), jitter=1.0, speed=1.0
        )
        assert evaluate_motion(
            runtime, 0.36, center=(113.0, 100.0), jitter=1.0, speed=1.0
        )
        wait_emergency(runtime)
        assert "displacement" in runtime.controller.emergency_reason
    finally:
        runtime.close()


def test_short_detection_loss_is_ignored():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.DETECTING_TIRES)
        assert not runtime._evaluate_safety_metrics(None, None, None, False, 0.0)
        assert not runtime._evaluate_safety_metrics(None, None, None, False, 0.99)
        assert not runtime.controller.emergency_latched
    finally:
        runtime.close()


def test_long_detection_loss_triggers_emergency():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.DETECTING_TIRES)
        assert not runtime._evaluate_safety_metrics(None, None, None, False, 0.0)
        assert runtime._evaluate_safety_metrics(None, None, None, False, 1.0)
        wait_emergency(runtime)
        assert "detection lost" in runtime.controller.emergency_reason
    finally:
        runtime.close()


def test_stop_failure_still_sends_home_to_both_arms():
    arm = FakeArmManager(fail={("right", "STOP")})
    runtime = make_runtime(arm=arm)
    try:
        prime_safety(runtime, OperationState.EXECUTING_LEFT_REMOVE)
        assert runtime._trigger_vehicle_emergency("test motion")
        wait_emergency(runtime)
        assert not runtime.emergency_stop_results["right"].success
        assert ("left", "HOME") in arm.commands
        assert ("right", "HOME") in arm.commands
        assert runtime.controller.state == OperationState.EMERGENCY
    finally:
        runtime.close()


def test_late_general_worker_result_is_ignored_after_emergency():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    release = threading.Event()
    try:
        prime_safety(runtime, OperationState.SETTING_CAMERA)

        def late_set_camera():
            release.wait(timeout=1.0)
            return arm.send_both("SET_CAMERA")

        runtime.worker_operation = "SET_CAMERA"
        runtime.worker_future = runtime.executor.submit(late_set_camera)
        assert runtime._trigger_vehicle_emergency("vehicle moved")
        wait_emergency(runtime)
        release.set()
        for _ in range(100):
            runtime._apply_worker_result_if_ready()
            if runtime.worker_future is None:
                break
            time.sleep(0.005)
        assert runtime.controller.state == OperationState.EMERGENCY
        assert runtime.controller.last_successful_step == "EMERGENCY_HOME"
    finally:
        release.set()
        runtime.close()


def test_emergency_latch_prevents_duplicate_stop_home_sequences():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        prime_safety(runtime, OperationState.SETTING_CAMERA)
        assert runtime._trigger_vehicle_emergency("first")
        assert not runtime._trigger_vehicle_emergency("duplicate")
        wait_emergency(runtime)
        assert sum(command == ("left", "STOP") for command in arm.commands) == 1
        assert sum(command == ("right", "STOP") for command in arm.commands) == 1
        assert sum(command == ("left", "HOME") for command in arm.commands) == 1
        assert sum(command == ("right", "HOME") for command in arm.commands) == 1
    finally:
        runtime.close()


def test_emergency_requires_r_then_a_before_new_cycle():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.SETTING_CAMERA)
        runtime.safety_reference_center = (100.0, 100.0)
        assert runtime._trigger_vehicle_emergency("vehicle moved")
        wait_emergency(runtime)

        runtime.handle_key(ord("a"))
        runtime.handle_key(ord("n"))
        assert runtime.controller.state == OperationState.EMERGENCY
        assert not runtime.controller.armed

        runtime.handle_key(ord("r"))
        assert runtime.controller.state == OperationState.IDLE
        assert not runtime.controller.emergency_latched
        assert runtime.safety_reference_center is None
        runtime.handle_key(ord("a"))
        assert runtime.controller.armed
        assert runtime.controller.state == OperationState.WAITING_FOR_VEHICLE
    finally:
        runtime.close()


def test_manual_s_stop_does_not_automatically_home():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("s"))
        for _ in range(100):
            if len(arm.commands) >= 2:
                break
            time.sleep(0.005)
        assert runtime.controller.state == OperationState.STOPPED
        assert ("left", "STOP") in arm.commands
        assert ("right", "STOP") in arm.commands
        assert all(command != "HOME" for _, command in arm.commands)
    finally:
        runtime.close()


def test_parking_completion_captures_filtered_reference_center():
    runtime = make_runtime()
    try:
        runtime.controller.arm_system()
        runtime.controller.on_vehicle_detected()
        runtime.parking.parking_monitor.filtered_center = (305.0, 324.0)
        result = SimpleNamespace(center=(306, 325))
        runtime._on_parking_completed(result=result, now=10.0)
        assert runtime.safety_reference_center == (305.0, 324.0)
    finally:
        runtime.close()

def test_r_is_rejected_while_emergency_sequence_is_running():
    runtime = make_runtime()
    try:
        prime_safety(runtime, OperationState.SETTING_CAMERA)
        assert runtime._trigger_vehicle_emergency("vehicle moved")
        runtime.handle_key(ord("r"))
        assert runtime.controller.state == OperationState.EMERGENCY_STOPPING
        assert runtime.controller.emergency_latched
        wait_emergency(runtime)
    finally:
        runtime.close()


def test_home_disabled_still_latches_emergency_and_sends_stop():
    config = deepcopy(AUTO_CONFIG)
    config["safety"]["home_on_vehicle_motion"] = False
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm, auto_config=config)
    try:
        prime_safety(runtime, OperationState.DETECTING_TIRES)
        assert runtime._trigger_vehicle_emergency("vehicle moved")
        wait_emergency(runtime)
        assert runtime.controller.state == OperationState.EMERGENCY
        assert ("left", "STOP") in arm.commands
        assert ("right", "STOP") in arm.commands
        assert all(command != "HOME" for _, command in arm.commands)
    finally:
        runtime.close()

