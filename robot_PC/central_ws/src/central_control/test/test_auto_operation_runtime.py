from copy import deepcopy
import threading
import time
from types import SimpleNamespace

import pytest

from central_control.arm_client import ArmCommandResult
from central_control.auto_operation_node import (
    AutoOperationRuntime,
    apply_manual_test_mode_config,
)
from central_control.manual_test_input import ManualTestVisionClient
from central_control.operation_controller import OperationState
from central_control.transport_operations import (
    MockTransportOperationHandler,
    TransportOperationResult,
)
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
    "operation": {
        "auto_start": False,
        "mock_arms": True,
        "mock_vision": True,
        "detection_max_retries": 3,
        "mock_transport_operations": False,
        "mock_aruco_transport_operations": True,
        "mock_install_new_tire": True,
        "detection_retry_delay_sec": 0.0,
    },
    "delays": {"after_parking_sec": 0.0, "after_set_camera_sec": 0.0, "after_detection_sec": 0.0, "after_opposite_help_sec": 0.0},
    "commands": {"set_camera": "SET_CAMERA", "help": "HELP", "remove_bad_tire": "REMOVE_BAD_TIRE", "set_aruco": "SET_ARUCO", "install_new_tire": "INSTALL_NEW_TIRE", "home": "HOME", "stop": "STOP"},
    "timeouts": {"worker_shutdown_sec": 0.1, "pinky_move_sec": 180.0},
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
    def send_side(self, side, command):
        if side == "left":
            return self.send_left(command)
        if side == "right":
            return self.send_right(command)
        raise ValueError(f"unknown side: {side}")


    def send_left(self, command):
        return {"left": self._result("left", command)}

    @staticmethod
    def all_success(results):
        return bool(results) and all(r.success for r in results.values())


class TimeoutSetArucoArm(FakeArmManager):
    def send_side(self, side, command):
        if command == "SET_ARUCO":
            self.commands.append((side, command))
            return {
                side: ArmCommandResult(
                    side, command, False, error="arm command timeout"
                )
            }
        return super().send_side(side, command)


class BlockingSetArucoArm(FakeArmManager):
    def __init__(self):
        super().__init__()
        self.set_aruco_started = threading.Event()
        self.release_set_aruco = threading.Event()

    def send_side(self, side, command):
        if command == "SET_ARUCO":
            self.commands.append((side, command))
            self.set_aruco_started.set()
            self.release_set_aruco.wait(timeout=2.0)
            return {side: ArmCommandResult(side, command, True, response="OK")}
        return super().send_side(side, command)


class BlockingTransportOperationHandler:
    def __init__(self, result=None):
        self.result = result or TransportOperationResult(True, "late success")
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def place_removed_tire(self, side, slot):
        self.calls.append(("PLACE_REMOVED_TIRE", side, slot))
        self.started.set()
        self.release.wait(timeout=2.0)
        return self.result

    def pick_new_tire(self, side, slot):
        self.calls.append(("PICK_NEW_TIRE", side, slot))
        return TransportOperationResult(True, "pick success")


class FailedTransportOperationHandler:
    def __init__(self, message):
        self.message = message
        self.calls = []

    def place_removed_tire(self, side, slot):
        self.calls.append(("PLACE_REMOVED_TIRE", side, slot))
        return TransportOperationResult(False, self.message)

    def pick_new_tire(self, side, slot):
        raise AssertionError("PICK_NEW_TIRE must not run after PLACE failure")


class FakeVisionClient:
    def __init__(self, response=None, error=None):
        self.response = response or DETECTION
        self.error = error
        self.calls = 0

    def request_detection(self, reset_cycle=True):
        self.calls += 1
        if self.error:
            return VisionClientResult(False, error=self.error, request_id="detect-test")
        return VisionClientResult(True, response=self.response, request_id="detect-test")


class SequenceVisionClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.reset_flags = []

    def request_detection(self, reset_cycle=True):
        self.reset_flags.append(reset_cycle)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return VisionClientResult(True, response=response, request_id=f"detect-{self.calls}")


def side_result(status):
    class_name = {"GOOD": "good_tire", "BAD": "bad_tire"}.get(status, "none")
    return {
        "status": status,
        "class_name": class_name,
        "confidence": .9 if status in {"GOOD", "BAD"} else 0.0,
        "stable": status in {"GOOD", "BAD"},
        "votes": 3 if status in {"GOOD", "BAD"} else 0,
        "frames": 10,
        "detections": [],
    }


def detection_response(left_status, right_status):
    return {
        "left": side_result(left_status),
        "right": side_result(right_status),
    }


def make_runtime(
    arm=None, vision=None, auto_config=None, pinky_commands=None,
    pinky_subscription_count=None, transport_operations=None,
    test_mode=False, hardware_enabled=False,
):
    commands = [] if pinky_commands is None else pinky_commands
    parking = deepcopy(PARKING_CONFIG)
    yolo = deepcopy(YOLO_CONFIG)
    config = deepcopy(auto_config or AUTO_CONFIG)
    apply_manual_test_mode_config(
        parking, yolo, config, test_mode, hardware_enabled
    )
    runtime = AutoOperationRuntime(
        parking,
        yolo,
        deepcopy(NETWORK_CONFIG),
        config,
        arm_manager=arm or FakeArmManager(),
        vision_client=vision or FakeVisionClient(),
        transport_operations=transport_operations,
        pinky_command_sender=commands.append,
        pinky_subscription_count=pinky_subscription_count or (lambda: 1),
        test_mode=test_mode,
        hardware_enabled=hardware_enabled,
    )
    runtime.test_pinky_commands = commands
    return runtime


def pump(runtime, limit=100):
    for _ in range(limit):
        runtime.step()
        if runtime.worker_future is None and not runtime.controller.busy:
            if runtime.controller.state in (OperationState.COMPLETED, OperationState.ERROR, OperationState.STOPPED, OperationState.PARKING_STABLE, OperationState.IDLE, OperationState.ARMED, OperationState.WAITING_FOR_VEHICLE):
                return
        time.sleep(0.01)


def test_overlay_hides_only_mock_device_status_lines():
    lines = AutoOperationRuntime._connection_status_lines(
        {"vision": "mock", "left_arm": "ok", "right_arm": "error"}
    )

    assert lines == ["Left arm: ok", "Right arm: error"]


def wait_for_state(runtime, state, limit=300):
    for _ in range(limit):
        runtime.step()
        if runtime.controller.state == state:
            return
        time.sleep(0.005)
    raise AssertionError(f"state {state.value} not reached: {runtime.controller.state.value}")


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


def test_parking_complete_never_calls_set_camera():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n")); pump(runtime)
        assert all(command != "SET_CAMERA" for _, command in arm.commands)
    finally:
        runtime.close()


def test_full_bad_left_flow_completed():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.transport_operations.calls == []
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.COMPLETED)
        assert arm.commands[:4] == [
            ("left", "SET_ARUCO"),
            ("right", "HELP"),
            ("left", "REMOVE_BAD_TIRE"),
            ("left", "INSTALL_NEW_TIRE"),
        ]
        assert set(arm.commands[-2:]) == {("left", "HOME"), ("right", "HOME")}
        assert runtime.transport_operations.calls == [
            ("PLACE_REMOVED_TIRE", "left", 3),
            ("PICK_NEW_TIRE", "left", 4),
        ]
    finally:
        runtime.close()


def test_set_aruco_completion_is_a_hard_gate_before_pinky_start():
    arm = BlockingSetArucoArm()
    commands = []
    runtime = make_runtime(arm=arm, pinky_commands=commands)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(200):
            runtime.step()
            if arm.set_aruco_started.wait(timeout=0.005):
                break
        assert arm.set_aruco_started.is_set()
        for _ in range(20):
            runtime.step(); time.sleep(0.002)
        assert runtime.controller.state == OperationState.WAIT_SET_ARUCO_DONE
        assert commands == []

        arm.release_set_aruco.set()
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert commands == ["START:tire_stop_1"]
        assert arm.commands[:3] == [
            ("left", "SET_ARUCO"),
            ("right", "HELP"),
            ("left", "REMOVE_BAD_TIRE"),
        ]
    finally:
        arm.release_set_aruco.set()
        runtime.close()


def test_right_set_aruco_completes_before_tire_stop_2(capsys):
    commands = []
    arm = FakeArmManager()
    runtime = make_runtime(
        arm=arm,
        vision=FakeVisionClient(detection_response("GOOD", "BAD")),
        pinky_commands=commands,
    )
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert commands == ["START:tire_stop_2"]
        assert arm.commands[:3] == [
            ("right", "SET_ARUCO"),
            ("left", "HELP"),
            ("right", "REMOVE_BAD_TIRE"),
        ]
        output = capsys.readouterr().out
        requested = output.index("[RIGHT] SET_ARUCO requested")
        completed = output.index("[RIGHT] SET_ARUCO completed")
        started = output.index("[RIGHT] Starting Pinky route tire_stop_2")
        assert requested < completed < started
    finally:
        runtime.close()


def test_both_bad_starts_each_route_only_after_its_set_aruco():
    commands = []
    arm = FakeArmManager()
    runtime = make_runtime(
        arm=arm,
        vision=FakeVisionClient(detection_response("BAD", "BAD")),
        pinky_commands=commands,
    )
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.current_side == "left"
        assert commands == ["START:tire_stop_1"]
        assert arm.commands[:3] == [
            ("left", "SET_ARUCO"),
            ("right", "HELP"),
            ("left", "REMOVE_BAD_TIRE"),
        ]

        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.current_side == "right"
        assert commands == ["START:tire_stop_1", "START:tire_stop_2"]
        left_install = arm.commands.index(("left", "INSTALL_NEW_TIRE"))
        right_aruco = arm.commands.index(("right", "SET_ARUCO"))
        right_help = arm.commands.index(("left", "HELP"), right_aruco)
        right_remove = arm.commands.index(("right", "REMOVE_BAD_TIRE"))
        assert left_install < right_aruco < right_help < right_remove
    finally:
        runtime.close()


def test_set_aruco_failure_or_timeout_never_starts_pinky():
    for arm in (
        FakeArmManager(fail={("left", "SET_ARUCO")}),
        TimeoutSetArucoArm(),
    ):
        commands = []
        runtime = make_runtime(arm=arm, pinky_commands=commands)
        try:
            runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
            wait_for_state(runtime, OperationState.ERROR)
            assert commands == []
            assert ("left", "SET_ARUCO") in arm.commands
            assert all(command not in {"HELP", "REMOVE_BAD_TIRE"} for _, command in arm.commands)
        finally:
            runtime.close()


def test_set_camera_failure_configuration_is_irrelevant():
    arm = FakeArmManager(fail={("right", "SET_CAMERA")})
    runtime = make_runtime(arm=arm)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n")); pump(runtime)
        assert all(command != "SET_CAMERA" for _, command in arm.commands)
        assert runtime.controller.state != OperationState.ERROR
    finally:
        runtime.close()


def test_detection_timeout_error():
    runtime = make_runtime(vision=FakeVisionClient(error="vision response timeout"))
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(80):
            runtime.step(); time.sleep(0.01)
            if runtime.controller.state == OperationState.DETECTION_ERROR:
                break
        assert runtime.controller.state == OperationState.DETECTION_ERROR
        assert "timeout" in runtime.controller.last_error
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "first",
    [
        ("RECHECK", "GOOD"),
        ("BAD", "RECHECK"),
        ("RECHECK", "RECHECK"),
    ],
)
def test_any_recheck_discards_both_and_redetects_all(first):
    vision = SequenceVisionClient([
        detection_response(*first),
        detection_response("BAD", "GOOD"),
    ])
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm, vision=vision)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(120):
            runtime.step(); time.sleep(.005)
            if ("left", "REMOVE_BAD_TIRE") in arm.commands:
                break
        assert vision.calls == 2
        assert vision.reset_flags == [True, True]
        assert runtime.controller.target_arms == {"left"}
        assert ("left", "REMOVE_BAD_TIRE") in arm.commands
    finally:
        runtime.close()


def test_three_rechecks_enter_detection_error_without_remove():
    vision = SequenceVisionClient([
        detection_response("RECHECK", "RECHECK"),
    ])
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm, vision=vision)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        for _ in range(120):
            runtime.step(); time.sleep(.005)
            if runtime.controller.state == OperationState.DETECTION_ERROR:
                break
        assert vision.calls == 3
        assert runtime.controller.state == OperationState.DETECTION_ERROR
        assert runtime.controller.left_detection is None
        assert runtime.controller.right_detection is None
        assert all(command != "REMOVE_BAD_TIRE" for _, command in arm.commands)
    finally:
        runtime.close()


def test_emergency_during_redetection_discards_late_result():
    arm = FakeArmManager()
    release = threading.Event()

    class BlockingVision:
        def request_detection(self, reset_cycle=True):
            release.wait(timeout=1.0)
            return VisionClientResult(
                True,
                response=detection_response("BAD", "GOOD"),
                request_id="late",
            )

    runtime = make_runtime(arm=arm, vision=BlockingVision())
    try:
        prime_safety(runtime, OperationState.DETECTING_TIRES)
        runtime._advance_auto_workers()
        assert runtime.worker_future is not None
        assert runtime._trigger_vehicle_emergency("vehicle moved during redetection")
        release.set()
        wait_emergency(runtime)
        for _ in range(100):
            runtime._apply_worker_result_if_ready()
            if runtime.worker_future is None:
                break
            time.sleep(.005)
        assert runtime.controller.state == OperationState.EMERGENCY
        assert runtime.controller.left_detection is None
        assert all(command != "REMOVE_BAD_TIRE" for _, command in arm.commands)
    finally:
        release.set()
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


def test_left_good_right_bad_runs_right_cycle():
    response = {
        "left": {"class_name": "good_tire", "confidence": 0.91, "stable": True},
        "right": {"class_name": "bad_tire", "confidence": 0.88, "stable": True},
    }
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm, vision=FakeVisionClient(response=response))
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.replacement_plan == ("right",)
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.COMPLETED)
        assert arm.commands[:4] == [
            ("right", "SET_ARUCO"),
            ("left", "HELP"),
            ("right", "REMOVE_BAD_TIRE"),
            ("right", "INSTALL_NEW_TIRE"),
        ]
        assert runtime.transport_operations.calls == [
            ("PLACE_REMOVED_TIRE", "right", 1),
            ("PICK_NEW_TIRE", "right", 2),
        ]
    finally:
        runtime.close()


def test_completed_requires_r_before_reexecution():
    arm = FakeArmManager()
    runtime = make_runtime(arm=arm, vision=FakeVisionClient(detection_response("GOOD", "GOOD")))
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.COMPLETED)
        assert set(arm.commands) == {("left", "HOME"), ("right", "HOME")}
        count = len(arm.commands)
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n")); pump(runtime)
        assert len(arm.commands) == count
        runtime.handle_key(ord("r"))
        assert runtime.controller.state == OperationState.IDLE
    finally:
        runtime.close()


def test_both_bad_completes_left_before_right_starts_remove():
    arm = FakeArmManager()
    vision = FakeVisionClient(detection_response("BAD", "BAD"))
    runtime = make_runtime(arm=arm, vision=vision)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.current_side == "left"
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        left_install_index = arm.commands.index(("left", "INSTALL_NEW_TIRE"))
        right_remove_index = arm.commands.index(("right", "REMOVE_BAD_TIRE"))
        assert left_install_index < right_remove_index
        assert runtime.controller.current_side == "right"
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.COMPLETED)
        assert runtime.controller.replacement_plan == ("left", "right")
        assert runtime.transport_operations.calls == [
            ("PLACE_REMOVED_TIRE", "left", 3),
            ("PICK_NEW_TIRE", "left", 4),
            ("PLACE_REMOVED_TIRE", "right", 1),
            ("PICK_NEW_TIRE", "right", 2),
        ]
    finally:
        runtime.close()

def test_waiting_transport_never_starts_place_before_ready():
    runtime = make_runtime()
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        for _ in range(20):
            runtime.step()
            time.sleep(0.002)
        assert runtime.controller.state == OperationState.WAITING_TRANSPORT
        assert runtime.transport_operations.calls == []
    finally:
        runtime.close()


def test_place_failure_blocks_pick_and_install():
    arm = FakeArmManager()
    transport = MockTransportOperationHandler({"PLACE_REMOVED_TIRE"})
    runtime = make_runtime(arm=arm, transport_operations=transport)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.ERROR)
        assert transport.calls == [("PLACE_REMOVED_TIRE", "left", 3)]
        assert ("left", "INSTALL_NEW_TIRE") not in arm.commands
        assert ("left", "HOME") not in arm.commands
        assert ("right", "HOME") not in arm.commands
    finally:
        runtime.close()


def test_pick_failure_blocks_install():
    arm = FakeArmManager()
    transport = MockTransportOperationHandler({"PICK_NEW_TIRE"})
    runtime = make_runtime(arm=arm, transport_operations=transport)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.ERROR)
        assert transport.calls == [
            ("PLACE_REMOVED_TIRE", "left", 3),
            ("PICK_NEW_TIRE", "left", 4),
        ]
        assert ("left", "INSTALL_NEW_TIRE") not in arm.commands
    finally:
        runtime.close()


def test_transport_timeout_enters_error_without_retry_or_pick():
    transport = FailedTransportOperationHandler(
        "PLACE_REMOVED_TIRE timeout after 0.050s side=left slot=3"
    )
    runtime = make_runtime(transport_operations=transport)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.ERROR)
        assert transport.calls == [("PLACE_REMOVED_TIRE", "left", 3)]
        assert "timeout" in runtime.controller.last_error
    finally:
        runtime.close()


def test_repeated_timer_does_not_duplicate_in_flight_transport_call():
    transport = BlockingTransportOperationHandler()
    runtime = make_runtime(transport_operations=transport)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        runtime._advance_auto_workers()
        assert transport.started.wait(timeout=1.0)
        for _ in range(20):
            runtime._advance_auto_workers()
        assert transport.calls == [("PLACE_REMOVED_TIRE", "left", 3)]
    finally:
        transport.release.set()
        runtime.close()


def test_late_transport_success_is_ignored_after_emergency():
    arm = FakeArmManager()
    transport = BlockingTransportOperationHandler()
    runtime = make_runtime(arm=arm, transport_operations=transport)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        runtime._advance_auto_workers()
        assert transport.started.wait(timeout=1.0)
        assert runtime._trigger_vehicle_emergency("vehicle moved during PLACE")
        transport.release.set()
        wait_emergency(runtime)
        for _ in range(100):
            runtime._apply_worker_result_if_ready()
            if runtime.worker_future is None:
                break
            time.sleep(0.005)
        assert runtime.controller.state == OperationState.EMERGENCY
        assert transport.calls == [("PLACE_REMOVED_TIRE", "left", 3)]
        assert runtime.controller.last_successful_step == "EMERGENCY_HOME"
    finally:
        transport.release.set()
        runtime.close()


def test_install_failure_blocks_next_side_and_home():
    arm = FakeArmManager(fail={("left", "INSTALL_NEW_TIRE")})
    runtime = make_runtime(
        arm=arm,
        vision=FakeVisionClient(detection_response("BAD", "BAD")),
    )
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.ERROR)
        assert ("right", "REMOVE_BAD_TIRE") not in arm.commands
        assert ("left", "HOME") not in arm.commands
        assert ("right", "HOME") not in arm.commands
    finally:
        runtime.close()

def test_install_is_not_implemented_without_explicit_mock_mode():
    arm = FakeArmManager()
    config = deepcopy(AUTO_CONFIG)
    config["operation"]["mock_install_new_tire"] = False
    runtime = make_runtime(arm=arm, auto_config=config)
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.handle_transport_ready()
        wait_for_state(runtime, OperationState.ERROR)
        assert "NOT_IMPLEMENTED" in runtime.controller.last_error
        assert ("left", "INSTALL_NEW_TIRE") not in arm.commands
    finally:
        runtime.close()


def test_one_home_failure_enters_error():
    arm = FakeArmManager(fail={("right", "HOME")})
    runtime = make_runtime(
        arm=arm,
        vision=FakeVisionClient(detection_response("GOOD", "GOOD")),
    )
    try:
        runtime.handle_key(ord("a")); runtime.handle_key(ord("n"))
        wait_for_state(runtime, OperationState.ERROR)
        assert ("left", "HOME") in arm.commands
        assert ("right", "HOME") in arm.commands
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
        OperationState.HELPING_OPPOSITE_ARM,
        OperationState.REMOVING_BAD_TIRE,
        OperationState.SETTING_ARUCO,
        OperationState.WAITING_TRANSPORT,
        OperationState.PLACING_REMOVED_TIRE,
        OperationState.PICKING_NEW_TIRE,
        OperationState.INSTALLING_NEW_TIRE,
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


def test_pinky_waits_for_subscriber_and_publishes_start_once():
    subscribers = [0]
    commands = []
    runtime = make_runtime(
        pinky_commands=commands,
        pinky_subscription_count=lambda: subscribers[0],
    )
    try:
        runtime.controller.state = OperationState.REQUESTING_PINKY_MOVE
        runtime._advance_pinky_move(now=10.0)
        runtime._advance_pinky_move(now=11.0)
        assert runtime.controller.state == OperationState.REQUESTING_PINKY_MOVE
        assert commands == []

        subscribers[0] = 1
        runtime._advance_pinky_move(now=12.0)
        runtime._advance_pinky_move(now=13.0)
        assert runtime.controller.state == OperationState.WAITING_PINKY_COMPLETE
        assert commands == ["START:after_tire_service"]
    finally:
        runtime.close()

def test_pinky_completed_error_timeout_stop_and_reset():
    commands = []
    runtime = make_runtime(pinky_commands=commands)
    try:
        runtime.controller.state = OperationState.REQUESTING_PINKY_MOVE
        runtime._advance_pinky_move(now=10.0)
        runtime.handle_pinky_route_status("ERROR:blocked")
        assert runtime.controller.state == OperationState.ERROR
        assert "blocked" in runtime.controller.last_error

        runtime.reset_cycle()
        assert not runtime.pinky_start_sent
        assert runtime.pinky_move_started_at is None

        runtime.controller.state = OperationState.REQUESTING_PINKY_MOVE
        runtime._advance_pinky_move(now=20.0)
        runtime._advance_pinky_move(now=200.0)
        assert runtime.controller.state == OperationState.ERROR
        assert "timeout" in runtime.controller.last_error

        runtime.start_stop()
        assert commands[-1] == "STOP"
    finally:
        runtime.close()


def make_manual_runtime(arm=None, transport_operations=None):
    return make_runtime(
        arm=arm,
        vision=ManualTestVisionClient(),
        transport_operations=transport_operations,
        test_mode=True,
        hardware_enabled=False,
    )


def submit_manual_detection(runtime, key):
    assert runtime.handle_test_key("p") == "PARKED"
    wait_for_state(runtime, OperationState.DETECTING_TIRES)
    assert runtime.handle_test_key(key) == "TIRE_RESULT"


def test_manual_keys_good_good_returns_home_and_completes():
    arm = FakeArmManager()
    runtime = make_manual_runtime(arm=arm)
    try:
        submit_manual_detection(runtime, "1")
        wait_for_state(runtime, OperationState.COMPLETED)
        assert runtime.controller.replacement_plan == ()
        assert set(arm.commands) == {("left", "HOME"), ("right", "HOME")}
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "key,plan,side,transport_calls",
    [
        (
            "2",
            ("left",),
            "left",
            [
                ("PLACE_REMOVED_TIRE", "left", 3),
                ("PICK_NEW_TIRE", "left", 4),
            ],
        ),
        (
            "3",
            ("right",),
            "right",
            [
                ("PLACE_REMOVED_TIRE", "right", 1),
                ("PICK_NEW_TIRE", "right", 2),
            ],
        ),
    ],
)
def test_manual_keys_single_bad_follow_existing_policy(
    key, plan, side, transport_calls
):
    runtime = make_manual_runtime()
    try:
        submit_manual_detection(runtime, key)
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.replacement_plan == plan
        assert runtime.controller.current_side == side
        assert runtime.handle_test_key("t") == "TRANSPORT_READY"
        wait_for_state(runtime, OperationState.COMPLETED)
        assert runtime.transport_operations.calls == transport_calls
    finally:
        runtime.close()


def test_manual_key_both_bad_finishes_left_before_right():
    arm = FakeArmManager()
    runtime = make_manual_runtime(arm=arm)
    try:
        submit_manual_detection(runtime, "4")
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.current_side == "left"
        assert runtime.handle_test_key("t") == "TRANSPORT_READY"
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.current_side == "right"
        assert arm.commands.index(("left", "INSTALL_NEW_TIRE")) < (
            arm.commands.index(("right", "REMOVE_BAD_TIRE"))
        )
        assert runtime.handle_test_key("t") == "TRANSPORT_READY"
        wait_for_state(runtime, OperationState.COMPLETED)
        assert runtime.controller.replacement_plan == ("left", "right")
    finally:
        runtime.close()


def test_manual_transport_ready_is_rejected_early_and_when_repeated():
    runtime = make_manual_runtime()
    try:
        assert runtime.handle_test_key("t") == "IGNORED"
        submit_manual_detection(runtime, "2")
        assert runtime.handle_test_key("t") == "IGNORED"
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.transport_operations.calls == []
        assert runtime.handle_test_key("t") == "TRANSPORT_READY"
        assert runtime.handle_test_key("t") == "IGNORED"
        wait_for_state(runtime, OperationState.COMPLETED)
        assert runtime.transport_operations.calls == [
            ("PLACE_REMOVED_TIRE", "left", 3),
            ("PICK_NEW_TIRE", "left", 4),
        ]
    finally:
        runtime.close()


def test_manual_stop_and_emergency_reuse_existing_paths():
    stopped = make_manual_runtime()
    try:
        assert stopped.handle_test_key("s") == "STOP"
        assert stopped.controller.state == OperationState.STOPPED
    finally:
        stopped.close()

    emergency = make_manual_runtime()
    try:
        assert emergency.handle_test_key("e") == "EMERGENCY"
        wait_emergency(emergency)
        assert emergency.controller.state == OperationState.EMERGENCY
        assert emergency.controller.emergency_latched
        assert emergency.handle_test_key("r") == "RESET"
        assert emergency.controller.state == OperationState.IDLE
        assert not emergency.controller.emergency_latched
    finally:
        emergency.close()


def test_manual_test_mode_without_hardware_forces_existing_mocks():
    parking = deepcopy(PARKING_CONFIG)
    yolo = deepcopy(YOLO_CONFIG)
    config = deepcopy(AUTO_CONFIG)
    apply_manual_test_mode_config(
        parking, yolo, config, test_mode=True, hardware_enabled=False
    )
    assert parking["mock_camera"] is True
    assert yolo["mock_vehicle_detector"] is True
    assert config["operation"]["mock_arms"] is True
    assert config["operation"]["mock_transport_operations"] is False
    assert config["operation"]["mock_aruco_transport_operations"] is True
    assert config["operation"]["mock_install_new_tire"] is True


def test_production_config_is_unchanged_when_test_mode_is_false():
    parking = deepcopy(PARKING_CONFIG)
    yolo = deepcopy(YOLO_CONFIG)
    config = deepcopy(AUTO_CONFIG)
    original = deepcopy((parking, yolo, config))
    apply_manual_test_mode_config(
        parking, yolo, config, test_mode=False, hardware_enabled=False
    )
    assert (parking, yolo, config) == original


def test_manual_status_reports_policy_and_safety_flags():
    runtime = make_manual_runtime()
    try:
        status = runtime.test_status()
        assert status["TEST_MODE"] is True
        assert status["HARDWARE_ENABLED"] is False
        assert status["TRANSPORT_SLOTS"] == {
            "left": {"removed_tire": 3, "new_tire": 4},
            "right": {"removed_tire": 1, "new_tire": 2},
        }
    finally:
        runtime.close()



def test_keyboard_queue_processes_one_per_step_without_direct_commands():
    runtime = make_manual_runtime()
    try:
        runtime.enqueue_test_key("p")
        runtime.enqueue_test_key("2")
        runtime.step()
        assert runtime.controller.state == OperationState.DETECTING_TIRES
        runtime.step()
        wait_for_state(runtime, OperationState.WAITING_TRANSPORT)
        assert runtime.controller.replacement_plan == ("left",)
    finally:
        runtime.close()


def test_q_stops_keyboard_input_only():
    runtime = make_manual_runtime()
    try:
        assert runtime.handle_test_key("q") == "QUIT_KEYBOARD"
        assert not runtime.quit_requested
        assert runtime.controller.state == OperationState.IDLE
    finally:
        runtime.close()



def full_auto_without_pinky_config():
    config = deepcopy(AUTO_CONFIG)
    config["operation"].update(
        {
            "mock_arms": False,
            "mock_vision": False,
            "mock_transport_operations": True,
            "mock_aruco_transport_operations": False,
            "mock_install_new_tire": False,
        }
    )
    return config


@pytest.mark.parametrize(
    "left_status,right_status,expected_plan,expected_transport",
    [
        (
            "BAD",
            "GOOD",
            ("left",),
            [
                ("PLACE_REMOVED_TIRE", "left", 3),
                ("PICK_NEW_TIRE", "left", 4),
            ],
        ),
        (
            "GOOD",
            "BAD",
            ("right",),
            [
                ("PLACE_REMOVED_TIRE", "right", 1),
                ("PICK_NEW_TIRE", "right", 2),
            ],
        ),
        (
            "BAD",
            "BAD",
            ("left", "right"),
            [
                ("PLACE_REMOVED_TIRE", "left", 3),
                ("PICK_NEW_TIRE", "left", 4),
                ("PLACE_REMOVED_TIRE", "right", 1),
                ("PICK_NEW_TIRE", "right", 2),
            ],
        ),
    ],
)
def test_full_auto_without_pinky_needs_no_manual_keys(
    left_status,
    right_status,
    expected_plan,
    expected_transport,
    capsys,
):
    arm = FakeArmManager()
    vision = FakeVisionClient(detection_response(left_status, right_status))
    aruco = MockTransportOperationHandler()
    runtime = make_runtime(
        arm=arm,
        vision=vision,
        auto_config=full_auto_without_pinky_config(),
        transport_operations=aruco,
        test_mode=False,
        hardware_enabled=True,
    )
    ready_from_states = []
    original_ready = runtime.controller.mark_transport_ready

    def record_ready_transition():
        ready_from_states.append(runtime.controller.state)
        return original_ready()

    runtime.controller.mark_transport_ready = record_ready_transition
    try:
        assert runtime.controller.arm_system()
        runtime.controller.on_vehicle_detected()
        runtime._on_parking_completed(force=True)
        wait_for_state(runtime, OperationState.COMPLETED)

        assert runtime.controller.replacement_plan == expected_plan
        assert aruco.calls == expected_transport
        assert ready_from_states == [
            OperationState.WAITING_TRANSPORT
        ] * len(expected_plan)
        assert runtime.test_pinky_commands == []
        assert not isinstance(runtime.vision_client, ManualTestVisionClient)

        output = capsys.readouterr().out
        assert output.count("[AUTO] Pinky transport bypassed") == len(
            expected_plan
        )
        assert output.count(
            "[AUTO] transport ready "
            "(mock_transport_operations=true)"
        ) == len(expected_plan)

        if expected_plan == ("left", "right"):
            assert arm.commands.index(("left", "INSTALL_NEW_TIRE")) < (
                arm.commands.index(("right", "REMOVE_BAD_TIRE"))
            )
        assert set(arm.commands[-2:]) == {
            ("left", "HOME"),
            ("right", "HOME"),
        }
    finally:
        runtime.close()


def test_full_auto_without_pinky_flags_do_not_mock_real_equipment():
    parking = deepcopy(PARKING_CONFIG)
    parking["mock_camera"] = False
    yolo = deepcopy(YOLO_CONFIG)
    yolo["mock_vehicle_detector"] = False
    config = full_auto_without_pinky_config()

    apply_manual_test_mode_config(
        parking,
        yolo,
        config,
        test_mode=False,
        hardware_enabled=True,
    )

    operation = config["operation"]
    assert parking["mock_camera"] is False
    assert yolo["mock_vehicle_detector"] is False
    assert operation["mock_vision"] is False
    assert operation["mock_arms"] is False
    assert operation["mock_install_new_tire"] is False
    assert operation["mock_transport_operations"] is True
    assert operation["mock_aruco_transport_operations"] is False


def test_real_arms_allow_install_new_tire_without_install_mock():
    arm = FakeArmManager()
    runtime = make_runtime(
        arm=arm,
        auto_config=full_auto_without_pinky_config(),
        transport_operations=MockTransportOperationHandler(),
        test_mode=False,
        hardware_enabled=True,
    )
    try:
        result = runtime._worker_install_new_tire("left")
        assert result["left"].success
        assert arm.commands == [("left", "INSTALL_NEW_TIRE")]
    finally:
        runtime.close()
