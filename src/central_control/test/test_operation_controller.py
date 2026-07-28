from central_control.operation_controller import OperationController, OperationState


LEFT_BAD_RIGHT_GOOD = {
    "left": {"class_name": "bad_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "good_tire", "confidence": 0.88, "stable": True},
}
LEFT_BAD_RIGHT_BAD = {
    "left": {"class_name": "bad_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "bad_tire", "confidence": 0.88, "stable": True},
}
LEFT_GOOD_RIGHT_BAD = {
    "left": {"class_name": "good_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "bad_tire", "confidence": 0.88, "stable": True},
}


def test_arm_and_vehicle_waiting():
    controller = OperationController()
    assert controller.arm_system()
    assert controller.state == OperationState.ARMED
    controller.on_vehicle_detected()
    assert controller.state == OperationState.WAITING_FOR_VEHICLE


def test_parking_before_arm_does_not_start():
    controller = OperationController()
    assert not controller.on_parking_completed()
    assert controller.state == OperationState.IDLE


def test_parking_after_arm_starts_set_camera_phase():
    controller = OperationController()
    controller.arm_system()
    controller.on_vehicle_detected()
    assert controller.on_parking_completed()
    assert controller.state == OperationState.PARKING_STABLE
    assert controller.start_set_camera()
    assert controller.state == OperationState.SETTING_CAMERA


def test_set_camera_success_then_detection():
    controller = OperationController()
    controller.arm_system(); controller.on_vehicle_detected(); controller.on_parking_completed()
    controller.start_set_camera()
    controller.apply_set_camera_result(True)
    assert controller.state == OperationState.DETECTING_TIRES
    assert controller.start_detection()


def test_set_camera_failure_error():
    controller = OperationController()
    controller.arm_system(); controller.on_vehicle_detected(); controller.on_parking_completed()
    controller.start_set_camera()
    controller.apply_set_camera_result(False, "right failed")
    assert controller.state == OperationState.ERROR
    assert "right failed" in controller.last_error


def test_detection_timeout_error():
    controller = OperationController()
    controller.apply_detection_result(None, "vision response timeout")
    assert controller.state == OperationState.ERROR
    assert "timeout" in controller.last_error


def test_request_id_mismatch_error():
    controller = OperationController()
    controller.apply_detection_result(None, "request_id mismatch")
    assert controller.state == OperationState.ERROR
    assert "request_id" in controller.last_error


def test_left_bad_right_good_runs_help():
    controller = OperationController()
    controller.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    assert controller.state == OperationState.CHECKING_LEFT_TIRE
    assert controller.evaluate_left_tire() == "right_help"
    assert controller.state == OperationState.EXECUTING_RIGHT_HELP


def test_left_bad_right_bad_uses_left_only_policy():
    controller = OperationController()
    controller.apply_detection_result(LEFT_BAD_RIGHT_BAD)
    assert controller.evaluate_left_tire() == "right_help"


def test_left_good_completes_without_action():
    controller = OperationController()
    controller.apply_detection_result(LEFT_GOOD_RIGHT_BAD)
    assert controller.evaluate_left_tire() == "completed"
    assert controller.state == OperationState.COMPLETED
    assert controller.last_successful_step == "GOOD_TIRE_NO_ACTION"


def test_left_none_error():
    controller = OperationController()
    controller.apply_detection_result({
        "left": {"class_name": "none", "confidence": 0.0, "stable": True},
        "right": {"class_name": "good_tire", "confidence": 0.8, "stable": True},
    })
    assert controller.state == OperationState.ERROR
    assert "none" in controller.last_error


def test_left_stable_false_error():
    controller = OperationController()
    controller.apply_detection_result({
        "left": {"class_name": "bad_tire", "confidence": 0.9, "stable": False},
        "right": {"class_name": "good_tire", "confidence": 0.8, "stable": True},
    })
    assert controller.state == OperationState.ERROR
    assert "stable" in controller.last_error


def test_right_help_success_then_left_remove():
    controller = OperationController()
    controller.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    controller.evaluate_left_tire()
    controller.apply_right_help_result(True)
    assert controller.state == OperationState.EXECUTING_LEFT_REMOVE
    assert controller.busy


def test_right_help_failure_blocks_left_remove():
    controller = OperationController()
    controller.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    controller.evaluate_left_tire()
    controller.apply_right_help_result(False, "HELP failed")
    assert controller.state == OperationState.ERROR
    assert "HELP failed" in controller.last_error


def test_left_remove_success_completed():
    controller = OperationController()
    controller.state = OperationState.EXECUTING_LEFT_REMOVE
    controller.busy = True
    controller.apply_left_remove_result(True)
    assert controller.state == OperationState.COMPLETED
    assert controller.last_successful_step == "LEFT_REMOVE_BAD_TIRE"


def test_cycle_latch_prevents_duplicate_start():
    controller = OperationController()
    controller.arm_system(); controller.on_vehicle_detected(); controller.on_parking_completed()
    assert controller.cycle_latched
    assert not controller.on_parking_completed()


def test_completed_requires_reset_before_rearm():
    controller = OperationController(state=OperationState.COMPLETED)
    assert not controller.arm_system()
    controller.reset()
    assert controller.arm_system()


def test_stop_priority_sets_stopped():
    controller = OperationController()
    controller.busy = True
    controller.stop()
    assert controller.state == OperationState.STOPPED
    assert not controller.busy


def test_error_does_not_auto_retry():
    controller = OperationController()
    controller.apply_detection_result(None, "bad json")
    assert controller.state == OperationState.ERROR
    assert not controller.on_parking_completed()

def test_emergency_state_transitions_and_manual_reset():
    controller = OperationController(state=OperationState.SETTING_CAMERA)
    controller.busy = True

    assert controller.trigger_emergency("vehicle motion")
    assert controller.state == OperationState.EMERGENCY_STOPPING
    assert controller.emergency_latched
    assert controller.emergency_reason == "vehicle motion"
    assert controller.mark_emergency_stopping()
    assert controller.mark_emergency_homing()
    assert controller.state == OperationState.EMERGENCY_HOMING
    assert controller.apply_emergency_home_result(True, "both HOME")
    assert controller.state == OperationState.EMERGENCY
    assert not controller.arm_system()

    controller.reset()
    assert controller.state == OperationState.IDLE
    assert not controller.emergency_latched
    assert controller.emergency_reason == ""
    assert controller.arm_system()


def test_emergency_home_failure_stays_latched():
    controller = OperationController(state=OperationState.DETECTING_TIRES)
    controller.trigger_emergency("vehicle lost")
    controller.mark_emergency_homing()
    controller.apply_emergency_home_result(False, "right HOME failed")

    assert controller.state == OperationState.EMERGENCY
    assert controller.emergency_latched
    assert "right HOME failed" in controller.last_error


def test_late_normal_worker_result_is_ignored_after_emergency():
    controller = OperationController(state=OperationState.SETTING_CAMERA)
    controller.trigger_emergency("vehicle moved")

    controller.apply_set_camera_result(True)
    controller.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    controller.apply_right_help_result(True)
    controller.apply_left_remove_result(True)

    assert controller.state == OperationState.EMERGENCY_STOPPING
    assert controller.emergency_latched

