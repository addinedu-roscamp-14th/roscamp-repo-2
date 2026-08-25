from central_control.work_state_machine import WorkState, WorkStateMachine


LEFT_BAD_RIGHT_GOOD = {
    "left": {"class_name": "bad_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "good_tire", "confidence": 0.88, "stable": True},
}
LEFT_GOOD_RIGHT_BAD = {
    "left": {"class_name": "good_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "bad_tire", "confidence": 0.88, "stable": True},
}
BOTH_BAD = {
    "left": {"class_name": "bad_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "bad_tire", "confidence": 0.88, "stable": True},
}
BOTH_GOOD = {
    "left": {"class_name": "good_tire", "confidence": 0.91, "stable": True},
    "right": {"class_name": "good_tire", "confidence": 0.88, "stable": True},
}


def parked_machine():
    machine = WorkStateMachine()
    machine.parking_completed()
    return machine


def test_parked_to_wait_detection_confirm():
    machine = WorkStateMachine()
    machine.parking_completed()
    assert machine.state == WorkState.WAIT_DETECTION_CONFIRM


def test_set_camera_stage_is_skipped():
    machine = parked_machine()
    assert not machine.start_set_camera()
    assert machine.state == WorkState.WAIT_DETECTION_CONFIRM


def test_detection_success_to_wait_action_confirm():
    machine = parked_machine()
    assert machine.start_detection()
    machine.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    assert machine.state == WorkState.WAIT_ACTION_CONFIRM


def test_left_bad_right_good_targets_left():
    machine = parked_machine()
    machine.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    assert machine.target_arms == {"left"}


def test_left_good_right_bad_targets_right():
    machine = parked_machine()
    machine.apply_detection_result(LEFT_GOOD_RIGHT_BAD)
    assert machine.target_arms == {"right"}


def test_both_bad_targets_both():
    machine = parked_machine()
    machine.apply_detection_result(BOTH_BAD)
    assert machine.target_arms == {"left", "right"}


def test_both_good_skips_remove_and_waits_home():
    machine = parked_machine()
    machine.apply_detection_result(BOTH_GOOD)
    assert machine.target_arms == set()
    assert machine.state == WorkState.WAIT_HOME_CONFIRM


def test_none_goes_error():
    response = {
        "left": {"class_name": "none", "confidence": 0.0, "stable": True},
        "right": {"class_name": "good_tire", "confidence": 0.88, "stable": True},
    }
    machine = parked_machine()
    machine.apply_detection_result(response)
    assert machine.state == WorkState.ERROR
    assert "none" in machine.last_error


def test_stable_false_goes_error():
    response = {
        "left": {"class_name": "bad_tire", "confidence": 0.91, "stable": False},
        "right": {"class_name": "good_tire", "confidence": 0.88, "stable": True},
    }
    machine = parked_machine()
    machine.apply_detection_result(response)
    assert machine.state == WorkState.ERROR
    assert "stable" in machine.last_error


def test_detection_failure_goes_error():
    machine = parked_machine()
    machine.apply_detection_result(None, "vision failed")
    assert machine.state == WorkState.ERROR
    assert machine.last_error == "vision failed"


def test_reset_returns_initial_state():
    machine = parked_machine()
    machine.apply_detection_result(LEFT_BAD_RIGHT_GOOD)
    machine.reset()
    assert machine.state == WorkState.WAITING_FOR_VEHICLE
    assert machine.target_arms == set()
    assert machine.left_detection is None
    assert machine.last_error == ""


def test_stop_priority():
    machine = parked_machine()
    assert machine.start_detection()
    machine.stop()
    assert machine.state == WorkState.STOPPED
    assert not machine.busy
