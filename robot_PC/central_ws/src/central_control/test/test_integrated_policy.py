import pytest

from central_control.replacement_policy import (
    DetectionPolicy, LeftFirstGate, TRANSPORT_SLOTS, validate_transport_slot,
)


def pair(left, right):
    return {"left": {"status": left}, "right": {"status": right}}


def test_any_recheck_requires_full_pair_redetection():
    assert DetectionPolicy.requires_full_recheck(pair("RECHECK", "GOOD"))
    assert DetectionPolicy.requires_full_recheck(pair("BAD", "RECHECK"))


def test_plan_is_always_left_first():
    assert DetectionPolicy.build_plan(pair("BAD", "BAD")).sides == ("left", "right")


def test_right_cannot_start_before_left_completion():
    gate = LeftFirstGate(DetectionPolicy.build_plan(pair("BAD", "BAD")))
    try:
        gate.start("right")
        assert False, "right start must be blocked"
    except RuntimeError:
        pass

    gate.start("left")
    gate.complete("left")
    gate.start("right")
def test_transport_slot_policy_is_side_specific():
    assert TRANSPORT_SLOTS == {
        "left": {"removed_tire": 3, "new_tire": 4},
        "right": {"removed_tire": 1, "new_tire": 2},
    }
    assert validate_transport_slot("left", "removed_tire", 3) == 3
    assert validate_transport_slot("right", "new_tire", 2) == 2


@pytest.mark.parametrize("side,kind,slot", [("left", "new_tire", 2), ("right", "removed_tire", 3)])
def test_cross_side_transport_slots_are_rejected(side, kind, slot):
    with pytest.raises(ValueError):
        validate_transport_slot(side, kind, slot)


def test_auto_runtime_has_no_set_camera_worker():
    from central_control.auto_operation_node import AutoOperationRuntime
    assert not hasattr(AutoOperationRuntime, "_worker_set_camera")
