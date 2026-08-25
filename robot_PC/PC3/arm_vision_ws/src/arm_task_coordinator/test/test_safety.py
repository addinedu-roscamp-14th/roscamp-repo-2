import numpy as np
import pytest

from arm_aruco_perception.handeye_transform import CalibrationUnavailable, HandEyeTransform
from arm_task_coordinator.replacement_coordinator import ReplacementCoordinator
from arm_task_coordinator.safe_pose_provider import PerceptionNotStable, SafePoseProvider


class Client:
    def __init__(self):
        self.calls = []
    def move_coords(self, coords, **kwargs):
        self.calls.append(("MOVE_COORDS", coords))
    def command(self, command):
        self.calls.append((command, None))
    def legacy_command(self, command):
        self.calls.append((command, None))
    def stop(self):
        self.calls.append(("STOP", None))
    def get_coords(self):
        return [0] * 6


class Provider:
    def targets_for_slot(self, slot):
        return {"approach": [1] * 6, "grasp": [2] * 6, "place": [3] * 6}


class Unstable:
    def stable_pose(self):
        return None


def test_unstable_aruco_blocks_coordinate_send():
    client = Client()
    provider = SafePoseProvider(Unstable(), object(), object(), client)
    with pytest.raises(PerceptionNotStable):
        provider.targets_for_slot(1)
    assert client.calls == []


def test_missing_handeye_file_blocks_start(tmp_path):
    with pytest.raises(CalibrationUnavailable):
        HandEyeTransform(tmp_path / "missing.npz")


def test_right_blocked_until_left_complete():
    left, right = Client(), Client()
    coordinator = ReplacementCoordinator(left, right, {"left": Provider(), "right": Provider()})
    coordinator.left_required = True
    with pytest.raises(RuntimeError):
        coordinator.run_side("right", 1)
    assert right.calls == []
    coordinator.run_side("left", 1)
    coordinator.run_side("right", 1)
    assert left.calls[-1][0] == "HOME"
    assert right.calls[-1][0] == "HOME"


def test_emergency_blocks_new_cycle():
    left, right = Client(), Client()
    coordinator = ReplacementCoordinator(left, right, {"left": Provider(), "right": Provider()})
    coordinator.emergency.trigger("test")
    with pytest.raises(Exception, match="emergency active"):
        coordinator.run_side("left", 1)
