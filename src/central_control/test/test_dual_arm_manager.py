import threading

from central_control.arm_client import ArmClient, ArmCommandResult
from central_control.dual_arm_manager import DualArmManager


class FakeArmClient:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.commands = []

    def send_command(self, command):
        self.commands.append(command)
        from central_control.arm_client import ArmCommandResult
        return ArmCommandResult(self.name, command, not self.fail, response="OK" if not self.fail else "ERR")


def test_send_both():
    left = FakeArmClient("left")
    right = FakeArmClient("right")
    manager = DualArmManager(left, right)
    results = manager.send_both("SET_CAMERA")
    assert results["left"].success
    assert results["right"].success
    assert left.commands == ["SET_CAMERA"]
    assert right.commands == ["SET_CAMERA"]


def test_one_side_failure():
    manager = DualArmManager(FakeArmClient("left"), FakeArmClient("right", fail=True))
    results = manager.send_both("HOME")
    assert results["left"].success
    assert not results["right"].success
    assert not manager.all_success(results)


def test_selected_arm_only():
    left = FakeArmClient("left")
    right = FakeArmClient("right")
    manager = DualArmManager(left, right)
    results = manager.send_selected(left_command="REMOVE_BAD_TIRE")
    assert set(results) == {"left"}
    assert left.commands == ["REMOVE_BAD_TIRE"]
    assert right.commands == []


def test_stop_both():
    left = FakeArmClient("left")
    right = FakeArmClient("right")
    manager = DualArmManager(left, right)
    results = manager.send_both("STOP")
    assert manager.all_success(results)
    assert left.commands == ["STOP"]
    assert right.commands == ["STOP"]

class BarrierArmClient:
    def __init__(self, name, barrier, raises=False):
        self.name = name
        self.barrier = barrier
        self.raises = raises
        self.started = False

    def send_command(self, command):
        self.started = True
        self.barrier.wait(timeout=1.0)
        if self.raises:
            raise RuntimeError(f"{self.name} failed")
        return ArmCommandResult(self.name, command, True, response="OK")


def test_send_both_starts_both_arms_in_parallel():
    barrier = threading.Barrier(2)
    left = BarrierArmClient("left", barrier)
    right = BarrierArmClient("right", barrier)
    manager = DualArmManager(left, right)

    results = manager.send_both("STOP")

    assert left.started and right.started
    assert manager.all_success(results)


def test_one_arm_exception_does_not_hide_other_result():
    barrier = threading.Barrier(2)
    left = BarrierArmClient("left", barrier, raises=True)
    right = BarrierArmClient("right", barrier)
    manager = DualArmManager(left, right)

    results = manager.send_both("HOME")

    assert not results["left"].success
    assert "raised" in results["left"].error
    assert results["right"].success

