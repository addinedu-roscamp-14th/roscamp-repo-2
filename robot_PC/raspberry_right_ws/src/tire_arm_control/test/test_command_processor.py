import json

from tire_arm_control.command_processor import ArmCommandProcessor


class Arm:
    def __init__(self):
        self.coords = [0.0] * 6
        self.stopped = False
    def send_coords(self, coords, speed, mode):
        self.coords = list(coords)
    def get_coords(self):
        return list(self.coords)
    def get_angles(self):
        return [1.0] * 6
    def set_gripper(self, value, speed):
        pass
    def stop(self):
        self.stopped = True


class Player:
    def __init__(self, arm):
        self.arm = arm
        self.played = []
    def play(self, name):
        self.played.append(name)
    def stop(self):
        self.arm.stop()


def processor():
    arm = Arm()
    return ArmCommandProcessor(arm, Player(arm), {
        "position_tolerance_mm": 1,
        "rotation_tolerance_deg": 1,
        "move_timeout_sec": .1,
        "move_poll_interval_sec": .001,
    }), arm


def request(command, **kwargs):
    return json.dumps({"request_id": "test", "command": command, **kwargs})


def test_bad_move_coords_is_rejected():
    subject, arm = processor()
    response = json.loads(subject.process_line(request("MOVE_COORDS", coords=[1, 2])))
    assert response["error"]["code"] == "INVALID_COORDS"
    assert arm.coords == [0.0] * 6


def test_move_success_only_after_target_reached():
    subject, _ = processor()
    response = json.loads(subject.process_line(
        request("MOVE_COORDS", coords=[10, 20, 30, 0, 0, 0])
    ))
    assert response["status"] == "ok"
    assert response["data"]["reached"] is True


def test_stop_is_prioritized_and_latches_emergency():
    subject, arm = processor()
    subject.busy_lock.acquire()
    try:
        response = json.loads(subject.process_line(request("STOP")))
        assert response["status"] == "ok"
        assert arm.stopped
    finally:
        subject.busy_lock.release()
    blocked = json.loads(subject.process_line(request("GRIPPER_OPEN")))
    assert blocked["error"]["code"] == "EMERGENCY_ACTIVE"


def test_duplicate_motion_is_blocked():
    subject, _ = processor()
    subject.busy_lock.acquire()
    try:
        response = json.loads(subject.process_line(
            request("MOVE_COORDS", coords=[0] * 6)
        ))
        assert response["error"]["code"] == "BUSY"
    finally:
        subject.busy_lock.release()


def test_legacy_home_and_remove_bad_tire_compatible():
    subject, _ = processor()
    assert subject.process_line("HOME") == "OK:home"
    assert subject.process_line("REMOVE_BAD_TIRE") == "OK:remove_bad_tire"
