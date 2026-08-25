from __future__ import annotations
import threading
import time
from pymycobot.mycobot280 import MyCobot280


class ArmDriver:
    def __init__(self, port="/dev/ttyJETCOBOT", baud=1000000):
        self.mc = MyCobot280(port, int(baud))
        self.mc.thread_lock = True
        self.command_lock = threading.RLock()
        time.sleep(1.0)

    @staticmethod
    def _six(values, name):
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise ValueError(f"{name} must contain six values")
        return [float(value) for value in values]

    def get_angles(self):
        with self.command_lock:
            result = self.mc.get_angles()
        return self._six(result, "robot angles")

    def get_coords(self):
        with self.command_lock:
            result = self.mc.get_coords()
        return self._six(result, "robot coords")

    def send_angles(self, angles, speed=50):
        target = self._six(angles, "angles")
        with self.command_lock:
            self.mc.send_angles(target, int(speed))

    def send_coords(self, coords, speed=30, mode=0):
        target = self._six(coords, "coords")
        with self.command_lock:
            self.mc.send_coords(target, int(speed), int(mode))

    def set_gripper(self, value, speed=50):
        with self.command_lock:
            self.mc.set_gripper_value(int(value), int(speed))

    def stop(self):
        self.mc.stop()

    def release_servos(self):
        with self.command_lock:
            self.mc.release_all_servos()

    def focus_servos(self):
        with self.command_lock:
            self.mc.focus_all_servos()
