import time

from pymycobot.mycobot280 import MyCobot280


class ArmDriver:
    def __init__(self, port="/dev/ttyJETCOBOT", baud=1000000):
        self.mc = MyCobot280(port, baud)
        self.mc.thread_lock = True
        time.sleep(1.0)

    def get_angles(self):
        return self.mc.get_angles()

    def get_coords(self):
        return self.mc.get_coords()

    def send_angles(self, angles, speed=50):
        if not angles or len(angles) != 6:
            raise ValueError(f"Invalid angles. Expected 6 values, got: {angles}")
        self.mc.send_angles(angles, speed)

    def send_coords(self, coords, speed=30, mode=0):
        if not coords or len(coords) != 6:
            raise ValueError(f"Invalid coords. Expected 6 values, got: {coords}")
        self.mc.send_coords(coords, speed, mode)

    def set_gripper(self, value, speed=50):
        self.mc.set_gripper_value(value, speed)

    def release_servos(self):
        self.mc.release_all_servos()

    def focus_servos(self):
        self.mc.focus_all_servos()
