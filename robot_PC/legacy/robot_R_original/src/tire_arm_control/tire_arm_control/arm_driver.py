import time
import threading

from pymycobot.mycobot280 import MyCobot280


class ArmDriver:
    def __init__(
        self,
        port="/dev/ttyJETCOBOT",
        baud=1000000,
    ):
        self.mc = MyCobot280(port, baud)
        self.mc.thread_lock = True

        # 모션 재생과 실시간 추적 명령이 동시에 들어오는 것을 방지
        self.command_lock = threading.RLock()

        time.sleep(1.0)

    def get_angles(self):
        with self.command_lock:
            angles = self.mc.get_angles()

        if not angles or len(angles) != 6:
            raise RuntimeError(
                f"로봇 관절각을 읽지 못했습니다: {angles}"
            )

        return [float(value) for value in angles]

    def get_coords(self):
        with self.command_lock:
            coords = self.mc.get_coords()

        if not coords or len(coords) != 6:
            raise RuntimeError(
                f"로봇 좌표를 읽지 못했습니다: {coords}"
            )

        return [float(value) for value in coords]

    def send_angles(
        self,
        angles,
        speed=50,
    ):
        if not angles or len(angles) != 6:
            raise ValueError(
                f"Invalid angles. Expected 6 values, got: {angles}"
            )

        target = [
            float(value)
            for value in angles
        ]

        with self.command_lock:
            self.mc.send_angles(
                target,
                int(speed),
            )

    def send_coords(
        self,
        coords,
        speed=30,
        mode=0,
    ):
        if not coords or len(coords) != 6:
            raise ValueError(
                f"Invalid coords. Expected 6 values, got: {coords}"
            )

        target = [
            float(value)
            for value in coords
        ]

        with self.command_lock:
            self.mc.send_coords(
                target,
                int(speed),
                int(mode),
            )

    def send_j1_j3(
        self,
        j1,
        j3,
        speed=30,
    ):
        """
        현재 J2, J4, J5, J6은 유지하고
        J1과 J3만 변경한다.
        """
        j1 = float(j1)
        j3 = float(j3)
        speed = int(speed)

        with self.command_lock:
            current_angles = self.mc.get_angles()

            if (
                not current_angles
                or len(current_angles) != 6
            ):
                raise RuntimeError(
                    "TRACK_JOINTS 실행 전 현재 관절각을 "
                    f"읽지 못했습니다: {current_angles}"
                )

            target_angles = [
                float(value)
                for value in current_angles
            ]

            target_angles[0] = j1
            target_angles[2] = j3

            self.mc.send_angles(
                target_angles,
                speed,
            )

    def stop(self):
        """
        현재 로봇 동작을 정지한다.
        """
        with self.command_lock:
            self.mc.stop()

    def set_gripper(
        self,
        value,
        speed=50,
    ):
        with self.command_lock:
            self.mc.set_gripper_value(
                int(value),
                int(speed),
            )

    def release_servos(self):
        with self.command_lock:
            self.mc.release_all_servos()

    def focus_servos(self):
        with self.command_lock:
            self.mc.focus_all_servos()