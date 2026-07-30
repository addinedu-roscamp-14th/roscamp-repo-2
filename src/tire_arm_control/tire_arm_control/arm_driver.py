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

        # 로봇 시리얼 통신이 여러 스레드에서 동시에 호출되는 것을 방지한다.
        # 모션 전체가 아니라 각 pymycobot API 호출 단위로만 잠근다.
        self.command_lock = threading.RLock()

        time.sleep(1.0)

    def get_angles(self):
        with self.command_lock:
            angles = self.mc.get_angles()

        if not angles or len(angles) != 6:
            raise RuntimeError(
                f"로봇 관절각을 읽지 못했습니다: {angles}"
            )

        return [
            float(value)
            for value in angles
        ]

    def get_coords(self):
        with self.command_lock:
            coords = self.mc.get_coords()

        if not coords or len(coords) != 6:
            raise RuntimeError(
                f"로봇 좌표를 읽지 못했습니다: {coords}"
            )

        return [
            float(value)
            for value in coords
        ]

    def send_angles(
        self,
        angles,
        speed=50,
    ):
        if not angles or len(angles) != 6:
            raise ValueError(
                "Invalid angles. "
                f"Expected 6 values, got: {angles}"
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
                "Invalid coords. "
                f"Expected 6 values, got: {coords}"
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
        현재 로봇 이동을 정지한다.

        STOP 명령을 로봇 컨트롤러에 전달하며,
        실제 정지 여부는 wait_until_stopped()에서 확인한다.
        """
        with self.command_lock:
            return self.mc.stop()

    def is_moving(self):
        """
        로봇 컨트롤러가 이동 중인지 확인한다.

        반환값:
        - True: 이동 중
        - False: 정지
        - None: API 미지원 또는 판정 실패
        """
        method = getattr(
            self.mc,
            "is_moving",
            None,
        )

        if not callable(method):
            return None

        try:
            with self.command_lock:
                result = method()
        except Exception:
            return None

        if result in (1, True):
            return True

        if result in (0, False):
            return False

        return None

    def wait_until_stopped(
        self,
        timeout=3.0,
        poll_interval=0.1,
        stable_samples=3,
        angle_tolerance=0.5,
    ):
        """
        로봇이 실제로 정지했는지 확인한다.

        1. 우선 mc.is_moving() 응답을 사용한다.
        2. is_moving()을 사용할 수 없거나 응답이 불확실하면
           연속 관절각 변화량을 이용해 정지를 판정한다.

        stable_samples 횟수만큼 연속으로 정지 상태가 확인되어야
        True를 반환한다.
        """
        timeout = max(
            0.1,
            float(timeout),
        )

        poll_interval = max(
            0.02,
            float(poll_interval),
        )

        stable_samples = max(
            1,
            int(stable_samples),
        )

        angle_tolerance = max(
            0.0,
            float(angle_tolerance),
        )

        deadline = time.monotonic() + timeout
        stable_count = 0
        previous_angles = None

        while time.monotonic() < deadline:
            moving = self.is_moving()

            if moving is False:
                stable_count += 1

                if stable_count >= stable_samples:
                    return True

                time.sleep(poll_interval)
                continue

            if moving is True:
                stable_count = 0
                previous_angles = None

                time.sleep(poll_interval)
                continue

            # is_moving()을 사용할 수 없는 경우
            # 연속 관절각 변화량으로 정지를 확인한다.
            try:
                current_angles = self.get_angles()
            except Exception:
                stable_count = 0
                previous_angles = None

                time.sleep(poll_interval)
                continue

            if previous_angles is None:
                previous_angles = current_angles
                stable_count = 0

                time.sleep(poll_interval)
                continue

            max_delta = max(
                abs(current - previous)
                for current, previous in zip(
                    current_angles,
                    previous_angles,
                )
            )

            previous_angles = current_angles

            if max_delta <= angle_tolerance:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= stable_samples:
                return True

            time.sleep(poll_interval)

        return False

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
