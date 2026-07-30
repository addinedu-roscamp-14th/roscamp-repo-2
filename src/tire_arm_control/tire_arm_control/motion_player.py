import json
import os
import threading
import time


class MotionStoppedError(RuntimeError):
    """STOP 요청으로 모션 실행이 중단됐음을 나타낸다."""


class MotionPlayer:
    def __init__(self, arm_driver, motion_dir):
        self.arm = arm_driver
        self.motion_dir = motion_dir

    def load_motion(self, motion_name):
        path = os.path.join(
            self.motion_dir,
            f"{motion_name}.json",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Motion file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _raise_if_stop_requested(
        stop_event: threading.Event | None,
        motion_name: str,
        step_index: int,
    ):
        if stop_event is not None and stop_event.is_set():
            raise MotionStoppedError(
                f"{motion_name} stopped before step "
                f"{step_index + 1}"
            )

    @staticmethod
    def _interruptible_sleep(
        seconds: float,
        stop_event: threading.Event | None,
        check_interval: float = 0.02,
    ):
        """
        일반 time.sleep() 대신 STOP 요청을 확인하면서 대기한다.

        긴 sleep 중에도 최대 check_interval 정도의 지연으로
        STOP 요청을 감지할 수 있다.
        """
        seconds = max(0.0, float(seconds))

        if seconds == 0.0:
            return

        deadline = time.monotonic() + seconds

        while True:
            if stop_event is not None and stop_event.is_set():
                raise MotionStoppedError(
                    "Motion stopped during sleep"
                )

            remaining = deadline - time.monotonic()

            if remaining <= 0.0:
                return

            time.sleep(
                min(check_interval, remaining)
            )

    def play(
        self,
        motion_name,
        stop_event: threading.Event | None = None,
    ):
        steps = self.load_motion(motion_name)

        for idx, step in enumerate(steps):
            self._raise_if_stop_requested(
                stop_event,
                motion_name,
                idx,
            )

            step_type = step.get("type")
            value = step.get("value")
            speed = step.get("speed", 50)
            sleep_time = step.get("sleep", 1.0)

            print(
                f"[MOTION] {motion_name} "
                f"step {idx + 1}/{len(steps)}: {step}"
            )

            if step_type == "angles":
                self.arm.send_angles(
                    value,
                    speed,
                )

            elif step_type == "coords":
                mode = step.get("mode", 0)

                self.arm.send_coords(
                    value,
                    speed,
                    mode,
                )

            elif step_type == "gripper":
                self.arm.set_gripper(
                    value,
                    speed,
                )

            elif step_type == "sleep":
                pass

            else:
                raise ValueError(
                    f"Unknown step type: {step_type}"
                )

            # 명령을 전송한 직후 STOP이 발생했는지 다시 확인한다.
            self._raise_if_stop_requested(
                stop_event,
                motion_name,
                idx,
            )

            self._interruptible_sleep(
                sleep_time,
                stop_event,
            )

        return True