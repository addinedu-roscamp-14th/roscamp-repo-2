from __future__ import annotations
import json
import threading
from pathlib import Path


class MotionStopped(RuntimeError):
    pass


class MotionPlayer:
    def __init__(self, arm_driver, motion_dir):
        self.arm = arm_driver
        self.motion_dir = Path(motion_dir)
        self.cancel_event = threading.Event()

    def stop(self):
        self.cancel_event.set()
        self.arm.stop()

    def load_motion(self, motion_name):
        if not motion_name.replace("_", "").isalnum():
            raise ValueError("invalid motion name")
        path = self.motion_dir / f"{motion_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"motion file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def play(self, motion_name):
        self.cancel_event.clear()
        for index, step in enumerate(self.load_motion(motion_name)):
            if self.cancel_event.is_set():
                raise MotionStopped(f"motion stopped before step {index + 1}")
            step_type = step.get("type")
            value = step.get("value")
            speed = int(step.get("speed", 50))
            if step_type == "angles":
                self.arm.send_angles(value, speed)
            elif step_type == "coords":
                self.arm.send_coords(value, speed, int(step.get("mode", 0)))
            elif step_type == "gripper":
                self.arm.set_gripper(value, speed)
            elif step_type != "sleep":
                raise ValueError(f"unknown motion step type: {step_type}")
            if self.cancel_event.wait(float(step.get("sleep", 1.0))):
                raise MotionStopped(f"motion stopped at step {index + 1}")
