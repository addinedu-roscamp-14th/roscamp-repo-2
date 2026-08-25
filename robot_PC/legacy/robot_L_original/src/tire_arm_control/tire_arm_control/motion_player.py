import json
import os
import time


class MotionPlayer:
    def __init__(self, arm_driver, motion_dir):
        self.arm = arm_driver
        self.motion_dir = motion_dir

    def load_motion(self, motion_name):
        path = os.path.join(self.motion_dir, f"{motion_name}.json")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Motion file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def play(self, motion_name):
        steps = self.load_motion(motion_name)

        for idx, step in enumerate(steps):
            step_type = step.get("type")
            value = step.get("value")
            speed = step.get("speed", 50)
            sleep_time = step.get("sleep", 1.0)

            print(f"[MOTION] {motion_name} step {idx + 1}: {step}")

            if step_type == "angles":
                self.arm.send_angles(value, speed)

            elif step_type == "coords":
                mode = step.get("mode", 0)
                self.arm.send_coords(value, speed, mode)

            elif step_type == "gripper":
                self.arm.set_gripper(value, speed)

            elif step_type == "sleep":
                pass

            else:
                raise ValueError(f"Unknown step type: {step_type}")

            time.sleep(sleep_time)
