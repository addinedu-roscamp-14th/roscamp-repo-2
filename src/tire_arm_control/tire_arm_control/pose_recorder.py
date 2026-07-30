import json
import os
import rclpy
from rclpy.node import Node

from tire_arm_control.arm_driver import ArmDriver


class PoseRecorderNode(Node):
    def __init__(self):
        super().__init__("pose_recorder_node")

        self.arm = ArmDriver("/dev/ttyJETCOBOT", 1000000)

        package_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        self.motion_dir = os.path.join(package_root, "motions")
        os.makedirs(self.motion_dir, exist_ok=True)

        self.get_logger().info("Pose recorder started.")
        self.get_logger().info("Commands:")
        self.get_logger().info("  release  : servo off")
        self.get_logger().info("  focus    : servo on")
        self.get_logger().info("  angles   : print current angles")
        self.get_logger().info("  coords   : print current coords")
        self.get_logger().info("  save     : save current angles to motion json")
        self.get_logger().info("  quit     : exit")

    def load_motion(self, motion_name):
        path = os.path.join(self.motion_dir, f"{motion_name}.json")

        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_motion(self, motion_name, steps):
        path = os.path.join(self.motion_dir, f"{motion_name}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2, ensure_ascii=False)

        self.get_logger().info(f"Saved: {path}")

    def save_current_angles(self):
        motion_name = input("motion name(home/observe_tire/remove_bad_tire): ").strip()

        if not motion_name:
            self.get_logger().warn("motion name is empty.")
            return

        angles = self.arm.get_angles()

        step = {
            "type": "angles",
            "value": angles,
            "speed": 50,
            "sleep": 2.0
        }

        steps = self.load_motion(motion_name)
        steps.append(step)
        self.save_motion(motion_name, steps)

        self.get_logger().info(f"Appended angles to {motion_name}: {angles}")

    def run_cli(self):
        while rclpy.ok():
            cmd = input("\npose_recorder> ").strip().lower()

            if cmd == "release":
                self.arm.release_servos()
                self.get_logger().info("Servos released.")

            elif cmd == "focus":
                self.arm.focus_servos()
                self.get_logger().info("Servos focused.")

            elif cmd == "angles":
                angles = self.arm.get_angles()
                self.get_logger().info(f"angles: {angles}")

            elif cmd == "coords":
                coords = self.arm.get_coords()
                self.get_logger().info(f"coords: {coords}")

            elif cmd == "save":
                self.save_current_angles()

            elif cmd == "quit":
                break

            else:
                self.get_logger().warn("Unknown command.")


def main(args=None):
    rclpy.init(args=args)

    node = PoseRecorderNode()

    try:
        node.run_cli()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
