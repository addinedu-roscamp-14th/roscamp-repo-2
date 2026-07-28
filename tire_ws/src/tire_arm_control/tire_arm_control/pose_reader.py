import rclpy
from rclpy.node import Node

from tire_arm_control.arm_driver import ArmDriver


class PoseReaderNode(Node):
    def __init__(self):
        super().__init__("pose_reader_node")

        self.arm = ArmDriver("/dev/ttyJETCOBOT", 1000000)
        self.timer = self.create_timer(1.0, self.print_pose)

    def print_pose(self):
        angles = self.arm.get_angles()
        coords = self.arm.get_coords()

        self.get_logger().info(f"angles: {angles}")
        self.get_logger().info(f"coords : {coords}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseReaderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
