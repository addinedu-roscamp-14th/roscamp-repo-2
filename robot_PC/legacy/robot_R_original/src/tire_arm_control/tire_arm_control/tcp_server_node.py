import os
import socket
import threading

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from tire_arm_control.arm_driver import ArmDriver
from tire_arm_control.motion_player import MotionPlayer


class TcpArmServerNode(Node):
    def __init__(self):
        super().__init__("tcp_arm_server_node")

        package_root = get_package_share_directory("tire_arm_control")
        config_path = os.path.join(package_root, "config", "arm_config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.arm = ArmDriver(
            port=self.config["robot_port"],
            baud=int(self.config["robot_baud"]),
        )

        motion_dir = os.path.join(package_root, self.config["motion_dir"])
        self.player = MotionPlayer(self.arm, motion_dir)

        self.command_pub = self.create_publisher(
            String,
            "/tire_arm_command_received",
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            "/tire_arm_status",
            10,
        )

        self.tcp_host = self.config["tcp_host"]
        self.tcp_port = int(self.config["tcp_port"])

        self.server_thread = threading.Thread(
            target=self.run_tcp_server,
            daemon=True,
        )
        self.server_thread.start()

        self.get_logger().info(
            f"TCP arm server started on {self.tcp_host}:{self.tcp_port}"
        )

    def publish_text(self, publisher, text):
        msg = String()
        msg.data = text
        publisher.publish(msg)

    def command_to_motion(self, command):
        command = command.strip().upper()

        if command.startswith("TRACK_JOINTS"):
            parts = command.split()
            if len(parts) != 3:
                raise ValueError("TRACK_JOINTS requires J1 and J3")

            self.arm.send_j1_j3(
                float(parts[1]),
                float(parts[2]),
                int(self.config.get("tracking_speed", 30)),
            )
            return None

        if command == "STOP":
            self.arm.stop()
            return None

        if command == "OBSERVE_TIRE":
            return "observe_tire"

        if command == "REMOVE_BAD_TIRE":
            return "remove_bad_tire"

        if command == "HOME":
            return "home"

        if command == "HELP":
            return "help"

        if command == "SET_CAMERA":
            return "set_camera"

        if command == "SET_ARUCO":
            return "set_aruco"

        if command == "FOCUS_SERVO":
            self.arm.focus_servos()
            return None

        raise ValueError(f"Unknown command: {command}")

    def handle_command(self, command):
        self.get_logger().info(f"Received command: {command}")
        self.publish_text(self.command_pub, command)

        motion_name = self.command_to_motion(command)

        if motion_name is None:
            self.publish_text(self.status_pub, f"DONE:{command}")
            return f"OK:{command}"

        self.publish_text(self.status_pub, f"PLAYING:{motion_name}")
        self.player.play(motion_name)
        self.publish_text(self.status_pub, f"DONE:{motion_name}")

        return f"OK:{motion_name}"

    def run_tcp_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.tcp_host, self.tcp_port))
            server.listen(5)

            while rclpy.ok():
                conn, addr = server.accept()

                with conn:
                    data = conn.recv(1024).decode("utf-8").strip()

                    if not data:
                        conn.sendall(b"ERR:EMPTY\n")
                        continue

                    try:
                        response = self.handle_command(data)
                    except Exception as e:
                        response = f"ERR:{e}"
                        self.get_logger().error(response)

                    conn.sendall((response + "\n").encode("utf-8"))


def main(args=None):
    rclpy.init(args=args)
    node = TcpArmServerNode()

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
