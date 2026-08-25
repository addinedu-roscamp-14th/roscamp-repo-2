from __future__ import annotations
import socket
import threading
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from tire_arm_control.arm_driver import ArmDriver
from tire_arm_control.command_processor import ArmCommandProcessor
from tire_arm_control.motion_player import MotionPlayer


class TcpArmServerNode(Node):
    def __init__(self):
        super().__init__("tcp_arm_server_node")
        share = Path(get_package_share_directory("tire_arm_control"))
        self.declare_parameter("config_file", str(share / "config" / "arm_config.yaml"))
        config_path = Path(self.get_parameter("config_file").value)
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.arm = ArmDriver(self.config["robot_port"], self.config["robot_baud"])
        self.player = MotionPlayer(self.arm, share / self.config.get("motion_dir", "motions"))
        self.processor = ArmCommandProcessor(self.arm, self.player, self.config)
        self.command_pub = self.create_publisher(String, "/tire_arm_command_received", 10)
        self.status_pub = self.create_publisher(String, "/tire_arm_status", 10)
        threading.Thread(target=self.run_tcp_server, daemon=True).start()
        self.get_logger().info(
            f"TCP arm server started on {self.config['tcp_host']}:{self.config['tcp_port']}"
        )

    def _publish(self, publisher, text):
        message = String()
        message.data = text
        publisher.publish(message)

    def _serve_connection(self, connection, address):
        with connection:
            connection.settimeout(float(self.config.get("client_timeout_sec", 65.0)))
            buffer = b""
            while b"\n" not in buffer and len(buffer) <= 65536:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                buffer += chunk
            line = buffer.split(b"\n", 1)[0].decode("utf-8", errors="replace")
            self._publish(self.command_pub, line)
            response = self.processor.process_line(line)
            self._publish(self.status_pub, response)
            connection.sendall((response + "\n").encode("utf-8"))

    def run_tcp_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config["tcp_host"], int(self.config["tcp_port"])))
            server.listen(8)
            server.settimeout(1.0)
            while rclpy.ok():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(
                    target=self._serve_connection,
                    args=(connection, address),
                    daemon=True,
                ).start()


def main(args=None):
    rclpy.init(args=args)
    node = TcpArmServerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
