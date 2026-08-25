#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PinkyKeyboardTrigger(Node):
    """Publish Pinky route test commands from an interactive terminal."""

    def __init__(self) -> None:
        super().__init__("pinky_keyboard_trigger")

        if not sys.stdin.isatty():
            raise RuntimeError("pinky_keyboard_trigger requires an interactive terminal")

        self.command_pub = self.create_publisher(
            String,
            "/pinky/route_command",
            10,
        )
        self.status_sub = self.create_subscription(
            String,
            "/pinky/route_status",
            self.status_callback,
            10,
        )

        self.current_status = "UNKNOWN"
        self.start_sent = False
        self.old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
        self.terminal_restored = False
        tty.setcbreak(sys.stdin.fileno())

        self.timer = self.create_timer(0.05, self.keyboard_callback)
        self.get_logger().info(
            "Pinky keyboard trigger started\n"
            "SPACE: start route | s: stop | q: quit"
        )

    def status_callback(self, msg: String) -> None:
        self.current_status = msg.data.strip()
        self.get_logger().info(f"Pinky status: {self.current_status}")

        if self.current_status in ("IDLE", "COMPLETED"):
            self.start_sent = False

    def publish_command(self, command: str) -> None:
        self.command_pub.publish(String(data=command))
        self.get_logger().info(f"Published command: {command}")

    def keyboard_callback(self) -> None:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return

        key = sys.stdin.read(1)
        if key == " ":
            self.start_route()
        elif key.lower() == "s":
            self.stop_route()
        elif key.lower() == "q":
            self.get_logger().info("Keyboard trigger shutting down")
            rclpy.shutdown()

    def start_route(self) -> None:
        if self.current_status == "RUNNING":
            self.get_logger().warning("Pinky is already running; START ignored")
            return
        if self.start_sent:
            self.get_logger().warning("START was already sent; duplicate ignored")
            return
        if self.command_pub.get_subscription_count() < 1:
            self.get_logger().warning(
                "Pinky route command subscriber is not connected"
            )
            return

        self.publish_command("START:after_tire_service")
        self.start_sent = True

    def stop_route(self) -> None:
        self.publish_command("STOP")
        self.start_sent = False

    def restore_terminal(self) -> None:
        if self.terminal_restored:
            return
        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            self.old_terminal_settings,
        )
        self.terminal_restored = True

    def destroy_node(self) -> bool:
        self.restore_terminal()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PinkyKeyboardTrigger()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
