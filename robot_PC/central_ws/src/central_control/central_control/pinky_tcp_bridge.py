"""Bridge central Pinky ROS topics to the Pinky newline-JSON TCP server."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
import yaml

from central_control.pinky_client import PinkyTcpClient, parse_route_command


VALID_ROUTE_STATES = {"IDLE", "RUNNING", "STOPPED", "COMPLETED", "ERROR"}


class PinkyTcpBridge(Node):
    """Keep the existing Pinky ROS topics while hiding TCP details."""

    def __init__(self):
        super().__init__("pinky_tcp_bridge")
        default_network = str(
            Path(get_package_share_directory("central_control"))
            / "config"
            / "network.yaml"
        )
        self.declare_parameter("network_config", default_network)
        self.declare_parameter("status_poll_sec", 0.5)

        network_path = Path(str(self.get_parameter("network_config").value))
        config = yaml.safe_load(network_path.read_text(encoding="utf-8")) or {}
        pinky = config.get("pinky")
        if not isinstance(pinky, dict):
            raise RuntimeError(f"network config has no pinky section: {network_path}")

        self.client = PinkyTcpClient(**pinky)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._command_future = None
        self._command_generation = 0
        self._stop_future = None
        self._poll_future = None
        self._last_published = None
        self._last_status_publish_at = 0.0
        self._last_connection_warning_at = 0.0
        self._lock = threading.Lock()

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = QoSReliabilityPolicy.RELIABLE
        status_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(
            String, "/pinky/route_status", status_qos
        )
        self.command_sub = self.create_subscription(
            String, "/pinky/route_command", self._on_command, 10
        )
        self.timer = self.create_timer(
            float(self.get_parameter("status_poll_sec").value),
            self._tick,
        )
        self.get_logger().info(
            f"Pinky TCP bridge configured for {self.client.host}:{self.client.port}"
        )

    def _on_command(self, message):
        try:
            operation, route = parse_route_command(message.data)
        except ValueError as error:
            self._publish_status(f"ERROR:INVALID_COMMAND:{error}")
            return

        with self._lock:
            if operation == "STOP":
                self._command_generation += 1
                if self._stop_future is None:
                    self._stop_future = self._executor.submit(self.client.stop)
                return
            if self._command_future is not None:
                self.get_logger().warning(
                    f"Pinky command ignored while another is pending: {message.data}"
                )
                return
            generation = self._command_generation
            if operation == "START_ROUTE":
                future = self._executor.submit(self.client.start_route, route)
            else:
                future = self._executor.submit(self.client.reset)
            self._command_future = (future, generation)

    def _tick(self):
        with self._lock:
            self._harvest_stop()
            self._harvest_command()
            self._harvest_poll()
            if self._poll_future is None:
                self._poll_future = self._executor.submit(self.client.get_status)

    def _harvest_stop(self):
        if self._stop_future is None or not self._stop_future.done():
            return
        future = self._stop_future
        self._stop_future = None
        self._publish_future(future)

    def _harvest_command(self):
        if self._command_future is None:
            return
        future, generation = self._command_future
        if not future.done():
            return
        self._command_future = None
        if generation == self._command_generation:
            self._publish_future(future)

    def _harvest_poll(self):
        if self._poll_future is None or not self._poll_future.done():
            return
        future = self._poll_future
        self._poll_future = None
        self._publish_future(future, polling=True)

    def _publish_future(self, future, polling=False):
        try:
            result = future.result()
        except Exception as error:
            self._publish_status(f"ERROR:TCP_INTERNAL_ERROR:{error}")
            self.get_logger().error(f"Pinky TCP worker failed: {error}")
            return
        self._publish_result(result, polling=polling)

    def _publish_result(self, result, polling=False):
        if not result.success:
            code = result.error_code or "TCP_ERROR"
            if code == "CONNECTION_FAILED":
                code = "TCP_CONNECTION_FAILED"
            elif code == "TIMEOUT":
                code = "TCP_TIMEOUT"
            status = f"ERROR:{code}:{result.error or 'unknown TCP error'}"
            self._publish_status(status)
            if polling:
                now = time.monotonic()
                if now - self._last_connection_warning_at >= 10.0:
                    self.get_logger().warning(status)
                    self._last_connection_warning_at = now
            return

        response = result.response or {}
        state = str(response.get("state", "ERROR")).upper()
        if state not in VALID_ROUTE_STATES:
            self._publish_status(f"ERROR:INVALID_STATE:{state}")
            return
        if state == "ERROR":
            detail = response.get("error") or response.get("message") or "unknown Pinky error"
            self._publish_status(f"ERROR:{detail}")
            return
        self._publish_status(state)

    def _publish_status(self, status):
        status = str(status)
        now = time.monotonic()
        changed = status != self._last_published
        if not changed and now - self._last_status_publish_at < 2.0:
            return
        self._last_published = status
        self._last_status_publish_at = now
        self.status_pub.publish(String(data=status))
        if changed:
            self.get_logger().info(f"Pinky route status: {status}")

    def destroy_node(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PinkyTcpBridge()
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
