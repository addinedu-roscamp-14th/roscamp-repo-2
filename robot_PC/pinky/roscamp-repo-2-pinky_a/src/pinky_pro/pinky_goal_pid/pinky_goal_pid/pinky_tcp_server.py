"""ROS-integrated Pinky TCP route server."""
from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading

import rclpy

from pinky_goal_pid.nav2_waypt import WaypointManager
from pinky_goal_pid.pinky_protocol import PinkyJsonTcpServer, RouteCommandProcessor


@dataclass
class _QueuedCall:
    operation: str
    argument: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class _ExecutorQueueController:
    """Marshal TCP-thread calls onto the ROS executor timer."""

    def __init__(self, command_queue, timeout_sec):
        self.command_queue = command_queue
        self.timeout_sec = float(timeout_sec)

    def _call(self, operation, argument=None):
        call = _QueuedCall(operation=operation, argument=argument)
        self.command_queue.put(call)
        if not call.done.wait(self.timeout_sec):
            return {
                "success": False,
                "code": "EXECUTOR_TIMEOUT",
                "message": f"ROS executor did not process {operation} in time",
            }
        return call.result

    def get_route_status(self):
        result = self._call("GET_STATUS")
        if result.get("success") is False:
            return {
                "state": "ERROR",
                "current_station": None,
                "station_index": 0,
                "total_stations": 0,
                "error": result.get("message"),
            }
        return result

    def start_route(self, route):
        return self._call("START_ROUTE", str(route))

    def stop_route(self):
        return self._call("STOP")

    def reset_route(self):
        return self._call("RESET")


class PinkyTcpRouteServer(WaypointManager):
    def __init__(self):
        super().__init__(node_name="pinky_tcp_route_server")
        if self.loop_mode:
            self.get_logger().warning(
                "loop_mode=true is ignored by TCP operation; forcing one route cycle"
            )
        self.loop_mode = False
        self.declare_parameter("tcp_host", "0.0.0.0")
        self.declare_parameter("tcp_port", 7000)
        self.declare_parameter("tcp_command_timeout_sec", 5.0)
        self.declare_parameter("request_cache_size", 256)

        self._tcp_queue = queue.Queue()
        bridge = _ExecutorQueueController(
            self._tcp_queue,
            self.get_parameter("tcp_command_timeout_sec").value,
        )
        processor = RouteCommandProcessor(
            bridge,
            cache_size=self.get_parameter("request_cache_size").value,
        )
        address = (
            str(self.get_parameter("tcp_host").value),
            int(self.get_parameter("tcp_port").value),
        )
        self._tcp_server = PinkyJsonTcpServer(address, processor)
        self._tcp_thread = threading.Thread(
            target=self._tcp_server.serve_forever,
            name="pinky-tcp-server",
            daemon=True,
        )
        self._tcp_thread.start()
        self._tcp_timer = self.create_timer(0.02, self._process_tcp_commands)
        self.get_logger().info(
            f"Pinky TCP server listening on {address[0]}:{address[1]} (state=IDLE)"
        )

    def _process_tcp_commands(self):
        for _ in range(32):
            try:
                call = self._tcp_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if call.operation == "GET_STATUS":
                    call.result = self.get_route_status()
                elif call.operation == "START_ROUTE":
                    call.result = self.start_route(call.argument)
                elif call.operation == "STOP":
                    call.result = self.stop_route()
                elif call.operation == "RESET":
                    call.result = self.reset_route()
                else:
                    call.result = {
                        "success": False,
                        "code": "UNKNOWN_COMMAND",
                        "message": f"unknown executor operation: {call.operation}",
                    }
            except Exception as error:
                call.result = {
                    "success": False,
                    "code": "EXECUTION_FAILED",
                    "message": str(error),
                }
            finally:
                call.done.set()

    def destroy_node(self):
        self.stop_route()
        self._tcp_server.shutdown()
        self._tcp_server.server_close()
        self._tcp_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PinkyTcpRouteServer()
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
