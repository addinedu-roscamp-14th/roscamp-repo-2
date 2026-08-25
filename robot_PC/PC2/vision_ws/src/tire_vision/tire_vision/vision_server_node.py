from __future__ import annotations

import os
import socketserver
import threading
import time
from pathlib import Path

import yaml

from tire_vision.protocol import ProtocolError, encode_response, make_error, parse_request
from tire_vision.vision_service import VisionService


DEFAULT_CONFIG = {
    "mock_mode": True,
    "tcp_host": "0.0.0.0",
    "tcp_port": 6000,
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 15,
    "model_path": "",
    "imgsz": 640,
    "conf_threshold": 0.5,
    "bad_class_name": "bad_tire",
    "good_class_name": "good_tire",
    "required_stable_count": 3,
    "detect_timeout_sec": 5.0,
    "detect_loop_sleep_sec": 0.01,
    "mock_detection_label": "bad_tire",
    "mock_detection_confidence": 0.91,
}


def load_config(path: str | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        config.update(loaded)
        model_path = config.get("model_path")
        if model_path and not Path(str(model_path)).is_absolute():
            config["model_path"] = str(Path(path).resolve().parent.parent / str(model_path))
    return config


class VisionTcpHandler(socketserver.StreamRequestHandler):
    def handle(self):
        service: VisionService = self.server.service
        for line in self.rfile:
            request = None
            started = time.monotonic()
            try:
                request = parse_request(line)
                client = f"{self.client_address[0]}:{self.client_address[1]}"
                self.server.log(
                    f"client={client} type={request.get('type')} "
                    f"request_id={request.get('request_id')}"
                )
                if request.get("type") == "detect_tires":
                    self.server.log(f"request_id={request.get('request_id')} detect_tires start")
                response = service.handle_request(request)
            except ProtocolError as error:
                response = make_error(error.code, error.message, request)
            except Exception as error:
                response = make_error("INTERNAL_ERROR", str(error), request)
            elapsed = time.monotonic() - started
            if request and request.get("type") == "detect_tires":
                left = response.get("left", {}).get("class_name", "none")
                right = response.get("right", {}).get("class_name", "none")
                self.server.log(
                    f"request_id={request.get('request_id')} left={left} right={right} "
                    f"status={response.get('status')} elapsed={elapsed:.3f}s"
                )
            elif request:
                self.server.log(
                    f"request_id={request.get('request_id')} "
                    f"status={response.get('status')} elapsed={elapsed:.3f}s"
                )
            self.wfile.write(encode_response(response))
            self.wfile.flush()


class ThreadedVisionServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, service: VisionService, logger=None):
        super().__init__(address, VisionTcpHandler)
        self.service = service
        self._logger = logger

    def log(self, message: str) -> None:
        if self._logger:
            self._logger.info(message)
        else:
            print(f"[INFO] {message}", flush=True)


class VisionServerRuntime:
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.service = VisionService(config)
        self.server = ThreadedVisionServer(
            (str(config.get("tcp_host", "0.0.0.0")), int(config.get("tcp_port", 6000))),
            self.service, logger,
        )
        self.thread: threading.Thread | None = None

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.service.close()
        if self.thread:
            self.thread.join(timeout=2.0)


def default_config_path() -> str | None:
    local = Path(__file__).resolve().parents[1] / "config" / "vision.yaml"
    return str(local) if local.exists() else None


def main(args=None):
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("tire_vision_server")
    node.declare_parameter("config_path", default_config_path() or "")
    node.declare_parameter("mock_mode", True)
    node.declare_parameter("tcp_host", "0.0.0.0")
    node.declare_parameter("tcp_port", 6000)
    config_path = node.get_parameter("config_path").get_parameter_value().string_value
    config = load_config(config_path or None)
    config.update({
        "mock_mode": node.get_parameter("mock_mode").value,
        "tcp_host": node.get_parameter("tcp_host").value,
        "tcp_port": node.get_parameter("tcp_port").value,
    })
    runtime = VisionServerRuntime(config, node.get_logger())
    runtime.start()
    node.get_logger().info(
        f"tire_vision TCP server listening on "
        f"{runtime.config['tcp_host']}:{runtime.config['tcp_port']}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
