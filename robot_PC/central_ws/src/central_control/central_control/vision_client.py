from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import uuid


ALLOWED_CLASSES = {"bad_tire", "good_tire", "none"}
ALLOWED_DETECTION_STATUSES = {"GOOD", "BAD", "RECHECK", "ERROR"}


@dataclass(frozen=True)
class VisionClientResult:
    success: bool
    response: dict | None = None
    error: str | None = None
    request_id: str | None = None


class VisionClient:
    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout_sec: float = 3.0,
        request_timeout_sec: float = 30.0,
        mock_mode: bool = False,
        mock_response: dict | None = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.request_timeout_sec = float(request_timeout_sec)
        self.mock_mode = bool(mock_mode)
        self.mock_response = mock_response

    def health_check(self) -> VisionClientResult:
        request_id = self._new_request_id("health")
        request = {"version": 1, "type": "health", "request_id": request_id}
        result = self._send_request(request)
        if not result.success:
            return result
        response = result.response or {}
        if response.get("request_id") != request_id:
            return VisionClientResult(False, response, "request_id mismatch", request_id)
        if response.get("status") != "ok":
            return VisionClientResult(False, response, f"status is not ok: {response.get('status')}", request_id)
        return result

    def request_detection(self, reset_cycle: bool = True) -> VisionClientResult:
        request_id = self._new_request_id("detect")
        request = {"version": 1, "type": "detect_tires", "request_id": request_id}
        request["reset_cycle"] = bool(reset_cycle)
        result = self._send_request(request)
        if not result.success:
            return result
        return self._validate_detection_response(result.response or {}, request_id)

    def _send_request(self, request: dict) -> VisionClientResult:
        request_id = request["request_id"]
        if self.mock_mode:
            response = self._mock_response(request)
            return VisionClientResult(True, response, request_id=request_id)

        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_sec
            ) as sock:
                sock.settimeout(self.request_timeout_sec)
                sock.sendall(payload)
                line = self._recv_line(sock)
        except socket.timeout:
            return VisionClientResult(False, error="vision response timeout", request_id=request_id)
        except OSError as error:
            return VisionClientResult(False, error=f"vision connection error: {error}", request_id=request_id)

        if not line:
            return VisionClientResult(False, error="empty vision response", request_id=request_id)
        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as error:
            return VisionClientResult(False, error=f"invalid vision JSON: {error}", request_id=request_id)
        if not isinstance(response, dict):
            return VisionClientResult(False, response, "vision response is not an object", request_id)
        return VisionClientResult(True, response, request_id=request_id)

    def _validate_detection_response(self, response: dict, request_id: str) -> VisionClientResult:
        if response.get("request_id") != request_id:
            return VisionClientResult(False, response, "request_id mismatch", request_id)
        if response.get("status") != "ok":
            return VisionClientResult(False, response, f"status is not ok: {response.get('status')}", request_id)
        if response.get("type") != "detection_result":
            return VisionClientResult(False, response, f"unexpected response type: {response.get('type')}", request_id)
        for side in ("left", "right"):
            side_result = response.get(side)
            if not isinstance(side_result, dict):
                return VisionClientResult(False, response, f"missing {side} result", request_id)
            class_name = side_result.get("class_name")
            if class_name not in ALLOWED_CLASSES:
                return VisionClientResult(False, response, f"invalid {side} class_name: {class_name}", request_id)
            status = side_result.get("status")
            if status is None:
                if (
                    side_result.get("stable") is not True
                    and class_name in {"good_tire", "bad_tire"}
                ):
                    return VisionClientResult(
                        False, response, f"{side} detection is not stable", request_id
                    )
                status = self._infer_status(side_result)
                side_result["status"] = status
            if status not in ALLOWED_DETECTION_STATUSES:
                return VisionClientResult(False, response, f"invalid {side} status: {status}", request_id)
            expected_class = {"GOOD": "good_tire", "BAD": "bad_tire"}.get(status, "none")
            if class_name != expected_class:
                return VisionClientResult(
                    False, response,
                    f"{side} status/class_name mismatch: {status}/{class_name}",
                    request_id,
                )
            if status in {"GOOD", "BAD"} and side_result.get("stable") is not True:
                return VisionClientResult(False, response, f"{side} detection is not stable", request_id)
        return VisionClientResult(True, response, request_id=request_id)

    @staticmethod
    def _infer_status(side_result):
        if side_result.get("stable") is True:
            if side_result.get("class_name") == "good_tire":
                return "GOOD"
            if side_result.get("class_name") == "bad_tire":
                return "BAD"
        return "RECHECK"

    @staticmethod
    def _recv_line(sock) -> bytes:
        chunks = []
        while True:
            chunk = sock.recv(1)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk == b"\n":
                break
        return b"".join(chunks).strip()

    @staticmethod
    def _new_request_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4()}"

    def _mock_response(self, request: dict) -> dict:
        if self.mock_response is not None:
            response = dict(self.mock_response)
            response["request_id"] = request["request_id"]
            return response
        if request["type"] == "health":
            return {
                "version": 1,
                "type": "health",
                "request_id": request["request_id"],
                "status": "ok",
                "health": {"ok": True, "mock": True},
            }
        return {
            "version": 1,
            "type": "detection_result",
            "request_id": request["request_id"],
            "status": "ok",
            "left": {
                "status": "BAD",
                "class_name": "bad_tire",
                "confidence": 0.91,
                "stable": True,
                "votes": 3,
                "frames": 3,
                "detections": [],
            },
            "right": {
                "status": "GOOD",
                "class_name": "good_tire",
                "confidence": 0.88,
                "stable": True,
                "votes": 3,
                "frames": 3,
                "detections": [],
            },
        }
