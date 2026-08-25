"""TCP client for the Pinky route server."""
from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import uuid


@dataclass(frozen=True)
class PinkyClientResult:
    success: bool
    response: dict | None = None
    error: str | None = None
    request_id: str | None = None
    error_code: str | None = None


def parse_route_command(command):
    command = str(command).strip()
    if command == "STOP":
        return "STOP", None
    if command == "RESET":
        return "RESET", None
    if command.startswith("START:"):
        route = command.partition(":")[2].strip()
        if route:
            return "START_ROUTE", route
    raise ValueError(f"unsupported Pinky route command: {command}")


class PinkyTcpClient:
    def __init__(
        self,
        host,
        port,
        connect_timeout_sec=3.0,
        command_timeout_sec=3.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)

    def ping(self):
        return self._request("PING")

    def get_status(self):
        return self._request("GET_STATUS")

    def start_route(self, route="after_tire_service"):
        return self._request("START_ROUTE", route=str(route))

    def stop(self):
        return self._request("STOP")

    def reset(self):
        return self._request("RESET")

    def _request(self, command, **fields):
        request_id = f"pinky-{uuid.uuid4()}"
        request = {
            "version": 1,
            "request_id": request_id,
            "command": str(command).upper(),
        }
        request.update(fields)
        payload = (
            json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")

        try:
            with socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_sec,
            ) as sock:
                sock.settimeout(self.command_timeout_sec)
                sock.sendall(payload)
                line = self._recv_line(sock)
        except socket.timeout:
            return PinkyClientResult(
                False,
                error="Pinky response timeout",
                request_id=request_id,
                error_code="TIMEOUT",
            )
        except OSError as error:
            return PinkyClientResult(
                False,
                error=f"Pinky connection error: {error}",
                request_id=request_id,
                error_code="CONNECTION_FAILED",
            )

        if not line:
            return PinkyClientResult(
                False,
                error="empty Pinky response",
                request_id=request_id,
                error_code="EMPTY_RESPONSE",
            )
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return PinkyClientResult(
                False,
                error=f"invalid Pinky JSON: {error}",
                request_id=request_id,
                error_code="INVALID_JSON",
            )
        if not isinstance(response, dict):
            return PinkyClientResult(
                False,
                response=response,
                error="Pinky response is not an object",
                request_id=request_id,
                error_code="INVALID_RESPONSE",
            )
        if response.get("request_id") != request_id:
            return PinkyClientResult(
                False,
                response=response,
                error="Pinky request_id mismatch",
                request_id=request_id,
                error_code="REQUEST_ID_MISMATCH",
            )
        if response.get("status") != "ok":
            error = response.get("error") or {}
            return PinkyClientResult(
                False,
                response=response,
                error=str(error.get("message", "Pinky command failed")),
                request_id=request_id,
                error_code=str(error.get("code", "REMOTE_ERROR")),
            )
        return PinkyClientResult(True, response=response, request_id=request_id)

    @staticmethod
    def _recv_line(sock, maximum=65536):
        data = bytearray()
        while len(data) <= maximum:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            newline = data.find(b"\n")
            if newline >= 0:
                return bytes(data[:newline])
        if len(data) > maximum:
            raise OSError(f"Pinky response exceeds {maximum} bytes")
        return bytes(data)
