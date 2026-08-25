"""Pinky newline-delimited JSON protocol, independent from ROS."""
from __future__ import annotations

from collections import OrderedDict
import json
import socketserver
import threading


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 65536
SUPPORTED_COMMANDS = {"PING", "GET_STATUS", "START_ROUTE", "STOP", "RESET"}


def _error_response(request_id, code, message, state=None):
    response = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": {"code": str(code), "message": str(message)},
    }
    if state is not None:
        response["state"] = state
    return response


class RouteCommandProcessor:
    """Validate requests, dispatch route operations, and deduplicate request IDs."""

    def __init__(self, controller, cache_size=256):
        self.controller = controller
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def process_line(self, line):
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError as error:
                return _error_response(None, "INVALID_JSON", f"invalid UTF-8: {error}")
        try:
            request = json.loads(str(line))
        except (json.JSONDecodeError, TypeError) as error:
            return _error_response(None, "INVALID_JSON", str(error))
        return self.process_request(request)

    def process_request(self, request):
        if not isinstance(request, dict):
            return _error_response(None, "INVALID_REQUEST", "request must be a JSON object")

        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            return _error_response(request_id, "INVALID_REQUEST", "request_id is required")
        request_id = request_id.strip()

        if request.get("version") != PROTOCOL_VERSION:
            return _error_response(request_id, "INVALID_REQUEST", "version must be 1")

        command = request.get("command")
        if not isinstance(command, str) or not command.strip():
            return _error_response(request_id, "INVALID_REQUEST", "command is required")
        command = command.strip().upper()

        fingerprint = json.dumps(request, sort_keys=True, separators=(",", ":"))
        with self._lock:
            cached = self._cache.get(request_id)
            if cached is not None:
                old_fingerprint, old_response = cached
                if old_fingerprint != fingerprint:
                    return _error_response(
                        request_id,
                        "REQUEST_ID_CONFLICT",
                        "request_id was already used for a different request",
                        self.controller.get_route_status().get("state"),
                    )
                return dict(old_response)

            response = self._dispatch(request_id, command, request)
            self._cache[request_id] = (fingerprint, dict(response))
            self._cache.move_to_end(request_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return response

    def _dispatch(self, request_id, command, request):
        status = self.controller.get_route_status()
        if command not in SUPPORTED_COMMANDS:
            return _error_response(
                request_id, "UNKNOWN_COMMAND", f"unknown command: {command}", status.get("state")
            )

        if command in ("PING", "GET_STATUS"):
            return self._ok_response(request_id, status)

        if command == "START_ROUTE":
            route = request.get("route", "after_tire_service")
            result = self.controller.start_route(route)
        elif command == "STOP":
            result = self.controller.stop_route()
        else:
            result = self.controller.reset_route()

        status = self.controller.get_route_status()
        if not result.get("success"):
            return _error_response(
                request_id,
                result.get("code", "EXECUTION_FAILED"),
                result.get("message", f"{command} failed"),
                status.get("state"),
            )
        response = self._ok_response(request_id, status)
        response["message"] = result.get("message", command)
        return response

    @staticmethod
    def _ok_response(request_id, status):
        response = {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "ok",
        }
        response.update(status)
        return response


class _JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            line = self.rfile.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_LINE_BYTES:
                response = _error_response(
                    None, "INVALID_REQUEST", f"request exceeds {MAX_LINE_BYTES} bytes"
                )
            else:
                response = self.server.processor.process_line(line)
            payload = json.dumps(
                response, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8") + b"\n"
            self.wfile.write(payload)
            self.wfile.flush()


class PinkyJsonTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, processor):
        self.processor = processor
        super().__init__(server_address, _JsonLineHandler)
