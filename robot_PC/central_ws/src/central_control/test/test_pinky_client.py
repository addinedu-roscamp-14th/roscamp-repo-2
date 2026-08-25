import json
import socketserver
import threading
import time

from central_control.pinky_client import PinkyTcpClient, parse_route_command


class _Handler(socketserver.StreamRequestHandler):
    response_delay = 0.0

    def handle(self):
        request = json.loads(self.rfile.readline())
        if self.response_delay:
            time.sleep(self.response_delay)
        response = {
            "version": 1,
            "request_id": request["request_id"],
            "status": "ok",
            "state": "RUNNING" if request["command"] == "START_ROUTE" else "IDLE",
            "current_station": None,
            "station_index": 0,
            "total_stations": 5,
            "error": None,
        }
        self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _run_server(handler=_Handler):
    server = _Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_route_command_parser():
    assert parse_route_command("START:after_tire_service") == (
        "START_ROUTE",
        "after_tire_service",
    )
    assert parse_route_command("STOP") == ("STOP", None)
    assert parse_route_command("RESET") == ("RESET", None)


def test_client_ping_status_start_stop_reset():
    server, thread = _run_server()
    try:
        client = PinkyTcpClient(
            server.server_address[0],
            server.server_address[1],
            connect_timeout_sec=0.2,
            command_timeout_sec=0.2,
        )
        assert client.ping().success
        assert client.get_status().response["state"] == "IDLE"
        assert client.start_route().response["state"] == "RUNNING"
        assert client.stop().success
        assert client.reset().success
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_client_connection_failure():
    probe = _Server(("127.0.0.1", 0), _Handler)
    host, port = probe.server_address
    probe.server_close()
    result = PinkyTcpClient(
        host, port, connect_timeout_sec=0.05, command_timeout_sec=0.05
    ).ping()
    assert not result.success
    assert result.error_code == "CONNECTION_FAILED"


def test_client_timeout_is_not_success():
    class SlowHandler(_Handler):
        response_delay = 0.2

    server, thread = _run_server(SlowHandler)
    try:
        result = PinkyTcpClient(
            server.server_address[0],
            server.server_address[1],
            connect_timeout_sec=0.05,
            command_timeout_sec=0.05,
        ).get_status()
        assert not result.success
        assert result.error_code == "TIMEOUT"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_client_rejects_invalid_json_response():
    class InvalidJsonHandler(socketserver.StreamRequestHandler):
        def handle(self):
            self.rfile.readline()
            self.wfile.write(b"{invalid-json\n")

    server, thread = _run_server(InvalidJsonHandler)
    try:
        result = PinkyTcpClient(
            server.server_address[0],
            server.server_address[1],
            connect_timeout_sec=0.05,
            command_timeout_sec=0.05,
        ).get_status()
        assert not result.success
        assert result.error_code == "INVALID_JSON"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
