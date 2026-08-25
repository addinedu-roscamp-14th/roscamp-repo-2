import json
import socket
import threading

from pinky_goal_pid.pinky_protocol import PinkyJsonTcpServer, RouteCommandProcessor


class FakeRouteController:
    def __init__(self):
        self.state = "IDLE"
        self.error = None
        self.start_count = 0

    def get_route_status(self):
        return {
            "state": self.state,
            "current_station": (
                "approach_tire_stop_1" if self.state == "RUNNING" else None
            ),
            "station_index": 0,
            "total_stations": 5,
            "error": self.error,
        }

    def start_route(self, route):
        if self.state == "RUNNING":
            return {"success": False, "code": "BUSY", "message": "already running"}
        if route != "after_tire_service":
            return {
                "success": False,
                "code": "UNKNOWN_ROUTE",
                "message": "unknown route",
            }
        self.start_count += 1
        self.state = "RUNNING"
        return {"success": True, "message": "started"}

    def stop_route(self):
        self.state = "STOPPED"
        return {"success": True, "message": "stopped"}

    def reset_route(self):
        if self.state not in ("STOPPED", "ERROR"):
            return {
                "success": False,
                "code": "INVALID_STATE",
                "message": "reset rejected",
            }
        self.state = "IDLE"
        self.error = None
        return {"success": True, "message": "reset"}


def request(command, request_id="req-1", **extra):
    value = {"version": 1, "request_id": request_id, "command": command}
    value.update(extra)
    return value


def test_ping_and_get_status_are_idle():
    processor = RouteCommandProcessor(FakeRouteController())
    assert processor.process_request(request("PING"))["state"] == "IDLE"
    response = processor.process_request(request("GET_STATUS", "req-2"))
    assert response["status"] == "ok"
    assert response["state"] == "IDLE"
    assert response["total_stations"] == 5


def test_start_duplicate_busy_stop_and_reset():
    controller = FakeRouteController()
    processor = RouteCommandProcessor(controller)

    started = processor.process_request(
        request("START_ROUTE", route="after_tire_service")
    )
    assert started["status"] == "ok"
    assert started["state"] == "RUNNING"

    duplicate = processor.process_request(
        request("START_ROUTE", "req-2", route="after_tire_service")
    )
    assert duplicate["status"] == "error"
    assert duplicate["error"]["code"] == "BUSY"
    assert controller.start_count == 1

    stopped = processor.process_request(request("STOP", "req-3"))
    assert stopped["state"] == "STOPPED"
    reset = processor.process_request(request("RESET", "req-4"))
    assert reset["state"] == "IDLE"


def test_invalid_json_unknown_command_and_request_id_deduplication():
    controller = FakeRouteController()
    processor = RouteCommandProcessor(controller)

    invalid = processor.process_line("{not-json")
    assert invalid["error"]["code"] == "INVALID_JSON"

    unknown = processor.process_request(request("FLY", "unknown"))
    assert unknown["error"]["code"] == "UNKNOWN_COMMAND"

    start = request("START_ROUTE", "same-id", route="after_tire_service")
    first = processor.process_request(start)
    second = processor.process_request(dict(start))
    assert first == second
    assert controller.start_count == 1

    conflict = processor.process_request(request("STOP", "same-id"))
    assert conflict["error"]["code"] == "REQUEST_ID_CONFLICT"
    assert controller.state == "RUNNING"


def test_completed_and_error_status_are_exposed():
    controller = FakeRouteController()
    processor = RouteCommandProcessor(controller)
    controller.state = "COMPLETED"
    assert processor.process_request(request("GET_STATUS"))["state"] == "COMPLETED"

    controller.state = "ERROR"
    controller.error = "Nav2 failed"
    response = processor.process_request(request("GET_STATUS", "req-error"))
    assert response["state"] == "ERROR"
    assert response["error"] == "Nav2 failed"


def test_tcp_server_newline_json_round_trip():
    controller = FakeRouteController()
    server = PinkyJsonTcpServer(("127.0.0.1", 0), RouteCommandProcessor(controller))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(server.server_address, timeout=1.0) as sock:
            payload = json.dumps(request("PING")).encode("utf-8") + b"\n"
            sock.sendall(payload)
            line = sock.makefile("rb").readline()
        response = json.loads(line)
        assert response["status"] == "ok"
        assert response["state"] == "IDLE"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
