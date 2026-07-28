import socket
import threading
import time

from fake_vision_server import DEFAULT_DETECTION, FakeVisionServer
from central_control.vision_client import VisionClient


def client_for(server, timeout=1.0):
    return VisionClient(server.host, server.port, 1.0, timeout)


def test_health_ok():
    server = FakeVisionServer()
    try:
        result = client_for(server).health_check()
        assert result.success
        assert result.response["status"] == "ok"
    finally:
        server.close()


def test_detect_tires_ok():
    server = FakeVisionServer()
    try:
        result = client_for(server).request_detection()
        assert result.success
        assert result.response["left"]["class_name"] == "bad_tire"
        assert result.response["right"]["class_name"] == "good_tire"
    finally:
        server.close()


def test_request_id_mismatch():
    response = dict(DEFAULT_DETECTION, request_id="wrong")
    import json
    server = FakeVisionServer(raw_response=(json.dumps(response) + "\n").encode("utf-8"))
    try:
        result = client_for(server).request_detection()
        assert not result.success
        assert "request_id" in result.error
    finally:
        server.close()


def test_status_error():
    response = dict(DEFAULT_DETECTION, status="error")
    server = FakeVisionServer(response=response)
    try:
        result = client_for(server).request_detection()
        assert not result.success
        assert "status" in result.error
    finally:
        server.close()


def test_missing_left_or_right():
    response = dict(DEFAULT_DETECTION)
    response.pop("right")
    server = FakeVisionServer(response=response)
    try:
        result = client_for(server).request_detection()
        assert not result.success
        assert "missing" in result.error
    finally:
        server.close()


def test_stable_false():
    response = dict(DEFAULT_DETECTION)
    response["left"] = dict(response["left"], stable=False)
    server = FakeVisionServer(response=response)
    try:
        result = client_for(server).request_detection()
        assert not result.success
        assert "stable" in result.error
    finally:
        server.close()


def test_invalid_class_name():
    response = dict(DEFAULT_DETECTION)
    response["left"] = dict(response["left"], class_name="unknown")
    server = FakeVisionServer(response=response)
    try:
        result = client_for(server).request_detection()
        assert not result.success
        assert "class_name" in result.error
    finally:
        server.close()


def test_invalid_json():
    server = FakeVisionServer(raw_response=b"not-json\n")
    try:
        result = client_for(server).request_detection()
        assert not result.success
        assert "invalid vision JSON" in result.error
    finally:
        server.close()


def test_timeout():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def accept_and_sleep():
        conn, _ = listener.accept()
        with conn:
            time.sleep(0.3)

    thread = threading.Thread(target=accept_and_sleep, daemon=True)
    thread.start()
    try:
        result = VisionClient(host, port, 1.0, 0.05).request_detection()
        assert not result.success
        assert "timeout" in result.error
    finally:
        listener.close()
        thread.join(timeout=1.0)
