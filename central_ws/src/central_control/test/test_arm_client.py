from fake_arm_server import FakeArmServer
from central_control.arm_client import ArmClient


def test_ok_response():
    server = FakeArmServer()
    try:
        result = ArmClient("left", server.host, server.port, 1.0, 1.0, mock_mode=False).send_command("HOME")
        assert result.success
        assert result.response.startswith("OK")
    finally:
        server.close()


def test_err_response():
    server = FakeArmServer(fail_commands={"HOME"})
    try:
        result = ArmClient("left", server.host, server.port, 1.0, 1.0, mock_mode=False).send_command("HOME")
        assert not result.success
        assert result.response.startswith("ERR")
    finally:
        server.close()


def test_connection_failure():
    result = ArmClient("left", "127.0.0.1", 9, 0.05, 0.05, mock_mode=False).send_command("HOME")
    assert not result.success
    assert "connection" in result.error


def test_timeout():
    server = FakeArmServer(delay_sec=0.3)
    try:
        result = ArmClient("left", server.host, server.port, 1.0, 0.05, mock_mode=False).send_command("HOME")
        assert not result.success
        assert "timeout" in result.error
    finally:
        server.close()


def test_mock_mode():
    result = ArmClient("left", "invalid", 1, mock_mode=True).send_command("STOP")
    assert result.success
    assert result.mock
    assert result.response.startswith("OK")
