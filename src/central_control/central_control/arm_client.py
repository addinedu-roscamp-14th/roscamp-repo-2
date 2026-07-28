from __future__ import annotations

from dataclasses import dataclass
import socket


SUPPORTED_COMMANDS = {"HELP", "SET_CAMERA", "REMOVE_BAD_TIRE", "SET_ARUCO", "HOME", "STOP"}


@dataclass(frozen=True)
class ArmCommandResult:
    arm: str
    command: str
    success: bool
    response: str = ""
    error: str | None = None
    mock: bool = False


class ArmClient:
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        connect_timeout_sec: float = 3.0,
        command_timeout_sec: float = 180.0,
        mock_mode: bool = True,
    ):
        self.name = str(name)
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)
        self.mock_mode = bool(mock_mode)

    def send_command(self, command: str) -> ArmCommandResult:
        command = str(command).strip().upper()
        if command not in SUPPORTED_COMMANDS:
            return ArmCommandResult(self.name, command, False, error=f"unsupported command: {command}")
        if self.mock_mode:
            return ArmCommandResult(self.name, command, True, response=f"OK:MOCK:{command}", mock=True)

        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_sec
            ) as sock:
                sock.settimeout(self.command_timeout_sec)
                sock.sendall((command + "\n").encode("utf-8"))
                response = sock.recv(1024).decode("utf-8", errors="replace").strip()
        except socket.timeout:
            return ArmCommandResult(self.name, command, False, error="arm command timeout")
        except OSError as error:
            return ArmCommandResult(self.name, command, False, error=f"arm connection error: {error}")

        if response.upper().startswith("OK"):
            return ArmCommandResult(self.name, command, True, response=response)
        return ArmCommandResult(self.name, command, False, response=response, error=f"arm rejected command: {response}")
