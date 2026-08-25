from __future__ import annotations

import json
import socket
import uuid


class ArmTcpError(RuntimeError):
    def __init__(self, message, code=None, data=None, response=None):
        super().__init__(message)
        self.code = code
        self.data = dict(data or {})
        self.response = response


class TcpArmClient:
    def __init__(
        self,
        host,
        port,
        connect_timeout_sec=3.0,
        command_timeout_sec=30.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)

    def command(self, command, **parameters):
        request = {
            "version": 1,
            "request_id": f"arm-{uuid.uuid4()}",
            "command": str(command).upper(),
            **parameters,
        }

        response = self._exchange(
            json.dumps(request, separators=(",", ":"))
        )

        if not isinstance(response, dict):
            raise ArmTcpError(
                f"non-JSON response to JSON command: {response}"
            )

        if response.get("status") != "ok":
            error = response.get("error", {})
            raise ArmTcpError(
                f"{error.get('code', 'ARM_ERROR')}: "
                f"{error.get('message', response)}",
                code=error.get("code"),
                data=response.get("data"),
                response=response,
            )

        return response

    def legacy_command(self, command):
        return self._exchange(str(command).strip())

    def stop(self):
        return self.command("STOP")

    def reset_emergency(self):
        return self.command("RESET_EMERGENCY")

    def play_motion(self, motion):
        return self.command(
            "PLAY_MOTION",
            motion=str(motion),
        )

    def move_coords(
        self,
        coords,
        speed=30,
        mode=0,
        timeout_sec=None,
        position_tolerance_mm=None,
        rotation_tolerance_deg=None,
        stable_required=None,
        require_tolerance=None,
    ):
        parameters = {
            "coords": list(coords),
            "speed": int(speed),
            "mode": int(mode),
        }

        if timeout_sec is not None:
            parameters["timeout_sec"] = float(timeout_sec)
        if position_tolerance_mm is not None:
            parameters["position_tolerance_mm"] = float(
                position_tolerance_mm
            )
        if rotation_tolerance_deg is not None:
            parameters["rotation_tolerance_deg"] = float(
                rotation_tolerance_deg
            )
        if stable_required is not None:
            parameters["stable_required"] = int(stable_required)
        if require_tolerance is not None:
            parameters["require_tolerance"] = bool(require_tolerance)

        return self.command(
            "MOVE_COORDS",
            **parameters,
        )

    def get_coords(self):
        return self.command("GET_COORDS")["data"]["coords"]

    def get_angles(self):
        return self.command("GET_ANGLES")["data"]["angles"]

    def _exchange(self, line):
        payload = (line + "\n").encode("utf-8")

        with socket.create_connection(
            (self.host, self.port),
            timeout=self.connect_timeout_sec,
        ) as connection:
            connection.settimeout(self.command_timeout_sec)
            connection.sendall(payload)

            chunks = []
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break

                chunks.append(chunk)

                if b"\n" in chunk:
                    break

        raw = (
            b"".join(chunks)
            .split(b"\n", 1)[0]
            .decode("utf-8")
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
