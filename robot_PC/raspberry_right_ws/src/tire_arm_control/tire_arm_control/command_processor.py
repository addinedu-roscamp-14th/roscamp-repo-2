from __future__ import annotations
import json
import math
import threading
import time


class CommandError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


LEGACY_MOTIONS = {
    "HOME": "home",
    "HELP": "help",
    "REMOVE_BAD_TIRE": "remove_bad_tire",
    "SET_CAMERA": "set_camera",
    "SET_ARUCO": "set_aruco",
}


class ArmCommandProcessor:
    def __init__(self, arm, player, config):
        self.arm = arm
        self.player = player
        self.config = dict(config)
        self.busy_lock = threading.Lock()
        self.emergency = threading.Event()

    def process_line(self, line):
        text = str(line).strip()
        if not text:
            return "ERR:EMPTY"
        if text.startswith("{"):
            try:
                request = json.loads(text)
            except json.JSONDecodeError as error:
                return json.dumps(self._error(None, "INVALID_JSON", str(error)), separators=(",", ":"))
            return json.dumps(self.process_json(request), separators=(",", ":"))
        return self.process_plain(text)

    def process_plain(self, text):
        command = text.strip().upper()
        if command == "STOP":
            self._stop()
            return "OK:STOP"
        if self.emergency.is_set():
            return "ERR:EMERGENCY_ACTIVE"
        if command not in LEGACY_MOTIONS:
            return f"ERR:UNKNOWN_COMMAND:{command}"
        if not self.busy_lock.acquire(blocking=False):
            return "ERR:BUSY"
        try:
            self.player.play(LEGACY_MOTIONS[command])
            return f"OK:{LEGACY_MOTIONS[command]}"
        except Exception as error:
            return f"ERR:MOTION_FAILED:{error}"
        finally:
            self.busy_lock.release()

    def process_json(self, request):
        request_id = request.get("request_id") if isinstance(request, dict) else None
        if not isinstance(request, dict):
            return self._error(request_id, "INVALID_REQUEST", "request must be an object")
        command = str(request.get("command", "")).upper()
        if command == "STOP":
            self._stop()
            return self._ok(request_id, {"stopped": True})
        if command == "RESET_EMERGENCY":
            if self.busy_lock.locked():
                return self._error(request_id, "BUSY", "cannot reset while command is active")
            self.emergency.clear()
            return self._ok(request_id, {"emergency": False})
        if command in {"GET_COORDS", "GET_ANGLES"}:
            try:
                values = self.arm.get_coords() if command == "GET_COORDS" else self.arm.get_angles()
                return self._ok(request_id, {
                    "coords" if command == "GET_COORDS" else "angles": values
                })
            except Exception as error:
                return self._error(request_id, "ROBOT_READ_FAILED", str(error))
        if self.emergency.is_set():
            return self._error(request_id, "EMERGENCY_ACTIVE", "STOP is latched; reset is required")
        if command not in {
            "MOVE_COORDS", "GRIPPER_OPEN", "GRIPPER_CLOSE", "PLAY_MOTION"
        }:
            return self._error(request_id, "UNKNOWN_COMMAND", f"unknown command: {command}")
        if not self.busy_lock.acquire(blocking=False):
            return self._error(request_id, "BUSY", "another command is in progress")
        try:
            if command == "MOVE_COORDS":
                data = self._move_coords(request)
            elif command == "PLAY_MOTION":
                motion = str(request.get("motion", ""))
                self.player.play(motion)
                data = {"motion": motion}
            else:
                opened = command == "GRIPPER_OPEN"
                value = int(self.config.get(
                    "gripper_open_value" if opened else "gripper_close_value",
                    100 if opened else 0,
                ))
                self.arm.set_gripper(value, int(request.get("speed", 50)))
                data = {"gripper": "open" if opened else "closed", "value": value}
            return self._ok(request_id, data)
        except CommandError as error:
            return self._error(request_id, error.code, str(error))
        except Exception as error:
            return self._error(request_id, "EXECUTION_FAILED", str(error))
        finally:
            self.busy_lock.release()

    def _move_coords(self, request):
        coords = self._validate_coords(request.get("coords"))
        speed = int(request.get("speed", self.config.get("default_coord_speed", 30)))
        mode = int(request.get("mode", 0))
        if not 1 <= speed <= 100:
            raise CommandError("INVALID_SPEED", "speed must be between 1 and 100")
        if mode not in (0, 1):
            raise CommandError("INVALID_MODE", "mode must be 0 or 1")
        timeout = float(request.get(
            "timeout_sec", self.config.get("move_timeout_sec", 30.0)
        ))
        self.arm.send_coords(coords, speed, mode)
        deadline = time.monotonic() + timeout
        position_tolerance = float(self.config.get("position_tolerance_mm", 5.0))
        rotation_tolerance = float(self.config.get("rotation_tolerance_deg", 5.0))
        while time.monotonic() < deadline:
            if self.emergency.is_set():
                raise CommandError("STOPPED", "movement interrupted by STOP")
            actual = self.arm.get_coords()
            position_error = math.sqrt(sum(
                (actual[index] - coords[index]) ** 2 for index in range(3)
            ))
            rotation_error = max(
                abs((actual[index] - coords[index] + 180.0) % 360.0 - 180.0)
                for index in range(3, 6)
            )
            if position_error <= position_tolerance and rotation_error <= rotation_tolerance:
                return {
                    "target": coords,
                    "actual": actual,
                    "position_error_mm": position_error,
                    "rotation_error_deg": rotation_error,
                    "reached": True,
                }
            time.sleep(float(self.config.get("move_poll_interval_sec", 0.1)))
        raise CommandError("TARGET_TIMEOUT", "target was not reached before timeout")

    @staticmethod
    def _validate_coords(values):
        if not isinstance(values, list) or len(values) != 6:
            raise CommandError("INVALID_COORDS", "coords must be a six-element JSON array")
        try:
            coords = [float(value) for value in values]
        except (TypeError, ValueError):
            raise CommandError("INVALID_COORDS", "coords must contain numbers")
        if not all(math.isfinite(value) for value in coords):
            raise CommandError("INVALID_COORDS", "coords must be finite")
        limits = [(-500, 500), (-500, 500), (-100, 600)] + [(-180, 180)] * 3
        if any(not low <= value <= high for value, (low, high) in zip(coords, limits)):
            raise CommandError("COORDS_OUT_OF_RANGE", "coords exceed configured robot safety envelope")
        return coords

    def _stop(self):
        self.emergency.set()
        self.player.stop()

    @staticmethod
    def _ok(request_id, data):
        return {"version": 1, "request_id": request_id, "status": "ok", "data": data}

    @staticmethod
    def _error(request_id, code, message):
        return {
            "version": 1, "request_id": request_id, "status": "error",
            "error": {"code": code, "message": str(message)},
        }
