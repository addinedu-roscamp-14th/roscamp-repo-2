"""Protocol-compatible client for the left JetCobot arm."""
from __future__ import annotations

from arm_task_coordinator.tcp_arm_client import ArmTcpError, TcpArmClient


class LeftArmRobotClient:
    """
    Adapter used by left_arm_transport_pick.

    It deliberately keeps the movement options from the previously validated
    right_arm_client.py instead of collapsing them to only speed/mode/timeout.
    """

    def __init__(
        self,
        host,
        port,
        connect_timeout_sec=3.0,
        command_timeout_sec=30.0,
    ):
        self._client = TcpArmClient(
            host,
            int(port),
            float(connect_timeout_sec),
            float(command_timeout_sec),
        )
        self.last_error = None

    @property
    def host(self):
        return self._client.host

    @property
    def port(self):
        return self._client.port

    def _call(self, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
            self.last_error = None
            return result
        except (ArmTcpError, OSError, TimeoutError) as error:
            self.last_error = str(error)
            raise

    def ping(self):
        self.get_coords()
        return True

    def status(self):
        try:
            coords = self.get_coords()
            return {
                "available": True,
                "coords": coords,
                "error": None,
            }
        except Exception as error:
            return {
                "available": False,
                "coords": None,
                "error": str(error),
            }

    def pose(self):
        return self.get_coords()

    def get_coords(self):
        values = self._call(self._client.get_coords)
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise RuntimeError(
                f"invalid GET_COORDS result: {values!r}"
            )
        return [float(value) for value in values]

    def get_angles(self):
        values = self._call(self._client.get_angles)
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise RuntimeError(
                f"invalid GET_ANGLES result: {values!r}"
            )
        return [float(value) for value in values]

    def set_aruco(self):
        return self._call(
            self._client.command,
            "SET_ARUCO",
        )

    def observe_aruco(self):
        result = self._call(
            self._client.command,
            "OBSERVE_ARUCO",
        )
        return {
            "ok": True,
            "result": result,
            "final_coords_mm_deg": self.get_coords(),
        }

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
        coords = [float(value) for value in coords]

        try:
            response = self._call(
                self._client.move_coords,
                coords,
                int(speed),
                int(mode),
                timeout_sec,
                position_tolerance_mm,
                rotation_tolerance_deg,
                stable_required,
                require_tolerance,
            )
        except ArmTcpError as error:
            # Hard require_tolerance failures still expose final pose/error data.
            if error.code == "TARGET_TIMEOUT" and error.data:
                return {
                    "ok": False,
                    "error": (
                        "목표 자세에 도달하지 못했습니다. "
                        "다음 단계 이동을 차단합니다."
                    ),
                    **error.data,
                }
            raise

        data = dict(response.get("data", {}))

        # The server can return status=ok but data.ok=False for ordinary
        # coarse/fine timeout. This is intentional and matches the old GUI
        # program's need to inspect final errors.
        result_ok = bool(data.get("ok", data.get("reached", True)))

        result = {
            "ok": result_ok,
            **data,
        }

        if not result_ok:
            result.setdefault(
                "error",
                (
                    "목표 자세에 도달하지 못했습니다. "
                    "다음 단계 이동을 차단합니다."
                ),
            )

        if result.get("final_coords_mm_deg") is None:
            result["final_coords_mm_deg"] = self.get_coords()

        return result

    def move_single_coord(
        self,
        axis_id,
        value,
        speed=15,
        mode=0,
        timeout_sec=None,
        position_tolerance_mm=None,
        rotation_tolerance_deg=None,
        stable_required=None,
        require_tolerance=None,
    ):
        axis_id = int(axis_id)
        if axis_id < 1 or axis_id > 6:
            raise ValueError("axis_id must be in [1, 6]")

        coords = self.get_coords()
        coords[axis_id - 1] = float(value)

        return self.move_coords(
            coords,
            speed=speed,
            mode=mode,
            timeout_sec=timeout_sec,
            position_tolerance_mm=position_tolerance_mm,
            rotation_tolerance_deg=rotation_tolerance_deg,
            stable_required=stable_required,
            require_tolerance=require_tolerance,
        )

    def gripper_open(self, speed=50):
        return self._call(
            self._client.command,
            "GRIPPER_OPEN",
            speed=int(speed),
        )

    def gripper_close(self, speed=50):
        return self._call(
            self._client.command,
            "GRIPPER_CLOSE",
            speed=int(speed),
        )

    def stop(self):
        return self._call(self._client.stop)

    def reset_emergency(self):
        return self._call(self._client.reset_emergency)
