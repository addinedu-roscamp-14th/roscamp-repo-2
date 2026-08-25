import json
import math
import socket
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from pymycobot.mycobot280 import MyCobot280


ROBOT_DEVICE = "/dev/ttyJETCOBOT"
ROBOT_BAUD = 1_000_000

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

READ_COUNT = 5
READ_INTERVAL_SEC = 0.10

MIN_SPEED = 1
MAX_SPEED = 10

# 넓은 비상 제한. 실제 핵심 제한은 현재 자세 대비 이동량 검사다.
WORKSPACE_LIMITS = {
    "x": (-350.0, 350.0),
    "y": (-350.0, 350.0),
    "z": (20.0, 500.0),
}

MAX_TRANSLATION_STEP_MM = 250.0
MAX_ROTATION_STEP_DEG = 100.0
MAX_JOINT_STEP_DEG = 120.0

GRIPPER_CLOSE_VALUE = 0
GRIPPER_OPEN_VALUE = 100
GRIPPER_SPEED = 50
GRIPPER_WAIT_SEC = 1.0


def log(*args: Any) -> None:
    print(*args, flush=True)


def valid_six(values: Any) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        return False

    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def median_six(samples: List[List[float]]) -> Optional[List[float]]:
    if not samples:
        return None

    return np.median(
        np.asarray(samples, dtype=np.float64),
        axis=0,
    ).tolist()


def wrapped_angle_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class Robot:
    def __init__(self) -> None:
        log("JetCobot 시리얼 연결:", ROBOT_DEVICE)

        self.mc = MyCobot280(
            ROBOT_DEVICE,
            ROBOT_BAUD,
        )

        try:
            log("power_on:", self.mc.power_on())
        except Exception as exc:
            log("power_on 경고:", repr(exc))

        time.sleep(0.8)

        # 등록 시 사용한 것과 동일하게 base / flange 기준
        try:
            self.mc.set_reference_frame(0)
            time.sleep(0.15)
            self.mc.set_end_type(0)
            time.sleep(0.15)
        except Exception as exc:
            log("frame 설정 경고:", repr(exc))

    def read_angles(self) -> Optional[List[float]]:
        samples: List[List[float]] = []

        for _ in range(READ_COUNT):
            try:
                values = self.mc.get_angles()

                if valid_six(values):
                    samples.append([float(v) for v in values])

            except Exception as exc:
                log("get_angles 경고:", repr(exc))

            time.sleep(READ_INTERVAL_SEC)

        return median_six(samples)

    def read_coords(self) -> Optional[List[float]]:
        samples: List[List[float]] = []

        for _ in range(READ_COUNT):
            try:
                values = self.mc.get_coords()

                if valid_six(values):
                    samples.append([float(v) for v in values])

            except Exception as exc:
                log("get_coords 경고:", repr(exc))

            time.sleep(READ_INTERVAL_SEC)

        return median_six(samples)

    def pose(self) -> Dict[str, Any]:
        angles = self.read_angles()
        coords = self.read_coords()

        if angles is None or coords is None:
            return {
                "ok": False,
                "error": "현재 angles 또는 coords를 읽지 못했습니다.",
                "angles_deg": angles,
                "coords_mm_deg": coords,
            }

        return {
            "ok": True,
            "angles_deg": angles,
            "coords_mm_deg": coords,
        }

    def validate_target(
        self,
        target: List[float],
        speed: int,
    ) -> Dict[str, Any]:
        if not valid_six(target):
            return {
                "ok": False,
                "error": "목표 coords는 유한한 숫자 6개여야 합니다.",
            }

        if speed < MIN_SPEED or speed > MAX_SPEED:
            return {
                "ok": False,
                "error": f"속도는 {MIN_SPEED}~{MAX_SPEED}만 허용합니다.",
            }

        x, y, z = [float(v) for v in target[:3]]

        for name, value in (("x", x), ("y", y), ("z", z)):
            lower, upper = WORKSPACE_LIMITS[name]

            if not lower <= value <= upper:
                return {
                    "ok": False,
                    "error": (
                        f"{name.upper()}={value:.1f}가 "
                        f"허용 범위 {lower:.1f}~{upper:.1f} 밖입니다."
                    ),
                }

        current = self.read_coords()

        if current is None:
            return {
                "ok": False,
                "error": "현재 좌표를 읽지 못해 이동을 차단했습니다.",
            }

        translation_delta = float(
            np.linalg.norm(
                np.asarray(target[:3], dtype=np.float64)
                - np.asarray(current[:3], dtype=np.float64)
            )
        )

        rotation_delta = max(
            wrapped_angle_delta_deg(target[i], current[i])
            for i in range(3, 6)
        )

        if translation_delta > MAX_TRANSLATION_STEP_MM:
            return {
                "ok": False,
                "error": (
                    f"현재 위치에서 목표까지 {translation_delta:.1f} mm입니다. "
                    f"한 번의 이동 제한 {MAX_TRANSLATION_STEP_MM:.1f} mm를 초과했습니다."
                ),
                "current_coords_mm_deg": current,
            }

        if rotation_delta > MAX_ROTATION_STEP_DEG:
            return {
                "ok": False,
                "error": (
                    f"회전 변화가 최대 {rotation_delta:.1f}°입니다. "
                    f"허용값 {MAX_ROTATION_STEP_DEG:.1f}°를 초과했습니다."
                ),
                "current_coords_mm_deg": current,
            }

        return {
            "ok": True,
            "current_coords_mm_deg": current,
            "translation_delta_mm": translation_delta,
            "maximum_rotation_delta_deg": rotation_delta,
        }

    def move_coords(
        self,
        target: List[float],
        speed: int,
        mode: int,
        position_tolerance_mm: float = 8.0,
        rotation_tolerance_deg: float = 6.0,
        stable_required: int = 2,
        timeout_sec: float = 30.0,
        require_tolerance: bool = False,
    ) -> Dict[str, Any]:
        target = [float(v) for v in target]
        speed = int(speed)
        mode = int(mode)
        position_tolerance_mm = float(position_tolerance_mm)
        rotation_tolerance_deg = float(rotation_tolerance_deg)
        stable_required = int(stable_required)
        timeout_sec = float(timeout_sec)
        require_tolerance = bool(require_tolerance)

        if not 0.5 <= position_tolerance_mm <= 15.0:
            return {
                "ok": False,
                "error": "position_tolerance_mm은 0.5~15.0만 허용합니다.",
            }

        if not 0.5 <= rotation_tolerance_deg <= 10.0:
            return {
                "ok": False,
                "error": "rotation_tolerance_deg은 0.5~10.0만 허용합니다.",
            }

        if not 1 <= stable_required <= 10:
            return {
                "ok": False,
                "error": "stable_required는 1~10만 허용합니다.",
            }

        if not 5.0 <= timeout_sec <= 60.0:
            return {
                "ok": False,
                "error": "timeout_sec은 5~60초만 허용합니다.",
            }

        validation = self.validate_target(target, speed)

        if not validation.get("ok"):
            return validation

        log()
        log("MOVE_COORDS 실행")
        log(
            "현재:",
            np.round(
                validation["current_coords_mm_deg"],
                3,
            ).tolist(),
        )
        log(
            "목표:",
            np.round(target, 3).tolist(),
        )
        log("속도:", speed, "mode:", mode)

        def position_error_mm(
            coords: List[float],
        ) -> float:
            return float(
                np.linalg.norm(
                    np.asarray(
                        coords[:3],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        target[:3],
                        dtype=np.float64,
                    )
                )
            )

        def rotation_error_deg(
            coords: List[float],
        ) -> float:
            return max(
                wrapped_angle_delta_deg(
                    coords[index],
                    target[index],
                )
                for index in range(3, 6)
            )

        try:
            # sync_send_coords가 실제 이동 완료 전에 반환되는 경우가 있어
            # 일반 send_coords 후 좌표를 직접 확인한다.
            result = self.mc.send_coords(
                target,
                speed,
                mode,
            )

            deadline = time.monotonic() + timeout_sec
            stable_count = 0
            command_retry_count = 0
            best_position_error_mm = float("inf")
            final_coords: Optional[List[float]] = None
            reached_requested_tolerance = False

            # 명령이 로봇 제어기에 반영될 시간을 준다.
            time.sleep(0.8)

            while time.monotonic() < deadline:
                coords = self.read_coords()

                if coords is None:
                    time.sleep(0.2)
                    continue

                final_coords = coords
                pos_error = position_error_mm(coords)
                rot_error = rotation_error_deg(coords)

                best_position_error_mm = min(
                    best_position_error_mm,
                    pos_error,
                )

                log(
                    "이동 확인 | 위치오차:",
                    f"{pos_error:.2f} mm |",
                    "회전오차:",
                    f"{rot_error:.2f} deg",
                )

                # 목표 근처에서 연속 두 번 확인되면 완료로 판정
                if (
                    pos_error <= position_tolerance_mm
                    and rot_error <= rotation_tolerance_deg
                ):
                    stable_count += 1

                    if stable_count >= stable_required:
                        reached_requested_tolerance = True
                        break
                else:
                    stable_count = 0

                # 명령 후에도 거의 움직이지 않았다면 한 번만 재전송
                elapsed = timeout_sec - (
                    deadline - time.monotonic()
                )

                if (
                    elapsed >= 3.0
                    and command_retry_count == 0
                    and pos_error
                    >= validation[
                        "translation_delta_mm"
                    ] - 8.0
                ):
                    log(
                        "이동 시작이 확인되지 않아 "
                        "명령을 1회 재전송합니다."
                    )
                    result = self.mc.send_coords(
                        target,
                        speed,
                        mode,
                    )
                    command_retry_count += 1

                time.sleep(0.25)

            if final_coords is None:
                return {
                    "ok": False,
                    "error": (
                        "이동 후 로봇 좌표를 "
                        "읽지 못했습니다."
                    ),
                    "target_coords_mm_deg": target,
                }

            final_pos_error = position_error_mm(
                final_coords
            )
            final_rot_error = rotation_error_deg(
                final_coords
            )

            if (
                require_tolerance
                and not reached_requested_tolerance
            ):
                try:
                    self.mc.stop()
                except Exception:
                    pass

                return {
                    "ok": False,
                    "error": (
                        "요청한 정밀 허용오차에 도달하지 못했습니다. "
                        "다음 동작을 차단합니다."
                    ),
                    "result": result,
                    "target_coords_mm_deg": target,
                    "final_coords_mm_deg": final_coords,
                    "final_position_error_mm": final_pos_error,
                    "final_rotation_error_deg": final_rot_error,
                    "requested_position_tolerance_mm": (
                        position_tolerance_mm
                    ),
                    "requested_rotation_tolerance_deg": (
                        rotation_tolerance_deg
                    ),
                    "stable_required": stable_required,
                    "best_position_error_mm": (
                        best_position_error_mm
                    ),
                    "command_retry_count": (
                        command_retry_count
                    ),
                }

            # 목표에 전혀 도달하지 못했는데도 성공으로 넘기지 않는다.
            if (
                final_pos_error > 15.0
                or final_rot_error > 10.0
            ):
                try:
                    self.mc.stop()
                except Exception:
                    pass

                return {
                    "ok": False,
                    "error": (
                        "목표 자세에 도달하지 못했습니다. "
                        "다음 단계 이동을 차단합니다."
                    ),
                    "result": result,
                    "target_coords_mm_deg": target,
                    "final_coords_mm_deg": final_coords,
                    "final_position_error_mm": (
                        final_pos_error
                    ),
                    "final_rotation_error_deg": (
                        final_rot_error
                    ),
                    "best_position_error_mm": (
                        best_position_error_mm
                    ),
                    "command_retry_count": (
                        command_retry_count
                    ),
                }

            return {
                "ok": True,
                "result": result,
                "target_coords_mm_deg": target,
                "final_coords_mm_deg": final_coords,
                "final_position_error_mm": (
                    final_pos_error
                ),
                "final_rotation_error_deg": (
                    final_rot_error
                ),
                "best_position_error_mm": (
                    best_position_error_mm
                ),
                "command_retry_count": (
                    command_retry_count
                ),
                "requested_position_tolerance_mm": (
                    position_tolerance_mm
                ),
                "requested_rotation_tolerance_deg": (
                    rotation_tolerance_deg
                ),
                "reached_requested_tolerance": (
                    reached_requested_tolerance
                ),
            }

        except Exception as exc:
            traceback.print_exc()

            try:
                self.mc.stop()
            except Exception:
                pass

            return {
                "ok": False,
                "error": str(exc),
            }


    def move_single_coord(
        self,
        axis_id: int,
        value: float,
        speed: int,
        tolerance: float = 1.0,
        timeout_sec: float = 8.0,
    ) -> Dict[str, Any]:
        """
        단일 Cartesian 축만 이동한다.
        axis_id:
          1=X, 2=Y, 3=Z, 4=Rx, 5=Ry, 6=Rz

        pymycobot의 send_coord()를 사용하고,
        이동 후 get_coords()로 실제 도착값을 확인한다.
        """
        try:
            axis_id = int(axis_id)
            value = float(value)
            speed = int(speed)
            tolerance = float(tolerance)
            timeout_sec = float(timeout_sec)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "error": "axis_id/value/speed 형식이 올바르지 않습니다.",
            }

        if axis_id < 1 or axis_id > 6:
            return {
                "ok": False,
                "error": "axis_id는 1~6만 허용합니다.",
            }

        if speed < MIN_SPEED or speed > MAX_SPEED:
            return {
                "ok": False,
                "error": f"속도는 {MIN_SPEED}~{MAX_SPEED}만 허용합니다.",
            }

        if not math.isfinite(value):
            return {
                "ok": False,
                "error": "목표 value는 유한한 숫자여야 합니다.",
            }

        if not 0.2 <= tolerance <= 10.0:
            return {
                "ok": False,
                "error": "tolerance는 0.2~10.0만 허용합니다.",
            }

        if not 2.0 <= timeout_sec <= 30.0:
            return {
                "ok": False,
                "error": "timeout_sec은 2~30초만 허용합니다.",
            }

        current = self.read_coords()

        if current is None:
            return {
                "ok": False,
                "error": "현재 좌표를 읽지 못해 단일축 이동을 차단했습니다.",
            }

        # XYZ는 기존 workspace 제한을 그대로 사용한다.
        if axis_id <= 3:
            axis_name = ("x", "y", "z")[axis_id - 1]
            lower, upper = WORKSPACE_LIMITS[axis_name]

            if not lower <= value <= upper:
                return {
                    "ok": False,
                    "error": (
                        f"{axis_name.upper()}={value:.1f}가 "
                        f"허용 범위 {lower:.1f}~{upper:.1f} 밖입니다."
                    ),
                }

            requested_delta = abs(value - float(current[axis_id - 1]))

            if requested_delta > MAX_TRANSLATION_STEP_MM:
                return {
                    "ok": False,
                    "error": (
                        f"단일축 이동량 {requested_delta:.1f}mm가 "
                        f"허용값 {MAX_TRANSLATION_STEP_MM:.1f}mm를 초과했습니다."
                    ),
                    "current_coords_mm_deg": current,
                }

        else:
            requested_delta = wrapped_angle_delta_deg(
                value,
                float(current[axis_id - 1]),
            )

            if requested_delta > MAX_ROTATION_STEP_DEG:
                return {
                    "ok": False,
                    "error": (
                        f"단일 회전축 변화 {requested_delta:.1f}°가 "
                        f"허용값 {MAX_ROTATION_STEP_DEG:.1f}°를 초과했습니다."
                    ),
                    "current_coords_mm_deg": current,
                }

        log()
        log("MOVE_SINGLE_COORD 실행")
        log(
            "축:",
            axis_id,
            "| 현재:",
            f"{float(current[axis_id - 1]):.3f}",
            "| 목표:",
            f"{value:.3f}",
            "| 속도:",
            speed,
        )

        def axis_error(coords: List[float]) -> float:
            actual = float(coords[axis_id - 1])

            if axis_id <= 3:
                return abs(actual - value)

            return wrapped_angle_delta_deg(actual, value)

        try:
            result = self.mc.send_coord(
                axis_id,
                value,
                speed,
            )

            deadline = time.monotonic() + timeout_sec
            final_coords: Optional[List[float]] = None
            best_axis_error = float("inf")
            command_retry_count = 0

            # 제어기에 명령이 반영될 시간을 조금 준다.
            time.sleep(0.45)

            while time.monotonic() < deadline:
                coords = self.read_coords()

                if coords is None:
                    time.sleep(0.15)
                    continue

                final_coords = coords
                error = axis_error(coords)
                best_axis_error = min(best_axis_error, error)

                log(
                    "단일축 이동 확인 |",
                    f"axis {axis_id} 실제 {float(coords[axis_id - 1]):.3f} |",
                    f"오차 {error:.3f}",
                )

                if error <= tolerance:
                    break

                # 2초 이상 지나도 거의 시작하지 않았다면 딱 한 번 재전송.
                elapsed = timeout_sec - (
                    deadline - time.monotonic()
                )

                if (
                    elapsed >= 2.0
                    and command_retry_count == 0
                    and requested_delta > 0.5
                    and error >= max(requested_delta - 0.25, tolerance)
                ):
                    log(
                        "단일축 이동 시작이 충분히 확인되지 않아 "
                        "명령을 1회 재전송합니다."
                    )
                    result = self.mc.send_coord(
                        axis_id,
                        value,
                        speed,
                    )
                    command_retry_count += 1

                time.sleep(0.15)

            if final_coords is None:
                return {
                    "ok": False,
                    "error": "단일축 이동 후 좌표를 읽지 못했습니다.",
                    "axis_id": axis_id,
                    "target_value": value,
                }

            final_error = axis_error(final_coords)

            # 클라이언트가 실제 변화량을 보고 다음 1mm를 결정하므로,
            # 여기서는 너무 엄격하게 실패시키지 않는다.
            # 단, 전혀 반응하지 않은 경우는 실패 처리한다.
            actual_delta = abs(
                float(final_coords[axis_id - 1])
                - float(current[axis_id - 1])
            )

            if requested_delta > 0.5 and actual_delta < 0.10:
                try:
                    self.mc.stop()
                except Exception:
                    pass

                return {
                    "ok": False,
                    "error": "단일축 명령 후 실제 축 이동이 거의 확인되지 않았습니다.",
                    "result": result,
                    "axis_id": axis_id,
                    "target_value": value,
                    "start_coords_mm_deg": current,
                    "final_coords_mm_deg": final_coords,
                    "final_axis_error": final_error,
                    "actual_axis_delta": actual_delta,
                    "best_axis_error": best_axis_error,
                    "command_retry_count": command_retry_count,
                }

            return {
                "ok": True,
                "result": result,
                "axis_id": axis_id,
                "target_value": value,
                "start_coords_mm_deg": current,
                "final_coords_mm_deg": final_coords,
                "final_axis_error": final_error,
                "actual_axis_delta": actual_delta,
                "best_axis_error": best_axis_error,
                "command_retry_count": command_retry_count,
            }

        except Exception as exc:
            traceback.print_exc()

            try:
                self.mc.stop()
            except Exception:
                pass

            return {
                "ok": False,
                "error": str(exc),
            }

    def move_angles(
        self,
        target: List[float],
        speed: int,
        angle_tolerance_deg: float = 2.0,
        stable_required: int = 2,
        timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """관절각 목표로 이동하고 실제 관절각 도착 여부를 확인한다."""
        if not valid_six(target):
            return {
                "ok": False,
                "error": "목표 angles는 유한한 숫자 6개여야 합니다.",
            }

        target = [float(v) for v in target]
        speed = int(speed)
        angle_tolerance_deg = float(angle_tolerance_deg)
        stable_required = int(stable_required)
        timeout_sec = float(timeout_sec)

        if speed < MIN_SPEED or speed > MAX_SPEED:
            return {
                "ok": False,
                "error": f"속도는 {MIN_SPEED}~{MAX_SPEED}만 허용합니다.",
            }

        if not 0.5 <= angle_tolerance_deg <= 10.0:
            return {
                "ok": False,
                "error": "angle_tolerance_deg은 0.5~10.0만 허용합니다.",
            }

        if not 1 <= stable_required <= 10:
            return {
                "ok": False,
                "error": "stable_required는 1~10만 허용합니다.",
            }

        if not 5.0 <= timeout_sec <= 60.0:
            return {
                "ok": False,
                "error": "timeout_sec은 5~60초만 허용합니다.",
            }

        current = self.read_angles()

        if current is None:
            return {
                "ok": False,
                "error": "현재 관절각을 읽지 못해 이동을 차단했습니다.",
            }

        joint_deltas = [
            wrapped_angle_delta_deg(target[i], current[i])
            for i in range(6)
        ]
        maximum_joint_delta = max(joint_deltas)

        if maximum_joint_delta > MAX_JOINT_STEP_DEG:
            return {
                "ok": False,
                "error": (
                    f"관절 변화가 최대 {maximum_joint_delta:.1f}°입니다. "
                    f"한 번의 이동 제한 {MAX_JOINT_STEP_DEG:.1f}°를 초과했습니다."
                ),
                "current_angles_deg": current,
                "joint_deltas_deg": joint_deltas,
            }

        log()
        log("MOVE_ANGLES 실행")
        log("현재:", np.round(current, 3).tolist())
        log("목표:", np.round(target, 3).tolist())
        log("속도:", speed)

        def maximum_error_deg(angles: List[float]) -> float:
            return max(
                wrapped_angle_delta_deg(angles[i], target[i])
                for i in range(6)
            )

        try:
            result = self.mc.send_angles(target, speed)
            deadline = time.monotonic() + timeout_sec
            stable_count = 0
            final_angles: Optional[List[float]] = None
            best_error_deg = float("inf")

            time.sleep(0.8)

            while time.monotonic() < deadline:
                angles = self.read_angles()

                if angles is None:
                    time.sleep(0.2)
                    continue

                final_angles = angles
                error_deg = maximum_error_deg(angles)
                best_error_deg = min(best_error_deg, error_deg)

                log(
                    "관절 이동 확인 | 최대 관절오차:",
                    f"{error_deg:.2f} deg",
                )

                if error_deg <= angle_tolerance_deg:
                    stable_count += 1

                    if stable_count >= stable_required:
                        break
                else:
                    stable_count = 0

                time.sleep(0.25)

            if final_angles is None:
                return {
                    "ok": False,
                    "error": "이동 후 관절각을 읽지 못했습니다.",
                    "target_angles_deg": target,
                }

            final_error_deg = maximum_error_deg(final_angles)

            if stable_count < stable_required:
                try:
                    self.mc.stop()
                except Exception:
                    pass

                return {
                    "ok": False,
                    "error": (
                        "관측 자세의 관절각 허용오차에 도달하지 못했습니다. "
                        "ArUco 재검출을 차단합니다."
                    ),
                    "result": result,
                    "target_angles_deg": target,
                    "final_angles_deg": final_angles,
                    "final_maximum_joint_error_deg": final_error_deg,
                    "best_maximum_joint_error_deg": best_error_deg,
                }

            return {
                "ok": True,
                "result": result,
                "target_angles_deg": target,
                "final_angles_deg": final_angles,
                "final_maximum_joint_error_deg": final_error_deg,
                "best_maximum_joint_error_deg": best_error_deg,
            }

        except Exception as exc:
            traceback.print_exc()

            try:
                self.mc.stop()
            except Exception:
                pass

            return {
                "ok": False,
                "error": str(exc),
            }

    def gripper_close(self) -> Dict[str, Any]:
        """그리퍼를 완전히 닫는다."""
        try:
            log("그리퍼를 완전히 닫습니다.")
            result = self.mc.set_gripper_value(
                GRIPPER_CLOSE_VALUE,
                GRIPPER_SPEED,
            )
            time.sleep(GRIPPER_WAIT_SEC)

            return {
                "ok": True,
                "result": result,
                "gripper_value": GRIPPER_CLOSE_VALUE,
                "gripper_speed": GRIPPER_SPEED,
            }

        except Exception as exc:
            traceback.print_exc()

            return {
                "ok": False,
                "error": str(exc),
            }

    def gripper_open(self) -> Dict[str, Any]:
        """그리퍼를 완전히 연다."""
        try:
            log("그리퍼를 완전히 엽니다.")
            result = self.mc.set_gripper_value(
                GRIPPER_OPEN_VALUE,
                GRIPPER_SPEED,
            )
            time.sleep(GRIPPER_WAIT_SEC)

            return {
                "ok": True,
                "result": result,
                "gripper_value": GRIPPER_OPEN_VALUE,
                "gripper_speed": GRIPPER_SPEED,
            }

        except Exception as exc:
            traceback.print_exc()

            return {
                "ok": False,
                "error": str(exc),
            }

    def stop(self) -> Dict[str, Any]:
        try:
            result = self.mc.stop()

            return {
                "ok": True,
                "result": result,
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
            }

    def release(self) -> Dict[str, Any]:
        try:
            result = self.mc.release_all_servos()

            return {
                "ok": True,
                "result": result,
                "warning": "서보가 풀렸습니다. 로봇팔을 손으로 받치세요.",
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
            }

    def lock(self) -> Dict[str, Any]:
        try:
            if hasattr(self.mc, "focus_all_servos"):
                result = self.mc.focus_all_servos()
            else:
                result = self.mc.power_on()

            time.sleep(0.8)

            return {
                "ok": True,
                "result": result,
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
            }

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "reference_frame": 0,
            "end_type": 0,
            "workspace_limits": WORKSPACE_LIMITS,
            "maximum_translation_step_mm": MAX_TRANSLATION_STEP_MM,
            "maximum_rotation_step_deg": MAX_ROTATION_STEP_DEG,
            "maximum_joint_step_deg": MAX_JOINT_STEP_DEG,
            "gripper_close_value": GRIPPER_CLOSE_VALUE,
            "gripper_open_value": GRIPPER_OPEN_VALUE,
            "gripper_speed": GRIPPER_SPEED,
        }


def parse_request(line: str) -> Dict[str, Any]:
    stripped = line.strip()

    if not stripped:
        return {"command": ""}

    if stripped.startswith("{"):
        payload = json.loads(stripped)

        if not isinstance(payload, dict):
            raise ValueError("JSON 요청은 객체여야 합니다.")

        return payload

    return {"command": stripped}


def send_json(
    conn: socket.socket,
    payload: Dict[str, Any],
) -> None:
    conn.sendall(
        (
            json.dumps(payload, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    )


def main() -> None:
    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )
    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(1)

    log("5000번 포트 열림:", f"{SERVER_HOST}:{SERVER_PORT}")

    robot = Robot()

    log()
    log("=" * 62)
    log("JetCobot 슬롯 재현·관측 자세 서버 v6 준비 완료")
    log("지원: MOVE_ANGLES / MOVE_COORDS / MOVE_SINGLE_COORD / GRIPPER_OPEN / GRIPPER_CLOSE / POSE / STOP")
    log("=" * 62)

    conn: Optional[socket.socket] = None

    try:
        conn, address = server.accept()
        conn.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

        log("클라이언트 연결:", address)

        buffer = ""

        while True:
            data = conn.recv(4096)

            if not data:
                log("클라이언트 연결 종료")
                break

            buffer += data.decode(
                "utf-8",
                errors="ignore",
            )

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)

                if not line.strip():
                    continue

                try:
                    request = parse_request(line)
                    command = str(
                        request.get("command", "")
                    ).strip().upper()

                    log("수신:", command)

                    if command == "PING":
                        response = {
                            "ok": True,
                            "reply": "PONG",
                        }

                    elif command == "STATUS":
                        response = robot.status()

                    elif command == "POSE":
                        response = robot.pose()

                    elif command == "MOVE_COORDS":
                        response = robot.move_coords(
                            request.get("coords", []),
                            int(request.get("speed", 5)),
                            int(request.get("mode", 0)),
                            float(
                                request.get(
                                    "position_tolerance_mm",
                                    8.0,
                                )
                            ),
                            float(
                                request.get(
                                    "rotation_tolerance_deg",
                                    6.0,
                                )
                            ),
                            int(
                                request.get(
                                    "stable_required",
                                    2,
                                )
                            ),
                            float(
                                request.get(
                                    "timeout_sec",
                                    30.0,
                                )
                            ),
                            bool(
                                request.get(
                                    "require_tolerance",
                                    False,
                                )
                            ),
                        )

                    elif command == "MOVE_SINGLE_COORD":
                        response = robot.move_single_coord(
                            int(request.get("axis_id", 0)),
                            float(request.get("value", 0.0)),
                            int(request.get("speed", 1)),
                            float(request.get("tolerance", 1.0)),
                            float(request.get("timeout_sec", 8.0)),
                        )

                    elif command == "MOVE_ANGLES":
                        response = robot.move_angles(
                            request.get("angles", []),
                            int(request.get("speed", 3)),
                            float(
                                request.get(
                                    "angle_tolerance_deg",
                                    2.0,
                                )
                            ),
                            int(
                                request.get(
                                    "stable_required",
                                    2,
                                )
                            ),
                            float(
                                request.get(
                                    "timeout_sec",
                                    30.0,
                                )
                            ),
                        )

                    elif command == "GRIPPER_OPEN":
                        response = robot.gripper_open()

                    elif command == "GRIPPER_CLOSE":
                        response = robot.gripper_close()

                    elif command == "STOP":
                        response = robot.stop()

                    elif command == "LOCK":
                        response = robot.lock()

                    elif command == "RELEASE":
                        response = robot.release()

                    elif command == "QUIT":
                        response = robot.lock()
                        send_json(conn, response)
                        return

                    else:
                        response = {
                            "ok": False,
                            "error": f"지원하지 않는 명령: {command}",
                        }

                except Exception as exc:
                    traceback.print_exc()

                    response = {
                        "ok": False,
                        "error": str(exc),
                    }

                send_json(conn, response)

    except KeyboardInterrupt:
        log("\nCtrl+C 입력")

    finally:
        try:
            robot.stop()
        except Exception:
            pass

        try:
            robot.lock()
        except Exception:
            pass

        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

        server.close()
        log("서버 종료")


if __name__ == "__main__":
    main()