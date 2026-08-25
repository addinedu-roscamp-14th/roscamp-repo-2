"""Left-arm approach/grasp sequence restored from the validated left_arm_client.py."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np

from .coordinate_transform import validate_robot_coords
from arm_aruco_perception.slot_pose_calculator import transform_to_coords


# ---------------------------------------------------------------------------
# Values restored from the working left_arm_client.py
# ---------------------------------------------------------------------------

TEST_SPEED = 4
GRASP_TEST_SPEED = 2
MOVE_MODE = 0
SAFE_Z_MARGIN_MM = 40.0
DIRECT_APPROACH_MAX_DISTANCE_MM = 100.0

MAX_APPROACH_POSITION_ERROR_MM = 20.0
MAX_APPROACH_ROTATION_ERROR_DEG = 10.0
MAX_GRASP_TRANSLATION_STEP_MM = 120.0
MAX_GRASP_ROTATION_STEP_DEG = 35.0

FINE_SETTLE_SPEED = 1
FINE_SETTLE_MIN_POSITION_MM = 2.5
FINE_SETTLE_MAX_POSITION_MM = 15.0
FINE_SETTLE_MAX_ROTATION_DEG = 10.0
FINE_SETTLE_GAIN = 1.0
FINE_SETTLE_MAX_AXIS_MM = 6.0

COARSE_MOVE_POSITION_TOLERANCE_MM = 12.0
COARSE_MOVE_ROTATION_TOLERANCE_DEG = 6.0
COARSE_MOVE_STABLE_REQUIRED = 1
COARSE_MOVE_TIMEOUT_SEC = 10.0

FINE_MOVE_POSITION_TOLERANCE_MM = 12.0
FINE_MOVE_ROTATION_TOLERANCE_DEG = 6.0
FINE_MOVE_STABLE_REQUIRED = 1
FINE_MOVE_TIMEOUT_SEC = 8.0

INTER_WAYPOINT_SETTLE_SEC = 0.2

SLOT3_INTERMEDIATE_POSITION_ACCEPT_MM = 12.0
SLOT3_INTERMEDIATE_ROTATION_ACCEPT_DEG = 15.0

APPROACH_POSITION_COMPENSATION_MM = {
    1: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    3: np.array([2.0, -7.0, 0.0], dtype=np.float64),
    4: np.array([-3.0, -7.0, 0.0], dtype=np.float64),
}

GRASP_PHYSICAL_DEPTH_OFFSET_MM = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
}

GRASP_POSITION_COMPENSATION_MM = {
    1: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    3: np.array([2.0, -7.0, 0.0], dtype=np.float64),
    4: np.array([-3.0, -7.0, 0.0], dtype=np.float64),
}

GRASP_ROTATION_COMPENSATION_DEG = {
    1: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    3: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    4: np.array([0.0, 0.0, 0.0], dtype=np.float64),
}

SLOT4_GRASP_Y_FINE_GAIN = 0.65

# ---------------------------------------------------------------------------
# Left-arm slot-specific final correction
# ---------------------------------------------------------------------------

# Slot 2: after the coarse grasp move, advance only in +Y in 1 mm steps.
SLOT2_SINGLE_Y_SPEED = 1
SLOT2_SINGLE_Y_STEP_MM = 1.0
SLOT2_SINGLE_Y_TOLERANCE_MM = 1.2
SLOT2_SINGLE_Y_MAX_TOTAL_MM = 8.0
SLOT2_SINGLE_Y_MAX_STEPS = 8
SLOT2_SINGLE_Y_MAX_TOTAL_X_DRIFT_MM = 3.0
SLOT2_SINGLE_Y_MAX_TOTAL_Z_DRIFT_MM = 3.0
SLOT2_SINGLE_Y_MAX_TOTAL_ROTATION_DRIFT_DEG = 4.0

# Slot 1: after the coarse grasp move, settle Y then Z one axis at a time.
SLOT1_AUTO_FINAL_SPEED = 1
SLOT1_AUTO_FINAL_STEP_MM = 1.0
SLOT1_AUTO_FINAL_Y_TOLERANCE_MM = 1.5
SLOT1_AUTO_FINAL_Z_TOLERANCE_MM = 1.5
SLOT1_AUTO_FINAL_MAX_Y_MM = 8.0
SLOT1_AUTO_FINAL_MAX_Z_MM = 12.0
SLOT1_AUTO_FINAL_MAX_Y_STEPS = 8
SLOT1_AUTO_FINAL_MAX_Z_STEPS = 12
SLOT1_AUTO_FINAL_MAX_X_DRIFT_MM = 4.0
SLOT1_AUTO_FINAL_MAX_ROTATION_DRIFT_DEG = 5.0
SLOT1_AUTO_FINAL_MIN_PROGRESS_MM = 0.15


def wrapped_angle_error_deg(actual_deg, target_deg):
    return abs(
        (float(actual_deg) - float(target_deg) + 180.0)
        % 360.0
        - 180.0
    )


def pose_error(actual, target):
    actual_array = np.asarray(actual, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)

    position_error_mm = float(
        np.linalg.norm(actual_array[:3] - target_array[:3])
    )
    rotation_error_deg = max(
        wrapped_angle_error_deg(
            actual_array[index],
            target_array[index],
        )
        for index in range(3, 6)
    )

    return position_error_mm, rotation_error_deg


@dataclass(frozen=True)
class PickSafety:
    preview_only: bool = True
    allow_robot_motion: bool = False
    allow_gripper_close: bool = False


class PickSequence:
    def __init__(
        self,
        client,
        safety,
        position_limits_mm,
        rotation_limits_deg,
        logger=None,
    ):
        self.client = client
        self.safety = safety
        self.position_limits_mm = position_limits_mm
        self.rotation_limits_deg = rotation_limits_deg
        self.log = logger or logging.getLogger(__name__)


    def _auto_single_coord_settle_slot1(
        self,
        axis_id,
        axis_name,
        target_value,
        tolerance_mm,
        max_total_mm,
        max_steps,
        runner,
    ):
        """
        Slot 1 grasp coarse 이동 후 지정 단일축만 1 mm씩 자동 보정한다.
        axis_id: 1=X, 2=Y, 3=Z.
        """
        axis_index = int(axis_id) - 1

        baseline = np.asarray(
            self.client.get_coords(),
            dtype=np.float64,
        )

        requested_total = abs(
            float(target_value) - float(baseline[axis_index])
        )

        print()
        print(
            f"슬롯 1 {axis_name} 자동 단일축 보정 시작 | "
            f"현재 {baseline[axis_index]:.3f} → "
            f"목표 {float(target_value):.3f} mm"
        )

        if requested_total > float(max_total_mm):
            print(
                f"슬롯 1 {axis_name} 자동보정 차단: "
                f"필요 이동량 {requested_total:.2f}mm가 "
                f"최대 {float(max_total_mm):.2f}mm를 초과합니다."
            )
            return baseline.tolist(), False

        latest = baseline.copy()

        for step_index in range(1, int(max_steps) + 1):
            current = np.asarray(
                self.client.get_coords(),
                dtype=np.float64,
            )
            remaining = float(
                float(target_value) - current[axis_index]
            )

            if abs(remaining) <= float(tolerance_mm):
                print(
                    f"슬롯 1 {axis_name} 자동보정 완료: "
                    f"남은 오차 {remaining:.3f}mm"
                )
                return current.tolist(), True

            step_mm = float(
                np.clip(
                    remaining,
                    -SLOT1_AUTO_FINAL_STEP_MM,
                    SLOT1_AUTO_FINAL_STEP_MM,
                )
            )
            command_value = float(
                current[axis_index] + step_mm
            )

            print(
                f"[{step_index}/{max_steps}] "
                f"{axis_name} {current[axis_index]:.3f} → "
                f"{command_value:.3f} mm"
            )

            response = runner(
                lambda command_value=command_value: (
                    self.client.move_single_coord(
                        axis_id=axis_id,
                        value=command_value,
                        speed=SLOT1_AUTO_FINAL_SPEED,
                    )
                ),
                (
                    f"slot 1 auto final {axis_name} "
                    f"{step_index}/{max_steps}"
                ),
            )

            print(
                f"슬롯 1 {axis_name} 단일축 결과:",
                response,
            )

            if not response.get("ok"):
                print(
                    f"슬롯 1 {axis_name} 자동보정 중단: "
                    "MOVE_SINGLE_COORD 실패"
                )
                return current.tolist(), False

            returned = response.get("final_coords_mm_deg")
            if returned is None:
                returned = self.client.get_coords()

            latest = np.asarray(
                returned,
                dtype=np.float64,
            )

            actual_delta = float(
                latest[axis_index] - current[axis_index]
            )

            if actual_delta * step_mm <= 0.0:
                print(
                    f"슬롯 1 {axis_name} 자동보정 중단: "
                    f"명령 {step_mm:+.3f}mm에 실제 "
                    f"{actual_delta:+.3f}mm로 반대/무응답 이동"
                )
                return latest.tolist(), False

            if abs(actual_delta) < SLOT1_AUTO_FINAL_MIN_PROGRESS_MM:
                print(
                    f"슬롯 1 {axis_name} 자동보정 중단: "
                    f"실제 이동량 {actual_delta:+.3f}mm가 너무 작습니다."
                )
                return latest.tolist(), False

            total_x_drift = abs(
                float(latest[0] - baseline[0])
            )
            total_rotation_drift = max(
                wrapped_angle_error_deg(
                    float(latest[index]),
                    float(baseline[index]),
                )
                for index in range(3, 6)
            )

            if (
                total_x_drift > SLOT1_AUTO_FINAL_MAX_X_DRIFT_MM
                or total_rotation_drift
                > SLOT1_AUTO_FINAL_MAX_ROTATION_DRIFT_DEG
            ):
                print(
                    f"슬롯 1 {axis_name} 자동보정 중단: "
                    "X 또는 손목 자세 drift가 안전 기준을 초과했습니다."
                )
                print(
                    "누적 drift | X:",
                    f"{total_x_drift:.2f}mm | 회전:",
                    f"{total_rotation_drift:.2f}deg",
                )
                try:
                    print("STOP:", self.client.stop())
                except Exception:
                    pass
                return latest.tolist(), False

        final_coords = [
            float(value)
            for value in self.client.get_coords()
        ]
        final_remaining = float(
            float(target_value) - float(final_coords[axis_index])
        )

        print(
            f"슬롯 1 {axis_name} 자동보정 최대 단계 종료 | "
            f"남은 오차 {final_remaining:.3f}mm"
        )

        return (
            final_coords,
            abs(final_remaining) <= float(tolerance_mm),
        )

    def _auto_single_y_settle_slot2(
        self,
        reference,
        runner,
    ):
        """
        Slot 2 grasp coarse 이동 후 목표 Y 방향(+Y)으로만 최대 1 mm씩
        미세전진한다. 목표보다 이미 앞쪽이면 자동 후진하지 않는다.
        """
        reference = np.asarray(
            reference,
            dtype=np.float64,
        )

        start_coords = [
            float(value)
            for value in self.client.get_coords()
        ]
        start_array = np.asarray(
            start_coords,
            dtype=np.float64,
        )
        initial_remaining_y = float(
            reference[1] - start_array[1]
        )

        print()
        print("=" * 70)
        print("슬롯 2 단일 Y축 미세전진 시작")
        print(
            "실제 집기 목표 Y:",
            f"{reference[1]:.3f} mm",
        )
        print(
            "현재 Y:",
            f"{start_array[1]:.3f} mm",
        )
        print(
            "남은 +Y 거리:",
            f"{initial_remaining_y:.3f} mm",
        )
        print(
            "한 번에 최대:",
            f"{SLOT2_SINGLE_Y_STEP_MM:.1f} mm",
        )
        print("=" * 70)

        if (
            0.0 <= initial_remaining_y
            <= SLOT2_SINGLE_Y_TOLERANCE_MM
        ):
            print(
                "이미 목표 Y 허용오차 안입니다. "
                "추가 이동하지 않습니다."
            )
            return start_coords, True

        if initial_remaining_y < 0.0:
            print(
                "현재 Y가 목표보다 이미 앞쪽입니다. "
                "자동으로 뒤로 이동하지 않습니다."
            )
            return start_coords, False

        if initial_remaining_y > SLOT2_SINGLE_Y_MAX_TOTAL_MM:
            print(
                "미세전진 차단: "
                f"남은 Y 거리가 {initial_remaining_y:.2f}mm로 "
                f"허용값 {SLOT2_SINGLE_Y_MAX_TOTAL_MM:.1f}mm를 "
                "초과했습니다."
            )
            return start_coords, False

        baseline = start_array.copy()
        final_coords = start_coords
        correction_ok = True

        for step_index in range(
            1,
            SLOT2_SINGLE_Y_MAX_STEPS + 1,
        ):
            current_coords = [
                float(value)
                for value in self.client.get_coords()
            ]
            current_array = np.asarray(
                current_coords,
                dtype=np.float64,
            )
            remaining_y = float(
                reference[1] - current_array[1]
            )

            if remaining_y <= SLOT2_SINGLE_Y_TOLERANCE_MM:
                final_coords = current_coords
                print(
                    "목표 Y 허용오차 도달:",
                    f"남은 {remaining_y:.3f}mm",
                )
                break

            step_mm = min(
                SLOT2_SINGLE_Y_STEP_MM,
                remaining_y,
            )
            command_y = float(
                current_array[1] + step_mm
            )

            print()
            print(
                f"[{step_index}/{SLOT2_SINGLE_Y_MAX_STEPS}] "
                "Y축 미세전진 요청"
            )
            print(
                "현재 Y → 명령 Y:",
                f"{current_array[1]:.3f} → "
                f"{command_y:.3f} mm",
            )

            response = runner(
                lambda command_y=command_y: (
                    self.client.move_single_coord(
                        axis_id=2,
                        value=command_y,
                        speed=SLOT2_SINGLE_Y_SPEED,
                    )
                ),
                (
                    f"slot 2 fine Y "
                    f"{step_index}/{SLOT2_SINGLE_Y_MAX_STEPS}"
                ),
            )

            print(
                "단일 Y축 이동 결과:",
                response,
            )

            if not response.get("ok"):
                print(
                    "Y축 미세전진 실패. 즉시 중단하고 "
                    "집게를 닫지 마세요."
                )
                correction_ok = False
                break

            returned = response.get("final_coords_mm_deg")
            if returned is None:
                returned = self.client.get_coords()

            final_coords = [
                float(value)
                for value in returned
            ]
            final_array = np.asarray(
                final_coords,
                dtype=np.float64,
            )

            total_x_drift = abs(
                float(final_array[0] - baseline[0])
            )
            total_z_drift = abs(
                float(final_array[2] - baseline[2])
            )
            total_rotation_drift = max(
                wrapped_angle_error_deg(
                    float(final_array[index]),
                    float(baseline[index]),
                )
                for index in range(3, 6)
            )

            if (
                total_x_drift
                > SLOT2_SINGLE_Y_MAX_TOTAL_X_DRIFT_MM
                or total_z_drift
                > SLOT2_SINGLE_Y_MAX_TOTAL_Z_DRIFT_MM
                or total_rotation_drift
                > SLOT2_SINGLE_Y_MAX_TOTAL_ROTATION_DRIFT_DEG
            ):
                print(
                    "누적 X/Z 또는 손목 자세 변화가 커서 "
                    "정지합니다."
                )
                print(
                    "누적 drift | X:",
                    f"{total_x_drift:.2f}mm | Z:",
                    f"{total_z_drift:.2f}mm | 회전:",
                    f"{total_rotation_drift:.2f}deg",
                )
                try:
                    print("STOP:", self.client.stop())
                except Exception:
                    pass
                correction_ok = False
                break

        final_coords = [
            float(value)
            for value in self.client.get_coords()
        ]
        final_remaining_y = float(
            reference[1] - float(final_coords[1])
        )

        print(
            "슬롯 2 최종 남은 Y 오차:",
            f"{final_remaining_y:.3f}mm",
        )

        if abs(final_remaining_y) > SLOT2_SINGLE_Y_TOLERANCE_MM:
            correction_ok = False

        return final_coords, correction_ok

    def _validate(self, coords):
        return validate_robot_coords(
            [float(value) for value in coords],
            self.position_limits_mm,
            self.rotation_limits_deg,
        )

    def coords_from_transforms(self, transforms):
        return {
            stage: self._validate(
                transform_to_coords(transform)
            )
            for stage, transform in transforms.items()
        }

    def targets_from_marker(
        self,
        registration,
        base_marker,
        slot,
    ):
        transforms = registration.targets_from_marker(
            base_marker,
            int(slot),
        )
        return self.coords_from_transforms(transforms)

    # ---------------------------------------------------------------------
    # p : approach preview
    # ---------------------------------------------------------------------

    def prepare_approach(
        self,
        registration,
        base_marker,
        slot,
    ):
        slot = int(slot)
        targets = self.targets_from_marker(
            registration,
            base_marker,
            slot,
        )

        reference = [
            float(value)
            for value in targets["approach"]
        ]

        compensation = APPROACH_POSITION_COMPENSATION_MM[
            slot
        ]

        for axis_index in range(3):
            reference[axis_index] += float(
                compensation[axis_index]
            )

        reference = self._validate(reference)

        return {
            "slot": slot,
            "reference": list(reference),
            "command": list(reference),
            "compensation_xyz_mm": compensation.tolist(),
        }

    # ---------------------------------------------------------------------
    # x : exact waypoint construction from the working Python test
    # ---------------------------------------------------------------------

    def approach_waypoints(
        self,
        current,
        target,
        slot,
    ):
        current = [float(value) for value in current]
        target = [float(value) for value in target]
        slot = int(slot)

        distance_to_target_mm = float(
            np.linalg.norm(
                np.asarray(
                    current[:3],
                    dtype=np.float64,
                )
                - np.asarray(
                    target[:3],
                    dtype=np.float64,
                )
            )
        )

        if (
            distance_to_target_mm
            <= DIRECT_APPROACH_MAX_DISTANCE_MM
        ):
            return [
                (
                    "가까운 접근 자세 직접 이동",
                    target,
                ),
            ]

        # Kept for parity with the original file even though the <=100 branch
        # currently catches this slot-2 <=60 special case first.
        if (
            slot == 2
            and distance_to_target_mm <= 60.0
        ):
            return [
                (
                    "저장된 접근 자세 재시도",
                    target,
                ),
            ]

        if slot == 2:
            transition_z = target[2] + 30.0
        else:
            transition_z = max(
                target[2] + SAFE_Z_MARGIN_MM,
                min(
                    current[2],
                    target[2] + 60.0,
                ),
            )

        midpoint = [
            (current[0] + target[0]) / 2.0,
            (current[1] + target[1]) / 2.0,
            current[2],
            current[3],
            current[4],
            current[5],
        ]

        lower_midpoint = [
            midpoint[0],
            midpoint[1],
            transition_z,
            current[3],
            current[4],
            current[5],
        ]

        xy_safe_point = [
            target[0],
            target[1],
            transition_z,
            current[3],
            current[4],
            current[5],
        ]

        if slot == 2:
            return [
                ("중간 안전점", midpoint),
                ("중간점 높이 변경", lower_midpoint),
                ("목표 XY 안전점", xy_safe_point),
                ("저장된 접근 자세", target),
            ]

        if slot == 3:
            split_25 = [
                current[0]
                + (target[0] - current[0]) * 0.25,
                current[1]
                + (target[1] - current[1]) * 0.25,
                current[2],
                current[3],
                current[4],
                current[5],
            ]
            split_50 = [
                current[0]
                + (target[0] - current[0]) * 0.50,
                current[1]
                + (target[1] - current[1]) * 0.50,
                current[2],
                current[3],
                current[4],
                current[5],
            ]
            split_75 = [
                current[0]
                + (target[0] - current[0]) * 0.75,
                current[1]
                + (target[1] - current[1]) * 0.75,
                current[2],
                current[3],
                current[4],
                current[5],
            ]

            return [
                ("분할 안전점 25%", split_25),
                ("분할 안전점 50%", split_50),
                ("분할 안전점 75%", split_75),
                ("목표 XY 안전점", xy_safe_point),
                ("저장된 접근 자세", target),
            ]

        return [
            ("중간 안전점", midpoint),
            ("목표 XY 안전점", xy_safe_point),
            ("저장된 접근 자세", target),
        ]

    def move_approach(
        self,
        preview,
        runner,
    ):
        if not self.safety.allow_robot_motion:
            raise RuntimeError(
                "allow_robot_motion=false: "
                "x 접근 이동이 차단되어 있습니다."
            )

        slot = int(preview["slot"])
        target = list(preview["command"])
        reference = list(preview["reference"])
        current = self.client.get_coords()

        waypoints = self.approach_waypoints(
            current,
            target,
            slot,
        )

        print()
        print("=" * 66)
        print(
            f"슬롯 {slot} 단계 이동 시작 "
            f"(속도 {TEST_SPEED})"
        )
        for name, coords in waypoints:
            print(
                name,
                ":",
                np.round(coords, 3).tolist(),
            )
        print("=" * 66)

        total_steps = len(waypoints)

        for step_index, (name, coords) in enumerate(
            waypoints,
            start=1,
        ):
            command_coords = [
                float(value)
                for value in coords
            ]

            # Original slot-3 behavior: maintain the live wrist RPY at every
            # intermediate point, and apply registered RPY only at final
            # approach.
            if (
                slot == 3
                and name != "저장된 접근 자세"
            ):
                live_coords = self.client.get_coords()
                command_coords[3:] = live_coords[3:]

                print(
                    "현재 손목 자세 유지:",
                    np.round(
                        command_coords[3:],
                        3,
                    ).tolist(),
                )

            print()
            print(
                f"[{step_index}/{total_steps}] "
                f"{name} 이동 요청"
            )

            response = runner(
                lambda coords=command_coords: (
                    self.client.move_coords(
                        coords,
                        speed=TEST_SPEED,
                        mode=MOVE_MODE,
                        position_tolerance_mm=(
                            COARSE_MOVE_POSITION_TOLERANCE_MM
                        ),
                        rotation_tolerance_deg=(
                            COARSE_MOVE_ROTATION_TOLERANCE_DEG
                        ),
                        stable_required=(
                            COARSE_MOVE_STABLE_REQUIRED
                        ),
                        timeout_sec=(
                            COARSE_MOVE_TIMEOUT_SEC
                        ),
                    )
                ),
                f"{step_index}/{total_steps} {name}",
            )

            print(
                f"[{step_index}/{total_steps}] 결과:",
                response,
            )

            slot3_position_only_accept = False

            if (
                slot == 3
                and name != "저장된 접근 자세"
                and not response.get("ok")
                and response.get("error")
                == (
                    "목표 자세에 도달하지 못했습니다. "
                    "다음 단계 이동을 차단합니다."
                )
            ):
                position_error = response.get(
                    "final_position_error_mm"
                )
                rotation_error = response.get(
                    "final_rotation_error_deg"
                )

                if (
                    position_error is not None
                    and rotation_error is not None
                    and float(position_error)
                    <= SLOT3_INTERMEDIATE_POSITION_ACCEPT_MM
                    and float(rotation_error)
                    <= SLOT3_INTERMEDIATE_ROTATION_ACCEPT_DEG
                ):
                    slot3_position_only_accept = True

                    print(
                        "슬롯 3 안전 중간점 위치우선 통과:",
                        f"위치오차 {float(position_error):.2f}mm,",
                        f"회전오차 {float(rotation_error):.2f}deg",
                    )

            if (
                not response.get("ok")
                and not slot3_position_only_accept
            ):
                raise RuntimeError(
                    response.get(
                        "error",
                        "접근 단계 이동 실패",
                    )
                )

            time.sleep(INTER_WAYPOINT_SETTLE_SEC)

        print()
        print("슬롯 접근 coarse 이동 완료")

        settle_current = self.client.get_coords()
        settle_pos_error, settle_rot_error = pose_error(
            settle_current,
            reference,
        )

        print(
            "원래 접근 기준 coarse 오차:",
            f"{settle_pos_error:.2f}mm,",
            f"{settle_rot_error:.2f}deg",
        )

        result = {
            "final_coords": settle_current,
            "coarse_position_error_mm": settle_pos_error,
            "coarse_rotation_error_deg": settle_rot_error,
            "fine_corrected": False,
        }

        should_settle = (
            FINE_SETTLE_MIN_POSITION_MM
            < settle_pos_error
            <= FINE_SETTLE_MAX_POSITION_MM
            and settle_rot_error
            <= FINE_SETTLE_MAX_ROTATION_DEG
        )

        if should_settle:
            actual_array = np.asarray(
                settle_current,
                dtype=np.float64,
            )
            reference_array = np.asarray(
                reference,
                dtype=np.float64,
            )
            previous_command = np.asarray(
                target,
                dtype=np.float64,
            )

            correction_xyz = np.clip(
                (
                    reference_array[:3]
                    - actual_array[:3]
                )
                * FINE_SETTLE_GAIN,
                -FINE_SETTLE_MAX_AXIS_MM,
                FINE_SETTLE_MAX_AXIS_MM,
            )

            if slot == 1:
                correction_xyz[2] = 0.0

            fine_target = previous_command.copy()
            fine_target[:3] += correction_xyz
            fine_target[3:] = reference_array[3:]

            print(
                "approach 남은 오차벡터 XYZ(mm):",
                np.round(
                    reference_array[:3]
                    - actual_array[:3],
                    3,
                ).tolist(),
            )
            print(
                "approach 미세 명령 XYZ 보정:",
                np.round(
                    correction_xyz,
                    3,
                ).tolist(),
            )

            settle_response = runner(
                lambda: self.client.move_coords(
                    fine_target.tolist(),
                    speed=FINE_SETTLE_SPEED,
                    mode=MOVE_MODE,
                    position_tolerance_mm=(
                        FINE_MOVE_POSITION_TOLERANCE_MM
                    ),
                    rotation_tolerance_deg=(
                        FINE_MOVE_ROTATION_TOLERANCE_DEG
                    ),
                    stable_required=(
                        FINE_MOVE_STABLE_REQUIRED
                    ),
                    timeout_sec=(
                        FINE_MOVE_TIMEOUT_SEC
                    ),
                ),
                "approach residual correction",
            )

            print(
                "approach 오차벡터 보정 결과:",
                settle_response,
            )

            final_coords = (
                settle_response.get(
                    "final_coords_mm_deg"
                )
                or self.client.get_coords()
            )

            final_pos_error, final_rot_error = pose_error(
                final_coords,
                reference,
            )

            result.update(
                {
                    "final_coords": final_coords,
                    "final_position_error_mm": final_pos_error,
                    "final_rotation_error_deg": final_rot_error,
                    "fine_corrected": True,
                }
            )

        return result

    # ---------------------------------------------------------------------
    # g : grasp preview
    # ---------------------------------------------------------------------

    def prepare_grasp(
        self,
        registration,
        base_marker,
        slot,
        current,
    ):
        slot = int(slot)
        targets = self.targets_from_marker(
            registration,
            base_marker,
            slot,
        )

        approach_reference = [
            float(value)
            for value in targets["approach"]
        ]

        for axis_index in range(3):
            approach_reference[axis_index] += float(
                APPROACH_POSITION_COMPENSATION_MM[
                    slot
                ][axis_index]
            )

        grasp_reference = [
            float(value)
            for value in targets["grasp"]
        ]

        grasp_desired_reference = list(grasp_reference)
        grasp_desired_reference[2] += float(
            GRASP_PHYSICAL_DEPTH_OFFSET_MM[
                slot
            ]
        )

        grasp_compensation = (
            GRASP_POSITION_COMPENSATION_MM[slot]
        )

        for axis_index in range(3):
            grasp_desired_reference[
                axis_index
            ] += float(
                grasp_compensation[axis_index]
            )

        grasp_command = list(
            grasp_desired_reference
        )

        rotation_compensation = (
            GRASP_ROTATION_COMPENSATION_DEG[
                slot
            ]
        )

        for axis_index in range(3):
            angle = (
                grasp_command[3 + axis_index]
                + float(
                    rotation_compensation[
                        axis_index
                    ]
                )
            )
            grasp_command[
                3 + axis_index
            ] = (
                angle + 180.0
            ) % 360.0 - 180.0

        approach_position_error, approach_rotation_error = pose_error(
            current,
            approach_reference,
        )
        grasp_step_mm, grasp_rotation_step_deg = pose_error(
            current,
            grasp_command,
        )

        safe = True
        reasons = []

        if (
            approach_position_error
            > MAX_APPROACH_POSITION_ERROR_MM
            or approach_rotation_error
            > MAX_APPROACH_ROTATION_ERROR_DEG
        ):
            safe = False
            reasons.append(
                "현재 로봇이 저장된 접근 자세 근처에 있지 않습니다."
            )

        if (
            grasp_step_mm
            > MAX_GRASP_TRANSLATION_STEP_MM
        ):
            safe = False
            reasons.append(
                "grasp까지 이동거리가 안전 기준을 초과합니다."
            )

        if (
            grasp_rotation_step_deg
            > MAX_GRASP_ROTATION_STEP_DEG
        ):
            safe = False
            reasons.append(
                "grasp까지 회전 변화가 안전 기준을 초과합니다."
            )

        return {
            "slot": slot,
            "safe": safe,
            "reasons": reasons,
            "approach_reference": self._validate(
                approach_reference
            ),
            "grasp_reference": self._validate(
                grasp_desired_reference
            ),
            "grasp_command": self._validate(
                grasp_command
            ),
            "approach_position_error_mm": (
                approach_position_error
            ),
            "approach_rotation_error_deg": (
                approach_rotation_error
            ),
            "grasp_step_mm": grasp_step_mm,
            "grasp_rotation_step_deg": (
                grasp_rotation_step_deg
            ),
            "grasp_compensation_xyz_mm": (
                grasp_compensation.tolist()
            ),
            "grasp_rotation_compensation_deg": (
                rotation_compensation.tolist()
            ),
        }

    # ---------------------------------------------------------------------
    # z : grasp move
    # ---------------------------------------------------------------------

    def move_grasp(
        self,
        preview,
        runner,
    ):
        if not self.safety.allow_robot_motion:
            raise RuntimeError(
                "allow_robot_motion=false: "
                "z grasp 이동이 차단되어 있습니다."
            )

        if not preview.get("safe"):
            raise RuntimeError(
                "grasp preview safety check failed: "
                f"{preview.get('reasons')}"
            )

        slot = int(preview["slot"])
        current = self.client.get_coords()

        approach_position_error, approach_rotation_error = pose_error(
            current,
            preview["approach_reference"],
        )
        grasp_step_mm, grasp_rotation_step_deg = pose_error(
            current,
            preview["grasp_command"],
        )

        if (
            approach_position_error
            > MAX_APPROACH_POSITION_ERROR_MM
            or approach_rotation_error
            > MAX_APPROACH_ROTATION_ERROR_DEG
        ):
            raise RuntimeError(
                "집기 이동 차단: 현재 자세가 "
                "접근 자세에서 벗어났습니다."
            )

        if (
            grasp_step_mm
            > MAX_GRASP_TRANSLATION_STEP_MM
            or grasp_rotation_step_deg
            > MAX_GRASP_ROTATION_STEP_DEG
        ):
            raise RuntimeError(
                "집기 이동 차단: 이동량 안전 기준을 초과했습니다."
            )

        response = runner(
            lambda: self.client.move_coords(
                preview["grasp_command"],
                speed=GRASP_TEST_SPEED,
                mode=MOVE_MODE,
                position_tolerance_mm=(
                    COARSE_MOVE_POSITION_TOLERANCE_MM
                ),
                rotation_tolerance_deg=(
                    COARSE_MOVE_ROTATION_TOLERANCE_DEG
                ),
                stable_required=(
                    COARSE_MOVE_STABLE_REQUIRED
                ),
                timeout_sec=(
                    COARSE_MOVE_TIMEOUT_SEC
                ),
            ),
            f"slot {slot} grasp move",
        )

        print(
            "집기 자세 이동 결과:",
            response,
        )

        if not response.get("ok"):
            raise RuntimeError(
                response.get(
                    "error",
                    "집기 자세 이동 실패",
                )
            )

        final_coords = (
            response.get("final_coords_mm_deg")
            or self.client.get_coords()
        )

        reference_position_error, reference_rotation_error = pose_error(
            final_coords,
            preview["grasp_reference"],
        )

        print(
            "실제 집기 깊이 목표 기준 coarse 위치오차:",
            f"{reference_position_error:.2f} mm",
        )
        print(
            "실제 집기 깊이 목표 기준 coarse 회전오차:",
            f"{reference_rotation_error:.2f} deg",
        )

        result = {
            "final_coords": final_coords,
            "position_error_mm": reference_position_error,
            "rotation_error_deg": reference_rotation_error,
            "fine_corrected": False,
        }

        if slot == 1:
            # 원본 왼팔 로직: coarse grasp 후 Y -> Z 순서로
            # 단일축 1 mm 자동 마무리 보정을 수행한다.
            slot1_y_coords, slot1_y_ok = (
                self._auto_single_coord_settle_slot1(
                    axis_id=2,
                    axis_name="Y",
                    target_value=preview["grasp_reference"][1],
                    tolerance_mm=SLOT1_AUTO_FINAL_Y_TOLERANCE_MM,
                    max_total_mm=SLOT1_AUTO_FINAL_MAX_Y_MM,
                    max_steps=SLOT1_AUTO_FINAL_MAX_Y_STEPS,
                    runner=runner,
                )
            )

            if not slot1_y_ok:
                raise RuntimeError(
                    "슬롯 1 Y 자동 마무리 보정에 실패했습니다. "
                    "집게를 닫지 마세요."
                )

            slot1_z_coords, slot1_z_ok = (
                self._auto_single_coord_settle_slot1(
                    axis_id=3,
                    axis_name="Z",
                    target_value=preview["grasp_reference"][2],
                    tolerance_mm=SLOT1_AUTO_FINAL_Z_TOLERANCE_MM,
                    max_total_mm=SLOT1_AUTO_FINAL_MAX_Z_MM,
                    max_steps=SLOT1_AUTO_FINAL_MAX_Z_STEPS,
                    runner=runner,
                )
            )

            if not slot1_z_ok:
                raise RuntimeError(
                    "슬롯 1 Z 자동 마무리 보정에 실패했습니다. "
                    "집게를 닫지 마세요."
                )

            final_coords = slot1_z_coords
            (
                reference_position_error,
                reference_rotation_error,
            ) = pose_error(
                final_coords,
                preview["grasp_reference"],
            )

            result.update(
                {
                    "final_coords": final_coords,
                    "position_error_mm": reference_position_error,
                    "rotation_error_deg": reference_rotation_error,
                    "fine_corrected": True,
                    "slot1_y_corrected": True,
                    "slot1_z_corrected": True,
                }
            )

        elif slot == 2:
            # 원본 왼팔 로직: 목표 Y보다 뒤에 있을 때만
            # +Y 방향으로 1 mm씩 전진한다. 자동 후진은 하지 않는다.
            slot2_coords, slot2_ok = (
                self._auto_single_y_settle_slot2(
                    preview["grasp_reference"],
                    runner,
                )
            )

            if not slot2_ok:
                raise RuntimeError(
                    "슬롯 2 Y 미세전진 보정에 실패했습니다. "
                    "집게를 닫지 마세요."
                )

            final_coords = slot2_coords
            (
                reference_position_error,
                reference_rotation_error,
            ) = pose_error(
                final_coords,
                preview["grasp_reference"],
            )

            result.update(
                {
                    "final_coords": final_coords,
                    "position_error_mm": reference_position_error,
                    "rotation_error_deg": reference_rotation_error,
                    "fine_corrected": True,
                    "slot2_y_corrected": True,
                }
            )

        elif (
            slot == 3
            and reference_position_error
            > FINE_SETTLE_MIN_POSITION_MM
        ):
            print(
                "슬롯 3은 X·Y 이동 시 Z가 함께 변하는 "
                "자세 결합을 방지하기 위해 집기 자동 미세보정을 "
                "생략합니다."
            )

        should_settle = (
            slot not in (1, 2, 3)
            and FINE_SETTLE_MIN_POSITION_MM
            < reference_position_error
            <= FINE_SETTLE_MAX_POSITION_MM
            and reference_rotation_error
            <= FINE_SETTLE_MAX_ROTATION_DEG
        )

        if should_settle:
            actual_array = np.asarray(
                final_coords,
                dtype=np.float64,
            )
            reference_array = np.asarray(
                preview["grasp_reference"],
                dtype=np.float64,
            )
            previous_command = np.asarray(
                preview["grasp_command"],
                dtype=np.float64,
            )

            correction_xyz = np.clip(
                (
                    reference_array[:3]
                    - actual_array[:3]
                )
                * FINE_SETTLE_GAIN,
                -FINE_SETTLE_MAX_AXIS_MM,
                FINE_SETTLE_MAX_AXIS_MM,
            )

            if slot == 4:
                original_y_correction = float(
                    correction_xyz[1]
                )
                correction_xyz[1] *= (
                    SLOT4_GRASP_Y_FINE_GAIN
                )

                print(
                    "슬롯 4 grasp Y 미세보정 축소:",
                    f"{original_y_correction:.3f} → "
                    f"{float(correction_xyz[1]):.3f}mm",
                )

            fine_target = previous_command.copy()
            fine_target[:3] += correction_xyz

            # Keep compensated wrist orientation unchanged.
            fine_target[3:] = previous_command[3:]

            settle_response = runner(
                lambda: self.client.move_coords(
                    fine_target.tolist(),
                    speed=FINE_SETTLE_SPEED,
                    mode=MOVE_MODE,
                    position_tolerance_mm=(
                        FINE_MOVE_POSITION_TOLERANCE_MM
                    ),
                    rotation_tolerance_deg=(
                        FINE_MOVE_ROTATION_TOLERANCE_DEG
                    ),
                    stable_required=(
                        FINE_MOVE_STABLE_REQUIRED
                    ),
                    timeout_sec=(
                        FINE_MOVE_TIMEOUT_SEC
                    ),
                ),
                "grasp residual correction",
            )

            print(
                "grasp 오차벡터 보정 결과:",
                settle_response,
            )

            settle_final = (
                settle_response.get(
                    "final_coords_mm_deg"
                )
                or self.client.get_coords()
            )

            reference_position_error, reference_rotation_error = pose_error(
                settle_final,
                preview["grasp_reference"],
            )

            result.update(
                {
                    "final_coords": settle_final,
                    "position_error_mm": (
                        reference_position_error
                    ),
                    "rotation_error_deg": (
                        reference_rotation_error
                    ),
                    "fine_corrected": True,
                }
            )

        return result

    # ---------------------------------------------------------------------
    # Existing ROS service compatibility.
    # Keyboard test remains the validated path for now.
    # ---------------------------------------------------------------------

    def execute(self, targets, slot):
        required = {
            "approach",
            "grasp",
            "place",
        }

        if not required.issubset(targets):
            raise ValueError(
                "missing pick stages: "
                f"{sorted(required - set(targets))}"
            )

        targets = {
            stage: self._validate(
                targets[stage]
            )
            for stage in (
                "approach",
                "grasp",
                "place",
            )
        }

        if (
            self.safety.preview_only
            or not self.safety.allow_robot_motion
        ):
            return {
                "status": "preview",
                "slot": int(slot),
                "targets": targets,
            }

        approach_result = self.client.move_coords(
            targets["approach"],
            speed=TEST_SPEED,
            mode=MOVE_MODE,
            position_tolerance_mm=(
                COARSE_MOVE_POSITION_TOLERANCE_MM
            ),
            rotation_tolerance_deg=(
                COARSE_MOVE_ROTATION_TOLERANCE_DEG
            ),
            stable_required=(
                COARSE_MOVE_STABLE_REQUIRED
            ),
            timeout_sec=(
                COARSE_MOVE_TIMEOUT_SEC
            ),
        )

        if not approach_result.get("ok"):
            raise RuntimeError(
                "service approach target was not reached"
            )

        grasp_result = self.client.move_coords(
            targets["grasp"],
            speed=GRASP_TEST_SPEED,
            mode=MOVE_MODE,
            position_tolerance_mm=(
                COARSE_MOVE_POSITION_TOLERANCE_MM
            ),
            rotation_tolerance_deg=(
                COARSE_MOVE_ROTATION_TOLERANCE_DEG
            ),
            stable_required=(
                COARSE_MOVE_STABLE_REQUIRED
            ),
            timeout_sec=(
                COARSE_MOVE_TIMEOUT_SEC
            ),
        )

        if not grasp_result.get("ok"):
            raise RuntimeError(
                "service grasp target was not reached"
            )

        if not self.safety.allow_gripper_close:
            raise RuntimeError(
                "gripper close is disabled; "
                "stopped before closing"
            )

        self.client.gripper_close()

        return {
            "status": "grasped",
            "slot": int(slot),
            "targets": targets,
        }