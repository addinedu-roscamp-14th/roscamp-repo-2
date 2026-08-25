"""ROS2 + OpenCV keyboard test node for the restored right-arm ArUco workflow."""
from __future__ import annotations

import json
import logging
import threading
import time

import cv2
import numpy as np

from .aruco_detector import (
    RightArmArucoCamera,
    rotation_difference_deg,
)
from .coordinate_transform import CoordinateTransformer
from .pick_sequence import PickSafety, PickSequence
from .robot_client import RightArmRobotClient
from .slot_registration import SlotRegistration


# r is now a Raspberry-Pi fixed motion (OBSERVE_ARUCO), but the reference
# angles are kept only for the operator log so the GUI still shows what pose
# the old validated Python program used.
OBSERVATION_ANGLES = [
    102.48,
    49.92,
    -73.56,
    -30.05,
    -8.08,
    -31.11,
]

OBSERVATION_SETTLE_SEC = 1.0
OBSERVATION_RECAPTURE_TIMEOUT_SEC = 10.0

MAX_OBSERVATION_MARKER_SHIFT_MM = 8.0
MAX_OBSERVATION_MARKER_ROTATION_DEG = 3.0
MAX_OBSERVATION_FALLBACK_SHIFT_MM = 30.0
MAX_OBSERVATION_FALLBACK_ROTATION_DEG = 10.0

GRASP_PREVIEW_VALID_SEC = 120.0
APPROACH_PREVIEW_VALID_SEC = 120.0


class RightArmTransportNode:
    def __init__(self, ros_node):
        self.node = ros_node
        p = ros_node.get_parameter
        self.log = logging.getLogger(
            "right_arm_transport_pick"
        )

        self.config = {
            "camera_width": p(
                "camera_width"
            ).value,
            "camera_height": p(
                "camera_height"
            ).value,
            "camera_fps": p(
                "camera_fps"
            ).value,
            "marker_length_m": p(
                "marker_length_m"
            ).value,
            "target_id": p(
                "target_id"
            ).value,
            "dictionary_id": p(
                "dictionary_id"
            ).value,
            "stable_buffer_size": p(
                "stable_buffer_size"
            ).value,
            "min_stable_samples": p(
                "min_stable_samples"
            ).value,
            "max_translation_spread_mm": p(
                "max_translation_spread_mm"
            ).value,
            "max_rotation_spread_deg": p(
                "max_rotation_spread_deg"
            ).value,
            "max_reprojection_error_px": p(
                "max_reprojection_error_px"
            ).value,
            "lost_limit": p(
                "lost_limit"
            ).value,
        }

        self.preview_only = bool(
            p("preview_only").value
        )
        self.allow_motion = bool(
            p("allow_robot_motion").value
        )
        self.allow_close = bool(
            p("allow_gripper_close").value
        )
        self.slot_id = int(
            p("slot_id").value
        )

        self.registration = SlotRegistration(
            p("slot_registration_file").value
        )
        self.transformer = CoordinateTransformer(
            p("handeye_file").value
        )
        self.client = RightArmRobotClient(
            p("robot_host").value,
            p("robot_port").value,
            p("connect_timeout_sec").value,
            p("command_timeout_sec").value,
        )
        self.camera = RightArmArucoCamera(
            p("camera_device").value,
            p("camera_calibration_file").value,
            **self.config,
        )
        self.sequence = PickSequence(
            self.client,
            PickSafety(
                self.preview_only,
                self.allow_motion,
                self.allow_close,
            ),
            p("position_limits_mm").value,
            p("rotation_limits_deg").value,
            self.log,
        )

        self.current_base_marker = None
        self.base_marker_before_observation = None
        self.observation_recapture_pending = False
        self.observation_recapture_complete = False
        self.observation_recapture_not_before = 0.0
        self.observation_recapture_deadline = 0.0

        self.approach_preview = None
        self.approach_preview_timestamp = 0.0
        self.grasp_preview = None
        self.grasp_preview_timestamp = 0.0

        self.latest_camera = None
        self.latest_display = None
        self.exit_requested = False
        self.motion_lock = threading.Lock()

        from std_srvs.srv import Trigger

        self.service = ros_node.create_service(
            Trigger,
            "/right_arm/pick_transport_slot",
            self.pick,
        )

        self._print_help()

    def _print_help(self):
        print()
        print("=" * 72)
        print(
            "오른팔 ArUco 슬롯 접근/집기 "
            "원본 로직 복원 버전"
        )
        print(
            "시험 흐름: "
            "1~4 → m → r → p → x → g → z"
        )
        print("=" * 72)
        print("1~4 : 슬롯 선택")
        print("m   : 시작 자세에서 안정된 ArUco 기준 저장")
        print(
            "r   : OBSERVE_ARUCO 고정 자세 이동 "
            "→ 자동 재검출"
        )
        print("p   : 접근 목표 미리보기")
        print("x   : 원본 waypoint대로 접근 자세 이동")
        print("g   : 집기 목표 미리보기 + 안전 검사")
        print(
            "z   : 원본 조건으로 집기 자세까지 "
            "저속 이동 (그리퍼는 열림)"
        )
        print("s   : STOP")
        print("c   : 현재 robot coords")
        print("q   : 정상 종료")
        print("b/d/o/y : 현재 시험에서는 차단")
        print()
        print(
            f"preview_only={self.preview_only}"
        )
        print(
            f"allow_robot_motion={self.allow_motion}"
        )
        print(
            f"allow_gripper_close={self.allow_close}"
        )
        print("=" * 72)

    def _clear_previews(self):
        self.approach_preview = None
        self.approach_preview_timestamp = 0.0
        self.grasp_preview = None
        self.grasp_preview_timestamp = 0.0

    def _draw_status(
        self,
        camera_result,
    ):
        display = camera_result[
            "display"
        ].copy()
        stable = camera_result["stable"]
        current_error = camera_result[
            "current_error_px"
        ]

        if stable is not None:
            state_text = (
                "OBSERVE - recapturing"
                if self.observation_recapture_pending
                else "READY - press m"
            )
            state_color = (0, 255, 0)
        else:
            state_text = (
                f"stabilizing "
                f"{camera_result['stable_count']}/"
                f"{self.config['min_stable_samples']}"
            )
            state_color = (0, 165, 255)

        error_text = (
            f"{current_error:.3f}px"
            if current_error is not None
            else "NONE"
        )

        cv2.putText(
            display,
            state_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            state_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            (
                f"reproj: {error_text} | "
                f"slot: {self.slot_id}"
            ),
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            (
                "marker captured: "
                f"{self.current_base_marker is not None}"
            ),
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            (
                f"motion:{self.allow_motion} | "
                f"preview:{self.preview_only} | "
                "r-done:"
                f"{self.observation_recapture_complete}"
            ),
            (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            (
                "1-4 slot | m capture | r observe | "
                "p preview | x approach | g grasp | z down"
            ),
            (
                10,
                self.config["camera_height"] - 16,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        return display

    def _run_with_gui(
        self,
        function,
        status_text,
    ):
        if not self.motion_lock.acquire(
            blocking=False
        ):
            raise RuntimeError(
                "another robot command is already running"
            )

        result_box = {}
        error_box = {}

        def worker():
            try:
                result_box["result"] = function()
            except BaseException as error:
                error_box["error"] = error

        try:
            thread = threading.Thread(
                target=worker,
                daemon=True,
            )
            thread.start()

            animation = 0

            while thread.is_alive():
                base = (
                    self.latest_display.copy()
                    if self.latest_display
                    is not None
                    else np.zeros(
                        (
                            self.config[
                                "camera_height"
                            ],
                            self.config[
                                "camera_width"
                            ],
                            3,
                        ),
                        dtype=np.uint8,
                    )
                )

                dots = "." * (
                    animation % 4
                )
                animation += 1

                cv2.rectangle(
                    base,
                    (10, 125),
                    (
                        self.config[
                            "camera_width"
                        ]
                        - 10,
                        195,
                    ),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    base,
                    f"ROBOT MOVING{dots}",
                    (25, 153),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    base,
                    status_text,
                    (25, 182),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                self.camera.show(base)
                cv2.waitKey(50)

            thread.join()

            if "error" in error_box:
                raise error_box["error"]

            return result_box.get(
                "result"
            )

        finally:
            self.motion_lock.release()

    def _capture_base_marker(
        self,
        stable,
    ):
        robot_coords = (
            self.client.get_coords()
        )

        return self.transformer.base_marker(
            robot_coords,
            stable["transform"],
        )

    def _handle_observation_recapture(
        self,
        stable,
    ):
        if not self.observation_recapture_pending:
            return

        now = time.monotonic()

        if (
            now
            < self.observation_recapture_not_before
        ):
            return

        if stable is None:
            if (
                now
                >= self.observation_recapture_deadline
            ):
                print(
                    "r 재검출 시간 초과. "
                    "m부터 다시 실행하세요."
                )
                self.observation_recapture_pending = (
                    False
                )
                self.observation_recapture_complete = (
                    False
                )
                self.current_base_marker = None
                self.base_marker_before_observation = (
                    None
                )
                self.camera.reset()

            return

        candidate = self._capture_base_marker(
            stable
        )

        shift_mm = 0.0
        rotation_shift_deg = 0.0

        if (
            self.base_marker_before_observation
            is not None
        ):
            shift_mm = float(
                np.linalg.norm(
                    candidate[:3, 3]
                    - self.base_marker_before_observation[
                        :3,
                        3,
                    ]
                )
                * 1000.0
            )

            rotation_shift_deg = (
                rotation_difference_deg(
                    candidate[:3, :3],
                    self.base_marker_before_observation[
                        :3,
                        :3,
                    ],
                )
            )

        print()
        print(
            "중간 관측 자세 ArUco 재검출 완료"
        )
        print(
            "최초 검출 대비 위치 차이:",
            f"{shift_mm:.2f} mm",
        )
        print(
            "최초 검출 대비 회전 차이:",
            f"{rotation_shift_deg:.2f} deg",
        )

        if (
            shift_mm
            > MAX_OBSERVATION_FALLBACK_SHIFT_MM
            or rotation_shift_deg
            > MAX_OBSERVATION_FALLBACK_ROTATION_DEG
        ):
            print(
                "재검출 차이가 최대 안전 기준을 "
                "초과했습니다."
            )
            print(
                "트럭/로봇 상태 확인 후 "
                "m부터 다시 실행하세요."
            )

            self.current_base_marker = None
            self.observation_recapture_complete = (
                False
            )

        elif (
            shift_mm
            > MAX_OBSERVATION_MARKER_SHIFT_MM
            or rotation_shift_deg
            > MAX_OBSERVATION_MARKER_ROTATION_DEG
        ):
            print(
                "관측 자세별 핸드아이 편차로 판단하여 "
                "m 값을 복원합니다."
            )

            self.current_base_marker = (
                self.base_marker_before_observation.copy()
            )
            self.observation_recapture_complete = (
                True
            )

        else:
            self.current_base_marker = candidate
            self.observation_recapture_complete = (
                True
            )
            print(
                "r 재검출값을 새 base marker로 "
                "채택했습니다."
            )

        if self.current_base_marker is not None:
            print(
                "base marker XYZ(mm):",
                np.round(
                    self.current_base_marker[
                        :3,
                        3,
                    ]
                    * 1000.0,
                    3,
                ).tolist(),
            )

        self.observation_recapture_pending = (
            False
        )
        self.base_marker_before_observation = (
            None
        )
        self._clear_previews()
        self.camera.reset()

    def process_frame_and_keyboard(self):
        accept_stability = not (
            self.observation_recapture_pending
            and time.monotonic()
            < self.observation_recapture_not_before
        )

        camera_result = self.camera.read(
            accept_for_stability=accept_stability
        )

        self.latest_camera = camera_result

        self._handle_observation_recapture(
            camera_result["stable"]
        )

        display = self._draw_status(
            camera_result
        )

        self.latest_display = display
        self.camera.show(display)

        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            return

        if key in (
            ord("b"),
            ord("d"),
            ord("o"),
            ord("y"),
        ):
            print(
                "현재 시험에서는 b/d/o/y를 차단합니다. "
                "g 미리보기와 z 저속 이동까지만 사용하세요."
            )
            return

        if key in (
            ord("1"),
            ord("2"),
            ord("3"),
            ord("4"),
        ):
            self.slot_id = int(
                chr(key)
            )
            self._clear_previews()
            print(
                "선택 슬롯:",
                self.slot_id,
            )
            return

        if key == ord("m"):
            self._key_m()
        elif key == ord("r"):
            self._key_r()
        elif key == ord("p"):
            self._key_p()
        elif key == ord("x"):
            self._key_x()
        elif key == ord("g"):
            self._key_g()
        elif key == ord("z"):
            self._key_z()
        elif key == ord("s"):
            self._key_s()
        elif key == ord("c"):
            self._key_c()
        elif key == ord("q"):
            self.exit_requested = True

    def _key_m(self):
        if self.observation_recapture_pending:
            print(
                "r 재검출이 끝날 때까지 기다리세요."
            )
            return

        stable = (
            self.latest_camera["stable"]
            if self.latest_camera is not None
            else None
        )

        if stable is None:
            print(
                "READY 상태 이후 m을 누르세요."
            )
            return

        try:
            self.current_base_marker = (
                self._capture_base_marker(
                    stable
                )
            )
        except Exception as error:
            print(
                "m 실패:",
                error,
            )
            return

        self.base_marker_before_observation = None
        self.observation_recapture_pending = False
        self.observation_recapture_complete = False
        self._clear_previews()
        self.camera.reset()

        print()
        print(
            "현재 ArUco 기준 저장 완료"
        )
        print(
            "base marker XYZ(mm):",
            np.round(
                self.current_base_marker[
                    :3,
                    3,
                ]
                * 1000.0,
                3,
            ).tolist(),
        )
        print(
            "이제 트럭을 움직이지 말고 r을 누르세요."
        )

    def _key_r(self):
        if self.observation_recapture_pending:
            print(
                "이미 r 재검출 중입니다."
            )
            return

        if self.current_base_marker is None:
            print(
                "먼저 READY 상태에서 m을 누르세요."
            )
            return

        if not self.allow_motion:
            print(
                "allow_robot_motion=false: r 이동 차단"
            )
            return

        print(
            "중간 관측 자세 이동 시작:",
            OBSERVATION_ANGLES,
        )

        try:
            result = self._run_with_gui(
                lambda: self.client.observe_aruco(),
                "moving to ArUco observation pose",
            )
            print(
                "r 이동 결과:",
                result,
            )
        except Exception as error:
            print(
                "r 이동 실패:",
                error,
            )
            return

        self.base_marker_before_observation = (
            self.current_base_marker.copy()
        )
        self.current_base_marker = None
        self.observation_recapture_pending = True
        self.observation_recapture_complete = False
        self.observation_recapture_not_before = (
            time.monotonic()
            + OBSERVATION_SETTLE_SEC
        )
        self.observation_recapture_deadline = (
            self.observation_recapture_not_before
            + OBSERVATION_RECAPTURE_TIMEOUT_SEC
        )

        self._clear_previews()
        self.camera.reset()

        print(
            f"{OBSERVATION_SETTLE_SEC:.1f}초 정지 후 "
            "ArUco 여러 프레임을 자동 재수집합니다."
        )

    def _key_p(self):
        if self.observation_recapture_pending:
            print(
                "r 재검출 완료 후 p를 누르세요."
            )
            return

        if self.current_base_marker is None:
            print(
                "m → r을 먼저 완료하세요."
            )
            return

        if not self.observation_recapture_complete:
            print(
                "r 재검출 성공 후 p를 누르세요."
            )
            return

        try:
            preview = (
                self.sequence.prepare_approach(
                    self.registration,
                    self.current_base_marker,
                    self.slot_id,
                )
            )
            current = (
                self.client.get_coords()
            )

        except Exception as error:
            print(
                "p 미리보기 실패:",
                error,
            )
            return

        self.approach_preview = preview
        self.approach_preview_timestamp = (
            time.monotonic()
        )
        self.grasp_preview = None

        print()
        print("=" * 68)
        print(
            f"슬롯 {self.slot_id} 접근 목표 미리보기"
        )
        print(
            "현재 coords:",
            np.round(
                current,
                3,
            ).tolist(),
        )
        print(
            "적용 보정 XYZ(mm):",
            np.round(
                preview[
                    "compensation_xyz_mm"
                ],
                3,
            ).tolist(),
        )
        print(
            "보정 후 목표 coords:",
            np.round(
                preview["command"],
                3,
            ).tolist(),
        )
        print(
            "확인 후 x를 누르세요."
        )
        print("=" * 68)

    def _key_x(self):
        if self.approach_preview is None:
            print(
                "먼저 p로 접근 목표를 확인하세요."
            )
            return

        if (
            self.approach_preview["slot"]
            != self.slot_id
        ):
            print(
                "슬롯이 변경되었습니다. "
                "p를 다시 누르세요."
            )
            self.approach_preview = None
            return

        if (
            time.monotonic()
            - self.approach_preview_timestamp
            > APPROACH_PREVIEW_VALID_SEC
        ):
            print(
                "p 미리보기가 오래되었습니다. "
                "p를 다시 누르세요."
            )
            self.approach_preview = None
            return

        try:
            result = self.sequence.move_approach(
                self.approach_preview,
                self._run_with_gui,
            )
            print(
                "x 접근 결과:",
                json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            )
            print(
                "중심/간섭 확인 후 g를 누르세요."
            )

        except Exception as error:
            print(
                "x 접근 이동 실패:",
                error,
            )

        finally:
            self.approach_preview = None

    def _key_g(self):
        if self.current_base_marker is None:
            print(
                "현재 base marker가 없습니다. "
                "m → r부터 확인하세요."
            )
            return

        try:
            current = (
                self.client.get_coords()
            )
            preview = (
                self.sequence.prepare_grasp(
                    self.registration,
                    self.current_base_marker,
                    self.slot_id,
                    current,
                )
            )

        except Exception as error:
            print(
                "g 미리보기 실패:",
                error,
            )
            return

        print()
        print("=" * 72)
        print(
            f"슬롯 {self.slot_id} 집기 목표 미리보기"
        )
        print(
            "현재 coords:",
            np.round(
                current,
                3,
            ).tolist(),
        )
        print(
            "접근 기준:",
            np.round(
                preview[
                    "approach_reference"
                ],
                3,
            ).tolist(),
        )
        print(
            "현재→접근 위치/회전오차:",
            f"{preview['approach_position_error_mm']:.2f}mm / "
            f"{preview['approach_rotation_error_deg']:.2f}deg",
        )
        print(
            "grasp 기준:",
            np.round(
                preview[
                    "grasp_reference"
                ],
                3,
            ).tolist(),
        )
        print(
            "실제 명령 grasp:",
            np.round(
                preview[
                    "grasp_command"
                ],
                3,
            ).tolist(),
        )
        print(
            "grasp XYZ 보정:",
            preview[
                "grasp_compensation_xyz_mm"
            ],
        )
        print(
            "grasp RPY 보정:",
            preview[
                "grasp_rotation_compensation_deg"
            ],
        )
        print(
            "현재→grasp 이동/회전:",
            f"{preview['grasp_step_mm']:.2f}mm / "
            f"{preview['grasp_rotation_step_deg']:.2f}deg",
        )
        print(
            "안전검사:",
            (
                "PASS"
                if preview["safe"]
                else "BLOCK"
            ),
        )

        if preview["reasons"]:
            print(
                "차단 사유:",
                preview["reasons"],
            )

        print("=" * 72)

        if preview["safe"]:
            self.grasp_preview = preview
            self.grasp_preview_timestamp = (
                time.monotonic()
            )
            print(
                "미리보기 통과. "
                "손/간섭 확인 후 z를 누르세요."
            )
        else:
            self.grasp_preview = None

    def _key_z(self):
        if self.grasp_preview is None:
            print(
                "먼저 g로 grasp 목표와 "
                "안전검사를 확인하세요."
            )
            return

        if (
            self.grasp_preview["slot"]
            != self.slot_id
        ):
            print(
                "슬롯이 변경되었습니다. "
                "g를 다시 누르세요."
            )
            self.grasp_preview = None
            return

        if (
            time.monotonic()
            - self.grasp_preview_timestamp
            > GRASP_PREVIEW_VALID_SEC
        ):
            print(
                "g 미리보기가 오래되었습니다. "
                "g를 다시 누르세요."
            )
            self.grasp_preview = None
            return

        try:
            result = self.sequence.move_grasp(
                self.grasp_preview,
                self._run_with_gui,
            )

            print()
            print(
                "z 집기 자세 이동 결과:"
            )
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
            )
            print(
                "집게는 열린 상태입니다. "
                "b/d/o/y는 계속 차단되어 있습니다."
            )

        except Exception as error:
            print(
                "z 집기 이동 실패:",
                error,
            )

        finally:
            self.grasp_preview = None

    def _key_s(self):
        try:
            print(
                "STOP:",
                self.client.stop(),
            )
        except Exception as error:
            print(
                "STOP 실패:",
                error,
            )

    def _key_c(self):
        try:
            print(
                "현재 coords:",
                self.client.get_coords(),
            )
        except Exception as error:
            print(
                "GET_COORDS 실패:",
                error,
            )

    def pick(
        self,
        request,
        response,
    ):
        del request

        try:
            stable = (
                self.latest_camera["stable"]
                if self.latest_camera is not None
                else None
            )

            if stable is None:
                raise RuntimeError(
                    "stable ArUco pose is not available"
                )

            if self.preview_only:
                robot_coords = (
                    self.registration.marker_reference_coords()
                )
            else:
                robot_coords = (
                    self.client.get_coords()
                )

            base_marker = (
                self.transformer.base_marker(
                    robot_coords,
                    stable["transform"],
                )
            )

            transforms = (
                self.registration.targets_from_marker(
                    base_marker,
                    self.slot_id,
                )
            )

            targets = (
                self.sequence.coords_from_transforms(
                    transforms
                )
            )

            result = self.sequence.execute(
                targets,
                self.slot_id,
            )

            response.success = True
            response.message = json.dumps(
                result,
                ensure_ascii=False,
            )

        except Exception as error:
            self.log.exception(
                "right-arm transport pick blocked: %s",
                error,
            )
            response.success = False
            response.message = str(error)

        return response

    def close(self):
        self.camera.close()


def declare_parameters(node):
    defaults = {
        "robot_host": "192.168.0.112",
        "robot_port": 5001,
        "connect_timeout_sec": 3.0,
        # Must exceed the largest per-move timeout used in keyboard tests.
        "command_timeout_sec": 60.0,
        "camera_device": "/dev/video4",
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "camera_calibration_file": "",
        "handeye_file": "",
        "slot_registration_file": "",
        "marker_length_m": 0.030,
        "target_id": 0,
        "dictionary_id": int(
            cv2.aruco.DICT_4X4_50
        ),
        "stable_buffer_size": 12,
        "min_stable_samples": 10,
        "max_translation_spread_mm": 3.0,
        "max_rotation_spread_deg": 2.0,
        "max_reprojection_error_px": 4.0,
        "lost_limit": 5,
        "slot_id": 1,
        "preview_only": True,
        "allow_robot_motion": False,
        "allow_gripper_close": False,
        "marker_acquire_timeout_sec": 10.0,
        "preview_robot_coords": [
            62.7,
            -48.9,
            298.0,
            -143.33,
            -23.3,
            36.6,
        ],
        "position_limits_mm": [
            -350.0,
            350.0,
            0.0,
            500.0,
        ],
        "rotation_limits_deg": [
            -180.0,
            180.0,
        ],
    }

    for name, value in defaults.items():
        node.declare_parameter(
            name,
            value,
        )


def main(args=None):
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)

    node = Node(
        "right_arm_transport_pick"
    )
    app = None

    try:
        declare_parameters(node)
        app = RightArmTransportNode(
            node
        )

        node.get_logger().info(
            "right_arm_transport_pick started"
        )
        node.get_logger().info(
            "camera_device="
            f"{node.get_parameter('camera_device').value}"
        )
        node.get_logger().info(
            "robot="
            f"{node.get_parameter('robot_host').value}:"
            f"{node.get_parameter('robot_port').value}"
        )
        node.get_logger().info(
            "service ready: "
            "/right_arm/pick_transport_slot"
        )

        while (
            rclpy.ok()
            and not app.exit_requested
        ):
            app.process_frame_and_keyboard()
            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

    except KeyboardInterrupt:
        node.get_logger().info(
            "right_arm_transport_pick interrupted"
        )

    except Exception as error:
        node.get_logger().fatal(
            "right_arm_transport_pick failed: "
            f"{error}"
        )
        raise

    finally:
        # q / Ctrl+C does NOT latch STOP. Use the explicit 's' key for STOP.
        if app is not None:
            try:
                app.close()
            except Exception as close_error:
                node.get_logger().error(
                    "failed to close resources: "
                    f"{close_error}"
                )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
