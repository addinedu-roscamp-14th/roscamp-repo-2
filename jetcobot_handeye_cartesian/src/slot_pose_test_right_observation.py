import json
import math
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np


ROBOT_IP = "192.168.5.1"
ROBOT_PORT = 5000

CAMERA_DEVICE = "/dev/video0"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

CAMERA_CALIBRATION_FILE = Path(
    "/home/choiminjoon/charuco2/camera_calibration.npz"
)

HAND_EYE_FILE_CANDIDATES = [
    Path(
        "/home/choiminjoon/handeye_dataset_v2/"
        "handeye_calibration_FINAL.npz"
    ),
    Path(
        "/home/choiminjoon/handeye_dataset_v2/"
        "handeye_calibration.npz"
    ),
]

REGISTRATION_FILE = Path(
    "/home/choiminjoon/slot_pose_registration/"
    "slot_registration.json"
)

MARKER_LENGTH_M = 0.030
DICTIONARY_ID = aruco.DICT_4X4_50
TARGET_ID = 0

STABLE_BUFFER_SIZE = 12
MIN_STABLE_SAMPLES = 10
MAX_TRANSLATION_SPREAD_MM = 3.0
MAX_ROTATION_SPREAD_DEG = 2.0
MAX_REPROJECTION_ERROR_PX = 4.0
LOST_LIMIT = 5

TEST_SPEED = 4
GRASP_TEST_SPEED = 2
MOVE_MODE = 0
SAFE_Z_MARGIN_MM = 40.0

# r 관측 자세는 접근 목표와 이미 가까우므로, 이 거리 안에서는
# 기존 시작 자세용 +40mm 고상 안전점을 만들지 않고 접근 자세를
# 직접 요청한다. 높은 Z와 관측 자세의 손목각 조합에서 IK가
# 거부되는 문제를 피하기 위한 관측 자세 전용 경로다.
DIRECT_APPROACH_MAX_DISTANCE_MM = 100.0

# 접근 전에 ArUco를 다시 보는 오른팔 중간 관측 자세.
# 실제 측정값을 관절각으로 재현해 IK 경로 변화 가능성을 줄인다.
OBSERVATION_ANGLES = [
    98.43,
    -25.75,
    -5.18,
    -57.12,
    2.54,
    -32.34
]
# 슬롯 1 수동 티칭으로 확정한 관절각
# 운반대와 타이어 위치가 현재 티칭 당시와 같을 때 사용한다.
SLOT1_DIRECT_APPROACH_ANGLES = [
    106.43,
    -36.29,
    -4.39,
    -41.66,
    -1.31,
    -26.10,
]

SLOT1_DIRECT_GRASP_ANGLES = [
    104.15,
    -43.46,
    -4.39,
    -37.71,
    2.19,
    -29.61,
]

SLOT1_DIRECT_APPROACH_SPEED = 3
SLOT1_DIRECT_GRASP_SPEED = 2
SLOT1_DIRECT_ANGLE_TOLERANCE_DEG = 2.5
SLOT1_DIRECT_STABLE_REQUIRED = 2
SLOT1_DIRECT_TIMEOUT_SEC = 30.0

OBSERVATION_SPEED = 3
OBSERVATION_ANGLE_TOLERANCE_DEG = 3.0
OBSERVATION_STABLE_REQUIRED = 2
OBSERVATION_MOVE_TIMEOUT_SEC = 30.0
OBSERVATION_SETTLE_SEC = 1.0
OBSERVATION_RECAPTURE_TIMEOUT_SEC = 10.0

# 정지된 마커의 m 최초 검출과 r 재검출이 이 범위 안에서만
# 일치할 때 r 값을 채택한다. 관측 자세에 따른 핸드아이 오차가
# 목표 자세로 전파되는 것을 막는다.
MAX_OBSERVATION_MARKER_SHIFT_MM = 8.0
MAX_OBSERVATION_MARKER_ROTATION_DEG = 3.0

# 차이가 이 범위 안이면 트럭이 움직인 것이 아니라 관측 자세별
# 핸드아이 편차로 보고, 검증된 m 최초 검출값으로 되돌린다.
# 이 범위까지도 넘으면 실제 트럭 이동 가능성이 있으므로 차단한다.
MAX_OBSERVATION_FALLBACK_SHIFT_MM = 30.0
MAX_OBSERVATION_FALLBACK_ROTATION_DEG = 10.0

# 현재 시험에서는 집기 목표 확인과 저속 하강까지만 허용한다.
# 그리퍼 닫기·상승·내려놓기·열기는 계속 차단한다.
ENABLE_GRASP_PREVIEW_AND_DESCENT = True

# 접근 자세에서 실제 집기 자세로 들어가기 전 안전 검사 기준
MAX_APPROACH_POSITION_ERROR_MM = 20.0
MAX_APPROACH_ROTATION_ERROR_DEG = 10.0
MAX_GRASP_TRANSLATION_STEP_MM = 120.0
MAX_GRASP_ROTATION_STEP_DEG = 35.0

# coarse 이동 뒤 남은 오차벡터를 현재 명령값에 더해
# 미세 보정 명령을 1회 계산한다.
FINE_SETTLE_SPEED = 1
FINE_SETTLE_MIN_POSITION_MM = 2.5
FINE_SETTLE_MAX_POSITION_MM = 15.0
FINE_SETTLE_MAX_ROTATION_DEG = 10.0
FINE_SETTLE_GAIN = 1.0
FINE_SETTLE_MAX_AXIS_MM = 6.0
GRASP_PREVIEW_VALID_SEC = 120.0

# 접근·집기 시험 단계는 마지막 정밀 내려놓기와 달리 다음 동작 전
# 육안 확인 및 별도 잔여오차 보정이 있다. 서버 기본값(8mm, 2회,
# 최대 30초)을 그대로 쓰면 팔이 이미 멈춘 뒤에도 GUI에
# ROBOT MOVING이 오래 표시되므로 이 단계만 빠른 완료 판정을 사용한다.
COARSE_MOVE_POSITION_TOLERANCE_MM = 12.0
COARSE_MOVE_ROTATION_TOLERANCE_DEG = 6.0
COARSE_MOVE_STABLE_REQUIRED = 1
COARSE_MOVE_TIMEOUT_SEC = 10.0
FINE_MOVE_POSITION_TOLERANCE_MM = 12.0
FINE_MOVE_ROTATION_TOLERANCE_DEG = 6.0
FINE_MOVE_STABLE_REQUIRED = 1
FINE_MOVE_TIMEOUT_SEC = 8.0
INTER_WAYPOINT_SETTLE_SEC = 0.2

# 슬롯 3의 높은 안전 중간점에서는 손목 RPY가 IK 특이구간 때문에
# 약 11~12도 변할 수 있다. 이 구간은 위치만 안전하게 도달하면
# 다음 단계에서 실제 손목 자세를 다시 읽어 이어간다.
# 최종 접근 자세와 집기 자세에는 이 완화 판정을 적용하지 않는다.
SLOT3_INTERMEDIATE_POSITION_ACCEPT_MM = 12.0
SLOT3_INTERMEDIATE_ROTATION_ACCEPT_DEG = 15.0

# 슬롯 4 자동 집기·상승 시험
AUTO_PICK_TEST_SLOT = 4
AUTO_PICK_CONFIRM_SEC = 8.0
AUTO_PLACE_CONFIRM_SEC = 8.0
AUTO_OPEN_CONFIRM_SEC = 8.0
AUTO_PICK_MAX_POSITION_ERROR_MM = 10.0
AUTO_PICK_MAX_ROTATION_ERROR_DEG = 8.0
AUTO_LIFT_DISTANCE_MM = 30.0
AUTO_LIFT_SPEED = 2
AUTO_PLACE_SPEED = 1
AUTO_RETREAT_SPEED = 2
AUTO_PLACE_SETTLE_SEC = 1.0
PRE_PLACE_HEIGHT_MM = 10.0

# 10mm 상공은 최종 내려놓기 지점이 아니라 정렬용 중간점이다.
# 이전 시험에서 best error 2.91mm까지 들어왔지만 2회 연속 안정 조건을
# 만족하지 못해 실패 처리되었다. 상공 단계만 5mm/1회로 완화하고,
# 실제 하강 단계의 2.5mm 정밀 조건과 Y 1mm 차단은 그대로 유지한다.
PRE_PLACE_POSITION_TOLERANCE_MM = 5.0
PRE_PLACE_ROTATION_TOLERANCE_DEG = 4.0
PRE_PLACE_STABLE_REQUIRED = 1

PRECISE_PLACE_POSITION_TOLERANCE_MM = 2.5
PRECISE_PLACE_ROTATION_TOLERANCE_DEG = 3.5
PRECISE_PLACE_STABLE_REQUIRED = 3
PRECISE_PLACE_TIMEOUT_SEC = 45.0
PRECISE_PLACE_MAX_Y_ERROR_MM = 1.0

# 슬롯 4는 grasp Y 잔여오차를 100% 보정하면 실제 팔이
# 명령보다 더 앞쪽(+Y)으로 반응하는 경향이 있었다.
# 반복 로그 기준으로 Y축 미세보정만 65% 적용한다.
SLOT4_GRASP_Y_FINE_GAIN = 0.65

# pymycobot send_coords mode: 0=angular, 1=linear.
# 슬롯 4의 짧은 수직 상승·하강·후퇴에는 직선 모드를 사용한다.
LINEAR_MOVE_MODE = 1

# 슬롯 4 내려놓기 전용 위치 보정 [X, Y, Z] mm.
# 현재 로그에서 하강 후 Y가 집기 기준보다 약 +1.6mm 앞쪽이어서
# 처음에는 Y -2mm로 보정한다. 집기 좌표 자체에는 적용하지 않는다.
PLACE_POSITION_OFFSET_MM = np.array(
    [0.0, -2.0, 0.0],
    dtype=np.float64,
)

# 슬롯 2 전용 단일 Y축 미세전진 설정
SLOT2_SINGLE_Y_SPEED = 1
SLOT2_SINGLE_Y_STEP_MM = 1.0
SLOT2_SINGLE_Y_TOLERANCE_MM = 1.2
SLOT2_SINGLE_Y_MAX_TOTAL_MM = 8.0
SLOT2_SINGLE_Y_MAX_STEPS = 8
SLOT2_SINGLE_Y_MAX_TOTAL_X_DRIFT_MM = 3.0
SLOT2_SINGLE_Y_MAX_TOTAL_Z_DRIFT_MM = 3.0
SLOT2_SINGLE_Y_MAX_TOTAL_ROTATION_DRIFT_DEG = 4.0

# 실제로 도달하려는 집기 깊이 기준.
# 슬롯 1은 집기 자세를 다시 등록했으므로 예전의 Z -2mm 보정을 제거한다.
GRASP_PHYSICAL_DEPTH_OFFSET_MM = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 3.0,
}

# 슬롯별 접근 자세 보정값 [X, Y, Z] mm
# 슬롯 1은 두 번의 반복 시험에서 평균적으로
# X -3.8mm, Y +2.6mm, Z -10.4mm 벗어났기 때문에
# 반대 방향으로 보정한다.
APPROACH_POSITION_COMPENSATION_MM = {
    # 동일 조건 2회 평균으로 갱신한 슬롯 1 approach 보정값.
    # 평균 actual - original_target:
    # [+3.14, -9.07, +4.09] mm
    # 기존 보정 [4.0, -3.3, 12.0]에서 평균 오차를 반대로 적용.
    # 슬롯 1: 현재 위치에서 왼쪽 방향 시험 X -7mm
    1: np.array([-7.0, 3.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 5.0], dtype=np.float64),
    3: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    4: np.array([0.0, 0.0, 0.0], dtype=np.float64),
}

# 집기 자세는 접근 자세와 실제 반복오차가 달랐으므로
# 접근 보정값과 분리해서 관리한다.
#
# 슬롯 1 grasp 반복 시험 결과:
# 원래 grasp : [-2.185, 233.231, 227.805]
# 실제 도착 : [ 3.800, 233.000, 224.000]
#
# 이전 명령과 실제 도착 편차를 역보정하여
# 슬롯 1 grasp 전용 XYZ 보정값을 적용한다.
GRASP_POSITION_COMPENSATION_MM = {
    # 슬롯 1: 새 등록값 기준 X/Y 방향은 아직 검증 전이므로 유지하고,
    # 반복 시험에서 실제 Z가 명령보다 약 6mm 낮게 도착한 경향을 고려해
    # 우선 명령 Z를 3mm 높여 안전하게 재시험한다.
    # 슬롯 1: 접근과 동일하게 왼쪽 X -7mm, 기존 안전 Z +3mm 유지
    1: np.array([-7.0, 9.0, 3.0], dtype=np.float64),
    # 슬롯 2: 위치 이동 반복시험 결과 X +5mm, Z +8mm를 최종값으로 사용한다.
    2: np.array([5.0, 0.0, 8.0], dtype=np.float64),
    # 슬롯 3: 중심 보정 X +1mm, Y +1mm와 눌림 방지 Z +10mm를 적용한다.
    # 슬롯 2·3은 집기 후 자동 XYZ 미세보정을 생략하여 실제 Z가 다시 내려가지 않게 한다.
    3: np.array([1.0, 1.0, 10.0], dtype=np.float64),
    4: np.array([0.0, 5.0, 3.0], dtype=np.float64),
}

# 슬롯별 집기 회전 보정값 [Rx, Ry, Rz] deg
# 슬롯 3은 반복 시험을 통해 Rx +3도, Ry -2도 보정을 적용한다.
# 집기 자동 미세보정에서는 이 회전값을 다시 변경하지 않는다.
GRASP_ROTATION_COMPENSATION_DEG = {
    1: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    3: np.array([3.0, -2.0, 0.0], dtype=np.float64),
    4: np.array([0.0, -2.0, 0.0], dtype=np.float64),
}


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=np.float64,
    )


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=np.float64,
    )


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        dtype=np.float64,
    )


def mycobot_coords_to_transform(
    coords_mm_deg: List[float],
) -> np.ndarray:
    x, y, z, rx_deg, ry_deg, rz_deg = [
        float(value)
        for value in coords_mm_deg
    ]

    rx, ry, rz = np.deg2rad(
        [rx_deg, ry_deg, rz_deg]
    )

    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    t_m = np.array([x, y, z], dtype=np.float64) / 1000.0

    return make_transform(R, t_m)


def rotation_matrix_to_rpy_deg(
    R: np.ndarray,
) -> np.ndarray:
    sy = math.sqrt(
        R[0, 0] ** 2
        + R[1, 0] ** 2
    )
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.rad2deg(
        [roll, pitch, yaw]
    )


def transform_to_mycobot_coords(
    T: np.ndarray,
) -> List[float]:
    xyz_mm = T[:3, 3] * 1000.0
    rpy_deg = rotation_matrix_to_rpy_deg(
        T[:3, :3]
    )

    return [
        float(xyz_mm[0]),
        float(xyz_mm[1]),
        float(xyz_mm[2]),
        float(rpy_deg[0]),
        float(rpy_deg[1]),
        float(rpy_deg[2]),
    ]


def wrapped_angle_error_deg(
    actual_deg: float,
    target_deg: float,
) -> float:
    delta = (
        float(actual_deg)
        - float(target_deg)
        + 180.0
    ) % 360.0 - 180.0

    return abs(delta)


def pose_error(
    actual: List[float],
    target: List[float],
) -> Tuple[float, float]:
    actual_array = np.asarray(
        actual,
        dtype=np.float64,
    )
    target_array = np.asarray(
        target,
        dtype=np.float64,
    )

    position_error_mm = float(
        np.linalg.norm(
            actual_array[:3]
            - target_array[:3]
        )
    )

    rotation_error_deg = max(
        wrapped_angle_error_deg(
            actual_array[index],
            target_array[index],
        )
        for index in range(3, 6)
    )

    return (
        position_error_mm,
        rotation_error_deg,
    )


def average_rotation(
    rotations: List[np.ndarray],
) -> np.ndarray:
    M = np.mean(
        np.stack(rotations, axis=0),
        axis=0,
    )
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rotation_difference_deg(
    R_a: np.ndarray,
    R_b: np.ndarray,
) -> float:
    R_delta = R_a.T @ R_b
    value = float(
        np.clip(
            (np.trace(R_delta) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(value)
    )


def find_hand_eye_file() -> Path:
    for path in HAND_EYE_FILE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Eye-in-Hand 파일을 찾지 못했습니다."
    )


def load_hand_eye(
    path: Path,
) -> np.ndarray:
    data = np.load(
        path,
        allow_pickle=True,
    )

    if "T_gripper_camera" in data:
        return np.asarray(
            data["T_gripper_camera"],
            dtype=np.float64,
        ).reshape(4, 4)

    if (
        "R_cam2gripper" in data
        and "t_cam2gripper_m" in data
    ):
        return make_transform(
            np.asarray(
                data["R_cam2gripper"],
                dtype=np.float64,
            ).reshape(3, 3),
            np.asarray(
                data["t_cam2gripper_m"],
                dtype=np.float64,
            ).reshape(3),
        )

    raise KeyError(
        "Eye-in-Hand 변환을 찾지 못했습니다."
    )


def load_camera_calibration(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(
        path,
        allow_pickle=False,
    )

    K = np.asarray(
        data["camera_matrix"],
        dtype=np.float64,
    ).reshape(3, 3)

    D = np.asarray(
        data["dist_coeffs"],
        dtype=np.float64,
    )

    return K, D


def load_registration(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"슬롯 등록 파일이 없습니다: {path}"
        )

    registration = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    for slot_number in ("1", "2", "3", "4"):
        slot = registration.get(
            "slots",
            {},
        ).get(slot_number, {})

        if "approach" not in slot:
            print(
                f"경고: 슬롯 {slot_number}에 접근 자세가 없습니다."
            )

        if "grasp" not in slot:
            print(
                f"경고: 슬롯 {slot_number}에 집기 자세가 없습니다."
            )

    return registration


class RobotClient:
    def __init__(
        self,
        ip: str,
        port: int,
    ) -> None:
        self.sock = socket.create_connection(
            (ip, port),
            timeout=5.0,
        )
        self.sock.settimeout(120.0)
        self.sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        self.buffer = ""

    def request(
        self,
        payload: Any,
    ) -> Dict[str, Any]:
        if isinstance(payload, str):
            line = payload
        else:
            line = json.dumps(
                payload,
                ensure_ascii=False,
            )

        self.sock.sendall(
            (
                line.strip()
                + "\n"
            ).encode("utf-8")
        )

        while "\n" not in self.buffer:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout as exc:
                raise TimeoutError(
                    "JetCobot 서버 응답을 120초 동안 받지 못했습니다. "
                    "JetCobot 서버 터미널에서 이동 상태와 오류를 확인하세요."
                ) from exc

            if not chunk:
                raise ConnectionError(
                    "JetCobot 서버 연결이 종료되었습니다."
                )

            self.buffer += chunk.decode(
                "utf-8",
                errors="ignore",
            )

        line, self.buffer = self.buffer.split(
            "\n",
            1,
        )

        return json.loads(line)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def robot_request_with_gui_pump(
    robot: RobotClient,
    payload: Any,
    window_name: str,
    display_frame: np.ndarray,
    status_text: str,
) -> Dict[str, Any]:
    """
    robot.request()는 서버 응답이 올 때까지 기다리기 때문에
    그대로 호출하면 OpenCV GUI 이벤트 처리가 멈춰
    Ubuntu에서 '응답 없음' 창이 뜰 수 있다.

    로봇 요청은 작업 스레드에서 처리하고,
    메인 스레드는 cv2.waitKey()를 계속 호출해
    창이 응답하도록 유지한다.
    """
    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def worker() -> None:
        try:
            result_box["response"] = robot.request(
                payload
            )
        except BaseException as exc:
            error_box["error"] = exc

    thread = threading.Thread(
        target=worker,
        daemon=True,
    )
    thread.start()

    animation_index = 0

    while thread.is_alive():
        wait_frame = display_frame.copy()
        dots = "." * (
            animation_index % 4
        )
        animation_index += 1

        cv2.rectangle(
            wait_frame,
            (10, 105),
            (FRAME_WIDTH - 10, 180),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            wait_frame,
            f"ROBOT MOVING{dots}",
            (25, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            wait_frame,
            status_text,
            (25, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            window_name,
            wait_frame,
        )

        # 창 이벤트를 계속 처리한다.
        # 이동 중에는 이 키 입력으로 로봇을 정지시키지 않는다.
        cv2.waitKey(50)

    thread.join()

    if "error" in error_box:
        raise error_box["error"]

    response = result_box.get("response")

    if not isinstance(response, dict):
        raise RuntimeError(
            "JetCobot 서버 응답 형식이 올바르지 않습니다."
        )

    return response


def create_detector():
    dictionary = aruco.getPredefinedDictionary(
        DICTIONARY_ID
    )

    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02

    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = (
            aruco.CORNER_REFINE_SUBPIX
        )

    detector = (
        aruco.ArucoDetector(
            dictionary,
            params,
        )
        if hasattr(aruco, "ArucoDetector")
        else None
    )

    return dictionary, params, detector


def detect_markers(
    gray: np.ndarray,
    dictionary,
    params,
    detector,
):
    if detector is not None:
        return detector.detectMarkers(
            gray
        )

    return aruco.detectMarkers(
        gray,
        dictionary,
        parameters=params,
    )


def marker_object_points() -> np.ndarray:
    half = MARKER_LENGTH_M / 2.0

    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_marker_pose(
    corners: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Optional[Dict[str, Any]]:
    image_points = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(4, 2)

    object_points = marker_object_points()

    flag = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(
            cv2,
            "SOLVEPNP_IPPE_SQUARE",
        )
        else cv2.SOLVEPNP_ITERATIVE
    )

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        D,
        flags=flag,
    )

    if not ok:
        return None

    if hasattr(
        cv2,
        "solvePnPRefineLM",
    ):
        try:
            rvec, tvec = (
                cv2.solvePnPRefineLM(
                    object_points,
                    image_points,
                    K,
                    D,
                    rvec,
                    tvec,
                )
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        D,
    )

    projected = np.asarray(
        projected,
        dtype=np.float64,
    ).reshape(4, 2)

    error_px = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        projected
                        - image_points
                    ) ** 2,
                    axis=1,
                )
            )
        )
    )

    R_camera_marker, _ = cv2.Rodrigues(
        rvec
    )
    t_camera_marker_m = np.asarray(
        tvec,
        dtype=np.float64,
    ).reshape(3)

    if (
        not np.all(
            np.isfinite(
                t_camera_marker_m
            )
        )
        or t_camera_marker_m[2] <= 0
    ):
        return None

    return {
        "R_camera_marker": R_camera_marker,
        "t_camera_marker_m": t_camera_marker_m,
        "rvec": np.asarray(
            rvec,
            dtype=np.float64,
        ).reshape(3),
        "error_px": error_px,
    }


PoseRecord = Tuple[
    np.ndarray,
    np.ndarray,
    float,
]

pose_buffer: Deque[PoseRecord] = deque(
    maxlen=STABLE_BUFFER_SIZE
)


def stable_marker_pose() -> Optional[Dict[str, Any]]:
    if len(pose_buffer) < MIN_STABLE_SAMPLES:
        return None

    records = list(
        pose_buffer
    )[-MIN_STABLE_SAMPLES:]

    rotations = [
        record[0]
        for record in records
    ]
    translations = np.stack(
        [
            record[1]
            for record in records
        ],
        axis=0,
    )
    errors = np.asarray(
        [
            record[2]
            for record in records
        ],
        dtype=np.float64,
    )

    R_mean = average_rotation(
        rotations
    )
    t_median = np.median(
        translations,
        axis=0,
    )

    translation_spread_mm = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        translations
                        - t_median
                    ) ** 2,
                    axis=1,
                )
            )
        )
        * 1000.0
    )

    rotation_spread_deg = float(
        np.sqrt(
            np.mean(
                [
                    rotation_difference_deg(
                        R_mean,
                        R,
                    ) ** 2
                    for R in rotations
                ]
            )
        )
    )

    reprojection_error_px = float(
        np.median(errors)
    )

    if (
        translation_spread_mm
        > MAX_TRANSLATION_SPREAD_MM
        or rotation_spread_deg
        > MAX_ROTATION_SPREAD_DEG
        or reprojection_error_px
        > MAX_REPROJECTION_ERROR_PX
    ):
        return None

    return {
        "R_camera_marker": R_mean,
        "t_camera_marker_m": t_median,
        "translation_spread_mm": translation_spread_mm,
        "rotation_spread_deg": rotation_spread_deg,
        "reprojection_error_px": reprojection_error_px,
    }


def select_target_marker(
    corners,
    ids,
) -> Optional[int]:
    if ids is None or len(ids) == 0:
        return None

    flat_ids = np.asarray(
        ids,
        dtype=np.int32,
    ).reshape(-1)

    matches = np.where(
        flat_ids == TARGET_ID
    )[0]

    if len(matches) == 0:
        return None

    return int(matches[0])


def compute_current_base_marker(
    robot_pose: Dict[str, Any],
    stable_pose: Dict[str, Any],
    T_gripper_camera: np.ndarray,
) -> np.ndarray:
    T_base_gripper = (
        mycobot_coords_to_transform(
            robot_pose["coords_mm_deg"]
        )
    )

    T_camera_marker = make_transform(
        stable_pose["R_camera_marker"],
        stable_pose["t_camera_marker_m"],
    )

    return (
        T_base_gripper
        @ T_gripper_camera
        @ T_camera_marker
    )


def compute_slot_target(
    registration: Dict[str, Any],
    slot_number: int,
    pose_name: str,
    T_base_marker_current: np.ndarray,
) -> np.ndarray:
    slot = registration["slots"][
        str(slot_number)
    ]

    if pose_name not in slot:
        raise KeyError(
            f"슬롯 {slot_number}의 {pose_name} 자세가 없습니다."
        )

    T_marker_gripper_saved = np.asarray(
        slot[pose_name][
            "T_marker_gripper"
        ],
        dtype=np.float64,
    ).reshape(4, 4)

    return (
        T_base_marker_current
        @ T_marker_gripper_saved
    )


def main() -> None:
    print("=" * 70)
    print("오른팔 슬롯 1 Z 상승 조합 미세조정 시험 v49")
    print("시험 흐름: m → r → p → x → g 미리보기 → z 저속 하강")
    print("=" * 70)

    registration = load_registration(
        REGISTRATION_FILE
    )

    T_gripper_camera = load_hand_eye(
        find_hand_eye_file()
    )
    K, D = load_camera_calibration(
        CAMERA_CALIBRATION_FILE
    )

    dictionary, params, detector = (
        create_detector()
    )

    robot = RobotClient(
        ROBOT_IP,
        ROBOT_PORT,
    )

    print("서버:", robot.request("PING"))
    print("안전 설정:", robot.request("STATUS"))

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )
    cap.set(
        cv2.CAP_PROP_FPS,
        FRAME_FPS,
    )
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not cap.isOpened():
        robot.close()
        raise RuntimeError(
            f"카메라를 열 수 없습니다: {CAMERA_DEVICE}"
        )

    time.sleep(1.0)

    for _ in range(10):
        cap.read()

    selected_slot = 1
    lost_count = 0

    current_base_marker: Optional[
        np.ndarray
    ] = None

    base_marker_before_observation: Optional[
        np.ndarray
    ] = None
    observation_recapture_pending = False
    observation_recapture_not_before = 0.0
    observation_recapture_deadline = 0.0
    observation_recapture_complete = False

    preview_target: Optional[
        List[float]
    ] = None

    preview_reference_target: Optional[
        List[float]
    ] = None

    preview_slot: Optional[int] = None
    preview_timestamp = 0.0

    grasp_preview_target: Optional[
        List[float]
    ] = None
    grasp_reference_target: Optional[
        List[float]
    ] = None
    grasp_preview_slot: Optional[int] = None
    grasp_preview_timestamp = 0.0

    # z 집기 coarse 이동이 성공한 뒤 y 미세전진에 사용하는 기준
    last_grasp_reference_target: Optional[List[float]] = None
    last_grasp_move_slot: Optional[int] = None

    # b/d/o는 실수 방지를 위해 제한시간 안에 두 번 눌러야 실행한다.
    pick_confirm_until = 0.0
    place_confirm_until = 0.0
    open_confirm_until = 0.0
    tire_lifted = False
    lift_origin_coords: Optional[List[float]] = None

    print()
    print("1~4: 슬롯 선택")
    print("m: 시작 자세에서 현재 마커 최초 저장")
    print("r: 중간 관측 자세 이동 → 정지 → ArUco 자동 재검출")
    print("p: 접근 목표 미리보기")
    print("x: 접근 자세로 단계 이동")
    print("g: 집기 목표 미리보기 및 안전 검사")
    print("z: 미리보기 통과 후 집기 자세까지 저속 하강(집게는 열림)")
    print("b/d/o/y: 현재 버전에서는 안전을 위해 차단")
    print("s: 정지 요청(프로그램이 키를 받을 수 있을 때)")
    print("c: 현재 로봇 pose")
    print("q: 종료")
    print()
    print("주의: r/x 이동 중에는 키 입력 처리가 멈출 수 있습니다.")
    print("실제 비상 상황에서는 로봇 전원/물리 정지를 사용하세요.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print(
                    "카메라 프레임 읽기 실패"
                )
                break

            display = frame.copy()
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            corners, ids, _ = detect_markers(
                gray,
                dictionary,
                params,
                detector,
            )

            marker_index = select_target_marker(
                corners,
                ids,
            )

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(
                    display,
                    corners,
                    ids,
                )

            current_error_px = None

            if marker_index is not None:
                lost_count = 0

                pose = estimate_marker_pose(
                    corners[marker_index],
                    K,
                    D,
                )

                if pose is not None:
                    current_error_px = float(
                        pose["error_px"]
                    )

                    if (
                        current_error_px
                        <= MAX_REPROJECTION_ERROR_PX
                        and (
                            not observation_recapture_pending
                            or time.monotonic()
                            >= observation_recapture_not_before
                        )
                    ):
                        pose_buffer.append(
                            (
                                pose[
                                    "R_camera_marker"
                                ],
                                pose[
                                    "t_camera_marker_m"
                                ],
                                current_error_px,
                            )
                        )

                    try:
                        cv2.drawFrameAxes(
                            display,
                            K,
                            D,
                            pose["rvec"].reshape(3, 1),
                            pose[
                                "t_camera_marker_m"
                            ].reshape(3, 1),
                            MARKER_LENGTH_M * 0.7,
                            2,
                        )
                    except cv2.error:
                        pass

            else:
                lost_count += 1

                if lost_count > LOST_LIMIT:
                    pose_buffer.clear()

            stable = stable_marker_pose()
            ready = stable is not None

            # r 이동 완료 후 1초간 기다린 다음, 새 프레임들만으로
            # 자동 재검출하여 이후 접근 계산에 사용할 기준을 갱신한다.
            if (
                observation_recapture_pending
                and time.monotonic()
                >= observation_recapture_not_before
                and stable is not None
            ):
                robot_pose = robot.request("POSE")

                if not robot_pose.get("ok"):
                    print(
                        "중간 관측 자세 POSE 실패:",
                        robot_pose,
                    )
                    observation_recapture_pending = False
                    observation_recapture_complete = False
                    current_base_marker = None
                    pose_buffer.clear()
                else:
                    candidate_base_marker = (
                        compute_current_base_marker(
                            robot_pose,
                            stable,
                            T_gripper_camera,
                        )
                    )

                    shift_mm = 0.0
                    rotation_shift_deg = 0.0

                    if base_marker_before_observation is not None:
                        shift_mm = float(
                            np.linalg.norm(
                                candidate_base_marker[:3, 3]
                                - base_marker_before_observation[:3, 3]
                            )
                            * 1000.0
                        )
                        rotation_shift_deg = rotation_difference_deg(
                            candidate_base_marker[:3, :3],
                            base_marker_before_observation[:3, :3],
                        )

                    print()
                    print("중간 관측 자세 ArUco 재검출 완료")
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
                            "재검출 차이가 최대 안전 기준보다 커서 "
                            "접근 계산을 차단했습니다."
                        )
                        print(
                            "로봇·운반대가 움직이지 않았는지 확인한 뒤 "
                            "시작 자세에서 m부터 다시 실행하세요."
                        )
                        current_base_marker = None
                        observation_recapture_complete = False
                    elif (
                        shift_mm
                        > MAX_OBSERVATION_MARKER_SHIFT_MM
                        or rotation_shift_deg
                        > MAX_OBSERVATION_MARKER_ROTATION_DEG
                    ):
                        print(
                            "r 재검출값이 정지 마커 일치 기준을 "
                            "벗어나므로 목표 계산에는 사용하지 않습니다."
                        )
                        print(
                            "관측 자세별 핸드아이 편차로 판단하여 "
                            "m 최초 검출값을 복원합니다."
                        )
                        current_base_marker = (
                            base_marker_before_observation.copy()
                        )
                        observation_recapture_complete = True
                        print(
                            "복원된 base marker XYZ(mm):",
                            np.round(
                                current_base_marker[:3, 3] * 1000.0,
                                3,
                            ),
                        )
                        print(
                            "이제 p로 m 기준 접근 목표를 확인한 뒤 "
                            "x를 누르세요."
                        )
                    else:
                        current_base_marker = candidate_base_marker
                        observation_recapture_complete = True
                        print(
                            "갱신된 base marker XYZ(mm):",
                            np.round(
                                current_base_marker[:3, 3] * 1000.0,
                                3,
                            ),
                        )
                        print("이제 p → x까지만 시험하세요.")

                    observation_recapture_pending = False
                    base_marker_before_observation = None
                    preview_target = None
                    preview_reference_target = None
                    preview_slot = None
                    grasp_preview_target = None
                    grasp_reference_target = None
                    grasp_preview_slot = None
                    last_grasp_reference_target = None
                    last_grasp_move_slot = None
                    pose_buffer.clear()
                    lost_count = 0

            elif (
                observation_recapture_pending
                and time.monotonic() >= observation_recapture_deadline
            ):
                print()
                print(
                    "중간 관측 자세 ArUco 재검출 시간 초과 "
                    f"({OBSERVATION_RECAPTURE_TIMEOUT_SEC:.1f}초)"
                )
                print(
                    "마커가 선명하게 보이는지 확인한 뒤 "
                    "시작 자세에서 m → r을 다시 실행하세요."
                )
                observation_recapture_pending = False
                observation_recapture_complete = False
                base_marker_before_observation = None
                current_base_marker = None
                pose_buffer.clear()
                lost_count = 0

            error_text = (
                f"{current_error_px:.3f}px"
                if current_error_px is not None
                else "NONE"
            )

            cv2.putText(
                display,
                (
                    (
                        "OBSERVE - recapturing"
                        if observation_recapture_pending
                        else "READY - press m"
                    )
                    if ready
                    else (
                        f"stabilizing "
                        f"{len(pose_buffer)}/"
                        f"{MIN_STABLE_SAMPLES}"
                    )
                ),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    (0, 255, 0)
                    if ready
                    else (0, 165, 255)
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"reproj: {error_text} | "
                    f"slot: {selected_slot}"
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
                    f"marker captured: "
                    f"{current_base_marker is not None}"
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
                "m first capture | r observe+recapture | p preview | x approach",
                (10, FRAME_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Slot Approach and Grasp Test",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if (
                not ENABLE_GRASP_PREVIEW_AND_DESCENT
                and key in (ord("g"), ord("z"))
            ):
                print(
                    "현재 버전에서는 집기 목표 확인과 하강이 "
                    "비활성화되어 있습니다."
                )
                continue

            if key in (
                ord("b"),
                ord("d"),
                ord("o"),
                ord("y"),
            ):
                print(
                    "현재 시험은 g 미리보기와 z 저속 하강까지만 "
                    "허용합니다. 그리퍼 닫기·상승·놓기는 차단되어 있습니다."
                )
                continue

            if key in (
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
            ):
                selected_slot = int(chr(key))
                preview_target = None
                preview_reference_target = None
                preview_slot = None
                grasp_preview_target = None
                grasp_reference_target = None
                grasp_preview_slot = None
                last_grasp_reference_target = None
                last_grasp_move_slot = None
                pick_confirm_until = 0.0
                place_confirm_until = 0.0
                open_confirm_until = 0.0
                tire_lifted = False
                lift_origin_coords = None
                print(
                    "선택 슬롯:",
                    selected_slot,
                )

            elif key == ord("m"):
                if observation_recapture_pending:
                    print("중간 자세 재검출이 끝날 때까지 기다리세요.")
                    continue

                stable = stable_marker_pose()

                if stable is None:
                    print(
                        "READY 이후 m을 누르세요."
                    )
                    continue

                robot_pose = robot.request(
                    "POSE"
                )

                if not robot_pose.get("ok"):
                    print(
                        "POSE 실패:",
                        robot_pose,
                    )
                    continue

                current_base_marker = (
                    compute_current_base_marker(
                        robot_pose,
                        stable,
                        T_gripper_camera,
                    )
                )
                base_marker_before_observation = None
                observation_recapture_pending = False
                observation_recapture_complete = False

                preview_target = None
                preview_slot = None
                grasp_preview_target = None
                grasp_reference_target = None
                grasp_preview_slot = None
                last_grasp_reference_target = None
                last_grasp_move_slot = None
                pick_confirm_until = 0.0
                place_confirm_until = 0.0
                open_confirm_until = 0.0
                tire_lifted = False
                lift_origin_coords = None
                pose_buffer.clear()
                lost_count = 0

                print()
                print("현재 ArUco 기준 저장 완료")
                print(
                    "base marker XYZ(mm):",
                    np.round(
                        current_base_marker[
                            :3,
                            3,
                        ]
                        * 1000.0,
                        3,
                    ),
                )
                print(
                    "이제 트럭을 움직이지 말고 r을 누르세요."
                )

            elif key == ord("r"):
                if observation_recapture_pending:
                    print("이미 중간 자세에서 ArUco를 재검출 중입니다.")
                    continue

                if current_base_marker is None:
                    print(
                        "먼저 시작 자세의 READY 상태에서 "
                        "m으로 최초 마커를 저장하세요."
                    )
                    continue

                print()
                print("중간 관측 자세 이동을 시작합니다.")
                print("목표 관절각:", OBSERVATION_ANGLES)
                print("주변 간섭이 없는지 확인하세요.")
                observation_recapture_complete = False

                try:
                    move_response = robot_request_with_gui_pump(
                        robot,
                        {
                            "command": "MOVE_ANGLES",
                            "angles": OBSERVATION_ANGLES,
                            "speed": OBSERVATION_SPEED,
                            "angle_tolerance_deg": (
                                OBSERVATION_ANGLE_TOLERANCE_DEG
                            ),
                            "stable_required": (
                                OBSERVATION_STABLE_REQUIRED
                            ),
                            "timeout_sec": (
                                OBSERVATION_MOVE_TIMEOUT_SEC
                            ),
                        },
                        "Slot Approach and Grasp Test",
                        display,
                        "moving to ArUco observation pose",
                    )
                except Exception as exc:
                    print("중간 관측 자세 이동 요청 실패:", exc)
                    continue

                print("중간 관측 자세 이동 결과:", move_response)

                if not move_response.get("ok"):
                    print(
                        "중간 관측 자세에 도달하지 못해 "
                        "재검출을 시작하지 않습니다."
                    )
                    continue

                base_marker_before_observation = (
                    current_base_marker.copy()
                )
                current_base_marker = None
                observation_recapture_pending = True
                observation_recapture_not_before = (
                    time.monotonic() + OBSERVATION_SETTLE_SEC
                )
                observation_recapture_deadline = (
                    observation_recapture_not_before
                    + OBSERVATION_RECAPTURE_TIMEOUT_SEC
                )

                preview_target = None
                preview_reference_target = None
                preview_slot = None
                grasp_preview_target = None
                grasp_reference_target = None
                grasp_preview_slot = None
                last_grasp_reference_target = None
                last_grasp_move_slot = None
                pose_buffer.clear()
                lost_count = 0

                print(
                    f"{OBSERVATION_SETTLE_SEC:.1f}초 정지 후 "
                    "안정된 ArUco 여러 프레임을 자동 수집합니다."
                )

            elif key == ord("p"):
                if observation_recapture_pending:
                    print("중간 자세의 ArUco 재검출이 끝날 때까지 기다리세요.")
                    continue

                if current_base_marker is None:
                    print(
                        "m 최초 검출과 r 중간 자세 재검출을 "
                        "먼저 완료하세요."
                    )
                    continue

                if not observation_recapture_complete:
                    print(
                        "r 중간 자세 ArUco 재검출이 성공한 뒤 "
                        "p를 누르세요."
                    )
                    continue

                try:
                    T_target = (
                        compute_slot_target(
                            registration,
                            selected_slot,
                            "approach",
                            current_base_marker,
                        )
                    )
                except KeyError as exc:
                    print("미리보기 실패:", exc)
                    continue

                preview_reference_target = (
                    transform_to_mycobot_coords(
                        T_target
                    )
                )

                compensation = (
                    APPROACH_POSITION_COMPENSATION_MM[
                        selected_slot
                    ]
                )

                # 보정된 접근 좌표를 미세보정 기준으로도 사용한다.
                for axis_index in range(3):
                    preview_reference_target[axis_index] += float(
                        compensation[axis_index]
                    )

                preview_target = [
                    float(value)
                    for value in preview_reference_target
                ]

                preview_slot = selected_slot
                preview_timestamp = time.monotonic()

                current_pose = robot.request(
                    "POSE"
                )

                print()
                print("=" * 66)
                print(
                    f"슬롯 {selected_slot} 접근 목표 미리보기"
                )
                print(
                    "현재 coords:",
                    (
                        np.round(
                            current_pose.get(
                                "coords_mm_deg",
                                [],
                            ),
                            3,
                        ).tolist()
                    ),
                )
                print(
                    "적용 보정 XYZ(mm):",
                    np.round(
                        compensation,
                        3,
                    ).tolist(),
                )
                print(
                    "보정 후 목표 coords:",
                    np.round(
                        preview_target,
                        3,
                    ).tolist(),
                )
                print(
                    "확인 후 x를 누르면 속도 "
                    f"{TEST_SPEED}으로 접근 자세까지만 이동"
                )
                print("=" * 66)

            elif key == ord("x"):
                # 슬롯 1은 Cartesian IK 대신 수동 티칭한 관절각으로 직행한다.
                if selected_slot == 1:
                    if (
                        preview_target is None
                        or preview_slot != selected_slot
                    ):
                        print(
                            "먼저 p로 슬롯 1 접근 목표를 "
                            "미리보기 하세요."
                        )
                        continue

                    print()
                    print("=" * 70)
                    print("슬롯 1 접근 관절각 직행")
                    print(
                        "목표 관절각:",
                        SLOT1_DIRECT_APPROACH_ANGLES,
                    )
                    print(
                        "주의: 운반대가 티칭 당시 위치와 같은지 "
                        "확인하세요."
                    )
                    print("=" * 70)

                    direct_response = robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_ANGLES",
                            "angles": SLOT1_DIRECT_APPROACH_ANGLES,
                            "speed": SLOT1_DIRECT_APPROACH_SPEED,
                            "angle_tolerance_deg": (
                                SLOT1_DIRECT_ANGLE_TOLERANCE_DEG
                            ),
                            "stable_required": (
                                SLOT1_DIRECT_STABLE_REQUIRED
                            ),
                            "timeout_sec": (
                                SLOT1_DIRECT_TIMEOUT_SEC
                            ),
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 1 direct approach angles"
                        ),
                    )

                    print(
                        "슬롯 1 접근 관절각 이동 결과:",
                        direct_response,
                    )

                    if not direct_response.get("ok"):
                        print(
                            "슬롯 1 접근 이동 실패. "
                            "g와 z를 실행하지 마세요."
                        )
                        continue

                    final_pose = robot.request("POSE")
                    print(
                        "슬롯 1 접근 도착 POSE:",
                        final_pose,
                    )
                    print(
                        "접근 자세 도착. 위치를 확인한 뒤 "
                        "g를 누르세요."
                    )
                    continue

                if not observation_recapture_complete:
                    print(
                        "r 중간 자세 ArUco 재검출이 성공하지 않아 "
                        "접근 이동을 차단합니다."
                    )
                    continue

                if (
                    preview_target is None
                    or preview_slot != selected_slot
                ):
                    print(
                        "먼저 p로 현재 슬롯 목표를 "
                        "미리보기 하세요."
                    )
                    continue

                if (
                    time.monotonic()
                    - preview_timestamp
                    > 120.0
                ):
                    print(
                        "미리보기가 오래되었습니다. "
                        "m과 p를 다시 실행하세요."
                    )
                    preview_target = None
                    continue

                current_pose = robot.request("POSE")

                if not current_pose.get("ok"):
                    print("현재 POSE 읽기 실패:", current_pose)
                    continue

                current = [
                    float(v)
                    for v in current_pose["coords_mm_deg"]
                ]
                target = [
                    float(v)
                    for v in preview_target
                ]

                # 현재 위치와 접근 목표 사이의 직선거리(mm)를 계산한다.
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

                # 슬롯 2에서 목표 근처까지 이미 이동한 상태라면
                # 중간 경로를 다시 만들지 않는다.
                # 멀리 뻗은 자세에서 Z를 다시 높이려 하면 IK가 실패할 수 있으므로
                # 저장된 접근 자세만 짧게 재명령한다.
                if distance_to_target_mm <= DIRECT_APPROACH_MAX_DISTANCE_MM:
                    print(
                        "관측 자세에서 접근 목표까지 "
                        f"{distance_to_target_mm:.1f}mm로 가깝습니다."
                    )
                    print(
                        "IK가 거부된 +40mm 고상 안전점을 생략하고 "
                        "접근 자세를 직접 요청합니다."
                    )

                    waypoints = [
                        ("가까운 접근 자세 직접 이동", target),
                    ]

                elif (
                    selected_slot == 2
                    and distance_to_target_mm <= 60.0
                ):
                    print(
                        "슬롯 2 목표 근처이므로 "
                        "중간 경로 없이 접근 자세만 다시 요청합니다."
                    )

                    waypoints = [
                        ("저장된 접근 자세 재시도", target),
                    ]

                else:
                    # 한 번에 목표로 가지 않고 안전 높이를 거쳐 이동한다.
                    # 목표 각도를 높은 Z에서 먼저 적용하면 IK가 풀리지
                    # 않는 경우가 있으므로, XY 이동 중에는 현재 손목 각도를
                    # 유지하고 마지막 단계에서 저장된 목표 각도로 전환한다.
                    if selected_slot == 2:
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

                    if selected_slot == 2:
                        waypoints = [
                            ("중간 안전점", midpoint),
                            ("중간점 높이 변경", lower_midpoint),
                            ("목표 XY 안전점", xy_safe_point),
                            ("저장된 접근 자세", target),
                        ]

                    elif selected_slot == 3:
                        # 슬롯 3은 초기 자세에서 목표까지의 XY 이동거리가 길어
                        # 한 번에 중간점으로 이동할 때 손목 자세 오차가 커질 수 있다.
                        # XY 이동을 25%, 50%, 75%, 100%로 나누고,
                        # 각 중간 단계에서는 해당 시점의 실제 손목 자세를 유지한다.
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

                        waypoints = [
                            ("분할 안전점 25%", split_25),
                            ("분할 안전점 50%", split_50),
                            ("분할 안전점 75%", split_75),
                            ("목표 XY 안전점", xy_safe_point),
                            ("저장된 접근 자세", target),
                        ]

                    else:
                        waypoints = [
                            ("중간 안전점", midpoint),
                            ("목표 XY 안전점", xy_safe_point),
                            ("저장된 접근 자세", target),
                        ]

                print()
                print("=" * 66)
                print(
                    f"슬롯 {selected_slot} 단계 이동 시작 "
                    f"(속도 {TEST_SPEED})"
                )
                for name, coords in waypoints:
                    print(
                        name,
                        ":",
                        np.round(coords, 3).tolist(),
                    )
                print("=" * 66)

                all_ok = True
                total_steps = len(waypoints)

                for step_index, (name, coords) in enumerate(
                    waypoints,
                    start=1,
                ):
                    print()
                    print(
                        f"[{step_index}/{total_steps}] {name} 이동 요청"
                    )

                    command_coords = [
                        float(value)
                        for value in coords
                    ]

                    # 슬롯 3의 분할 안전점에서는 매 단계 직전의
                    # 실제 손목 자세를 다시 읽어 그대로 유지한다.
                    # 마지막 단계에서만 등록된 접근 자세의 회전값을 적용한다.
                    if (
                        selected_slot == 3
                        and name != "저장된 접근 자세"
                    ):
                        live_pose = robot.request("POSE")

                        if not live_pose.get("ok"):
                            print(
                                "중간 경로 현재 POSE 읽기 실패:",
                                live_pose,
                            )
                            all_ok = False
                            break

                        live_coords = [
                            float(value)
                            for value in live_pose[
                                "coords_mm_deg"
                            ]
                        ]

                        command_coords[3:] = live_coords[3:]

                        print(
                            "현재 손목 자세 유지:",
                            np.round(
                                command_coords[3:],
                                3,
                            ).tolist(),
                        )

                    response = robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_COORDS",
                            "coords": command_coords,
                            "speed": TEST_SPEED,
                            "mode": MOVE_MODE,
                            "position_tolerance_mm": (
                                COARSE_MOVE_POSITION_TOLERANCE_MM
                            ),
                            "rotation_tolerance_deg": (
                                COARSE_MOVE_ROTATION_TOLERANCE_DEG
                            ),
                            "stable_required": (
                                COARSE_MOVE_STABLE_REQUIRED
                            ),
                            "timeout_sec": COARSE_MOVE_TIMEOUT_SEC,
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            f"{step_index}/{total_steps} {name}"
                        ),
                    )

                    print(
                        f"[{step_index}/{total_steps}] 결과:", response,
                    )

                    # 슬롯 3의 높은 안전 중간점에서는 최종 손목 방향보다
                    # 중간 위치 도달 여부가 중요하다. 로봇이 목표 XYZ 근처에
                    # 도착했고 회전 변화가 제한 범위 안이면, 실제 도착 RPY를
                    # 다음 단계에서 다시 읽어 이어간다.
                    slot3_position_only_accept = False

                    if (
                        selected_slot == 3
                        and name != "저장된 접근 자세"
                        and not response.get("ok")
                        and response.get("error")
                        == (
                            "목표 자세에 도달하지 못했습니다. "
                            "다음 단계 이동을 차단합니다."
                        )
                    ):
                        intermediate_position_error = response.get(
                            "final_position_error_mm"
                        )
                        intermediate_rotation_error = response.get(
                            "final_rotation_error_deg"
                        )

                        if (
                            intermediate_position_error is not None
                            and intermediate_rotation_error is not None
                            and float(intermediate_position_error)
                            <= SLOT3_INTERMEDIATE_POSITION_ACCEPT_MM
                            and float(intermediate_rotation_error)
                            <= SLOT3_INTERMEDIATE_ROTATION_ACCEPT_DEG
                        ):
                            slot3_position_only_accept = True
                            print(
                                "슬롯 3 안전 중간점 위치우선 통과:",
                                f"위치오차 {float(intermediate_position_error):.2f}mm,",
                                f"회전오차 {float(intermediate_rotation_error):.2f}deg",
                            )
                            print(
                                "다음 단계에서 실제 손목 자세를 다시 읽어 "
                                "경로를 계속합니다."
                            )

                    if (
                        not response.get("ok")
                        and not slot3_position_only_accept
                    ):
                        print(
                            "단계 이동 실패. 다음 단계는 실행하지 않습니다."
                        )
                        all_ok = False
                        break

                    time.sleep(INTER_WAYPOINT_SETTLE_SEC)

                if all_ok:
                    print()
                    print(
                        "슬롯 접근 coarse 이동 완료"
                    )

                    if preview_reference_target is not None:
                        settle_pose = robot.request("POSE")
                        settle_current = settle_pose.get(
                            "coords_mm_deg"
                        )

                        if settle_current is not None:
                            (
                                settle_pos_error,
                                settle_rot_error,
                            ) = pose_error(
                                [
                                    float(value)
                                    for value in settle_current
                                ],
                                preview_reference_target,
                            )

                            print(
                                "원래 접근 기준 coarse 오차:",
                                f"{settle_pos_error:.2f}mm,",
                                f"{settle_rot_error:.2f}deg",
                            )

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
                                    preview_reference_target,
                                    dtype=np.float64,
                                )
                                previous_command = np.asarray(
                                    preview_target,
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

                                if selected_slot == 1:
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
                                print(
                                    "approach 미세 명령 목표:",
                                    np.round(
                                        fine_target,
                                        3,
                                    ).tolist(),
                                )

                                settle_response = (
                                    robot_request_with_gui_pump(
                                        robot=robot,
                                        payload={
                                            "command": "MOVE_COORDS",
                                            "coords": (
                                                fine_target.tolist()
                                            ),
                                            "speed": (
                                                FINE_SETTLE_SPEED
                                            ),
                                            "mode": MOVE_MODE,
                                            "position_tolerance_mm": (
                                                FINE_MOVE_POSITION_TOLERANCE_MM
                                            ),
                                            "rotation_tolerance_deg": (
                                                FINE_MOVE_ROTATION_TOLERANCE_DEG
                                            ),
                                            "stable_required": (
                                                FINE_MOVE_STABLE_REQUIRED
                                            ),
                                            "timeout_sec": (
                                                FINE_MOVE_TIMEOUT_SEC
                                            ),
                                        },
                                        window_name=(
                                            "Slot Approach and Grasp Test"
                                        ),
                                        display_frame=display,
                                        status_text=(
                                            "approach residual correction"
                                        ),
                                    )
                                )

                                print(
                                    "approach 오차벡터 보정 결과:",
                                    settle_response,
                                )

                                final_settle = (
                                    settle_response.get(
                                        "final_coords_mm_deg"
                                    )
                                )

                                if final_settle is not None:
                                    (
                                        final_pos_error,
                                        final_rot_error,
                                    ) = pose_error(
                                        [
                                            float(value)
                                            for value in final_settle
                                        ],
                                        preview_reference_target,
                                    )
                                    print(
                                        "원래 접근 기준 최종 오차:",
                                        f"{final_pos_error:.2f}mm,",
                                        f"{final_rot_error:.2f}deg",
                                    )

                                    if (
                                        final_pos_error
                                        >= settle_pos_error
                                    ):
                                        print(
                                            "주의: 미세 보정 후 위치오차가 "
                                            "줄지 않았습니다."
                                        )

                    print(
                        "중심과 충돌 여유를 확인한 뒤 "
                        "g를 눌러 집기 목표만 확인하세요."
                    )

                preview_target = None
                preview_reference_target = None

            elif key == ord("g"):
                if current_base_marker is None:
                    print(
                        "먼저 READY에서 m으로 "
                        "현재 마커 위치를 저장하세요."
                    )
                    continue

                current_pose = robot.request("POSE")

                if not current_pose.get("ok"):
                    print(
                        "현재 POSE 읽기 실패:",
                        current_pose,
                    )
                    continue

                current = [
                    float(value)
                    for value in current_pose[
                        "coords_mm_deg"
                    ]
                ]

                try:
                    T_approach_reference = (
                        compute_slot_target(
                            registration,
                            selected_slot,
                            "approach",
                            current_base_marker,
                        )
                    )
                    T_grasp_reference = (
                        compute_slot_target(
                            registration,
                            selected_slot,
                            "grasp",
                            current_base_marker,
                        )
                    )
                except KeyError as exc:
                    print("집기 미리보기 실패:", exc)
                    continue

                approach_reference = (
                    transform_to_mycobot_coords(
                        T_approach_reference
                    )
                )
                grasp_reference = (
                    transform_to_mycobot_coords(
                        T_grasp_reference
                    )
                )

                approach_compensation = (
                    APPROACH_POSITION_COMPENSATION_MM[
                        selected_slot
                    ]
                )

                for axis_index in range(3):
                    approach_reference[axis_index] += float(
                        approach_compensation[axis_index]
                    )

                grasp_desired_reference = [
                    float(value)
                    for value in grasp_reference
                ]
                grasp_desired_reference[2] += float(
                    GRASP_PHYSICAL_DEPTH_OFFSET_MM[
                        selected_slot
                    ]
                )

                grasp_compensation = (
                    GRASP_POSITION_COMPENSATION_MM[
                        selected_slot
                    ]
                )

                # 집기 미세보정 기준에도 XYZ 보정을 포함한다.
                for axis_index in range(3):
                    grasp_desired_reference[axis_index] += float(
                        grasp_compensation[axis_index]
                    )

                # 실제 명령도 반드시 깊이·XYZ 보정이 적용된 기준에서 만든다.
                # 기존 v28은 grasp_desired_reference에는 깊이 보정을 넣고
                # grasp_command는 보정 전 grasp_reference에서 만들어
                # 화면의 '실제 집기 깊이 목표'와 로봇 명령 Z가 달랐다.
                grasp_command = [
                    float(value)
                    for value in grasp_desired_reference
                ]

                grasp_rotation_compensation = (
                    GRASP_ROTATION_COMPENSATION_DEG[
                        selected_slot
                    ]
                )

                for axis_index in range(3):
                    compensated_angle = (
                        grasp_command[3 + axis_index]
                        + float(
                            grasp_rotation_compensation[
                                axis_index
                            ]
                        )
                    )

                    # 각도를 -180~180도 범위로 정규화한다.
                    grasp_command[3 + axis_index] = (
                        compensated_angle + 180.0
                    ) % 360.0 - 180.0

                (
                    approach_position_error,
                    approach_rotation_error,
                ) = pose_error(
                    current,
                    approach_reference,
                )

                (
                    grasp_step_mm,
                    grasp_rotation_step_deg,
                ) = pose_error(
                    current,
                    grasp_command,
                )

                print()
                print("=" * 70)
                print(
                    f"슬롯 {selected_slot} 집기 목표 미리보기"
                )
                print(
                    "현재 coords:",
                    np.round(
                        current,
                        3,
                    ).tolist(),
                )
                print(
                    "원래 접근 기준 coords:",
                    np.round(
                        approach_reference,
                        3,
                    ).tolist(),
                )
                print(
                    "현재→접근 기준 위치오차:",
                    f"{approach_position_error:.2f} mm",
                )
                print(
                    "현재→접근 기준 회전오차:",
                    f"{approach_rotation_error:.2f} deg",
                )
                print(
                    "저장된 원래 grasp:",
                    np.round(
                        grasp_reference,
                        3,
                    ).tolist(),
                )
                print(
                    "실제 집기 깊이 목표:",
                    np.round(
                        grasp_desired_reference,
                        3,
                    ).tolist(),
                )
                print(
                    "grasp 전용 보정 XYZ(mm):",
                    np.round(
                        grasp_compensation,
                        3,
                    ).tolist(),
                )
                print(
                    "grasp 전용 회전 보정 RPY(deg):",
                    np.round(
                        grasp_rotation_compensation,
                        3,
                    ).tolist(),
                )
                print(
                    "실제 명령할 grasp:",
                    np.round(
                        grasp_command,
                        3,
                    ).tolist(),
                )
                print(
                    "현재→grasp 이동거리:",
                    f"{grasp_step_mm:.2f} mm",
                )
                print(
                    "현재→grasp 최대 회전 변화:",
                    f"{grasp_rotation_step_deg:.2f} deg",
                )

                safe_preview = True

                if (
                    approach_position_error
                    > MAX_APPROACH_POSITION_ERROR_MM
                    or approach_rotation_error
                    > MAX_APPROACH_ROTATION_ERROR_DEG
                ):
                    safe_preview = False
                    print(
                        "차단: 현재 로봇이 저장된 접근 자세 "
                        "근처에 있지 않습니다."
                    )

                if (
                    grasp_step_mm
                    > MAX_GRASP_TRANSLATION_STEP_MM
                ):
                    safe_preview = False
                    print(
                        "차단: grasp까지 이동거리가 "
                        f"{MAX_GRASP_TRANSLATION_STEP_MM:.0f}mm를 "
                        "초과합니다."
                    )

                if (
                    grasp_rotation_step_deg
                    > MAX_GRASP_ROTATION_STEP_DEG
                ):
                    safe_preview = False
                    print(
                        "차단: grasp까지 회전 변화가 "
                        f"{MAX_GRASP_ROTATION_STEP_DEG:.0f}도를 "
                        "초과합니다."
                    )

                if safe_preview:
                    grasp_preview_target = grasp_command
                    grasp_reference_target = (
                        grasp_desired_reference
                    )
                    grasp_preview_slot = selected_slot
                    grasp_preview_timestamp = (
                        time.monotonic()
                    )
                    print()
                    print(
                        "미리보기 통과."
                    )
                    print(
                        "집게를 닫지 않은 상태, 손을 치운 상태에서만 "
                        "z를 누르세요."
                    )
                else:
                    grasp_preview_target = None
                    grasp_reference_target = None
                    grasp_preview_slot = None

                print("=" * 70)

            elif key == ord("z"):
                # 슬롯 1은 Cartesian IK 대신 수동 티칭한 관절각으로 직행한다.
                if selected_slot == 1:
                    if (
                        grasp_preview_target is None
                        or grasp_preview_slot != selected_slot
                    ):
                        print(
                            "먼저 g로 슬롯 1 집기 목표를 "
                            "미리보기 하세요."
                        )
                        continue

                    if (
                        time.monotonic()
                        - grasp_preview_timestamp
                        > GRASP_PREVIEW_VALID_SEC
                    ):
                        print(
                            "집기 미리보기가 오래되었습니다. "
                            "g를 다시 누르세요."
                        )
                        grasp_preview_target = None
                        grasp_reference_target = None
                        grasp_preview_slot = None
                        continue

                    print()
                    print("=" * 70)
                    print("슬롯 1 집기 관절각 직행")
                    print(
                        "목표 관절각:",
                        SLOT1_DIRECT_GRASP_ANGLES,
                    )
                    print(
                        "주의: 이번 단계에서는 집게를 "
                        "닫지 않습니다."
                    )
                    print("=" * 70)

                    direct_response = robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_ANGLES",
                            "angles": SLOT1_DIRECT_GRASP_ANGLES,
                            "speed": SLOT1_DIRECT_GRASP_SPEED,
                            "angle_tolerance_deg": (
                                SLOT1_DIRECT_ANGLE_TOLERANCE_DEG
                            ),
                            "stable_required": (
                                SLOT1_DIRECT_STABLE_REQUIRED
                            ),
                            "timeout_sec": (
                                SLOT1_DIRECT_TIMEOUT_SEC
                            ),
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 1 direct grasp angles"
                        ),
                    )

                    print(
                        "슬롯 1 집기 관절각 이동 결과:",
                        direct_response,
                    )

                    if not direct_response.get("ok"):
                        print(
                            "슬롯 1 집기 이동 실패. "
                            "집게를 닫지 마세요."
                        )
                        continue

                    final_pose = robot.request("POSE")
                    print(
                        "슬롯 1 집기 도착 POSE:",
                        final_pose,
                    )
                    print(
                        "집기 자세 도착. 집게는 열린 상태입니다."
                    )
                    continue

                if (
                    grasp_preview_target is None
                    or grasp_reference_target is None
                    or grasp_preview_slot != selected_slot
                ):
                    print(
                        "먼저 g를 눌러 현재 슬롯의 "
                        "집기 목표와 안전 검사를 확인하세요."
                    )
                    continue

                if (
                    time.monotonic()
                    - grasp_preview_timestamp
                    > GRASP_PREVIEW_VALID_SEC
                ):
                    print(
                        "집기 미리보기가 오래되었습니다. "
                        "g를 다시 누르세요."
                    )
                    grasp_preview_target = None
                    grasp_reference_target = None
                    grasp_preview_slot = None
                    continue

                current_pose = robot.request("POSE")

                if not current_pose.get("ok"):
                    print(
                        "현재 POSE 읽기 실패:",
                        current_pose,
                    )
                    continue

                current = [
                    float(value)
                    for value in current_pose[
                        "coords_mm_deg"
                    ]
                ]

                T_approach_reference = (
                    compute_slot_target(
                        registration,
                        selected_slot,
                        "approach",
                        current_base_marker,
                    )
                )
                approach_reference = (
                    transform_to_mycobot_coords(
                        T_approach_reference
                    )
                )

                (
                    approach_position_error,
                    approach_rotation_error,
                ) = pose_error(
                    current,
                    approach_reference,
                )
                (
                    grasp_step_mm,
                    grasp_rotation_step_deg,
                ) = pose_error(
                    current,
                    grasp_preview_target,
                )

                if (
                    approach_position_error
                    > MAX_APPROACH_POSITION_ERROR_MM
                    or approach_rotation_error
                    > MAX_APPROACH_ROTATION_ERROR_DEG
                ):
                    print(
                        "집기 이동 차단: 현재 자세가 "
                        "접근 자세에서 벗어났습니다."
                    )
                    print(
                        "위치오차:",
                        f"{approach_position_error:.2f}mm,",
                        "회전오차:",
                        f"{approach_rotation_error:.2f}deg",
                    )
                    continue

                if (
                    grasp_step_mm
                    > MAX_GRASP_TRANSLATION_STEP_MM
                    or grasp_rotation_step_deg
                    > MAX_GRASP_ROTATION_STEP_DEG
                ):
                    print(
                        "집기 이동 차단: 이동량 안전 기준을 "
                        "초과했습니다."
                    )
                    continue

                print()
                print("=" * 70)
                print(
                    f"슬롯 {selected_slot} 집기 자세 저속 이동 시작"
                )
                print(
                    "속도:",
                    GRASP_TEST_SPEED,
                )
                print(
                    "목표:",
                    np.round(
                        grasp_preview_target,
                        3,
                    ).tolist(),
                )
                print(
                    "주의: 이번 단계에서는 집게를 닫지 않습니다."
                )
                print("=" * 70)

                response = robot_request_with_gui_pump(
                    robot=robot,
                    payload={
                        "command": "MOVE_COORDS",
                        "coords": grasp_preview_target,
                        "speed": GRASP_TEST_SPEED,
                        "mode": MOVE_MODE,
                        "position_tolerance_mm": (
                            COARSE_MOVE_POSITION_TOLERANCE_MM
                        ),
                        "rotation_tolerance_deg": (
                            COARSE_MOVE_ROTATION_TOLERANCE_DEG
                        ),
                        "stable_required": (
                            COARSE_MOVE_STABLE_REQUIRED
                        ),
                        "timeout_sec": COARSE_MOVE_TIMEOUT_SEC,
                    },
                    window_name=(
                        "Slot Approach and Grasp Test"
                    ),
                    display_frame=display,
                    status_text=(
                        f"slot {selected_slot} grasp move"
                    ),
                )

                print(
                    "집기 자세 이동 결과:",
                    response,
                )

                if response.get("ok"):
                    final_coords = response.get(
                        "final_coords_mm_deg"
                    )

                    if final_coords is None:
                        final_pose = robot.request(
                            "POSE"
                        )
                        final_coords = final_pose.get(
                            "coords_mm_deg"
                        )

                    if final_coords is not None:
                        (
                            reference_position_error,
                            reference_rotation_error,
                        ) = pose_error(
                            [
                                float(value)
                                for value in final_coords
                            ],
                            grasp_reference_target,
                        )

                        print(
                            "실제 집기 깊이 목표 기준 coarse 위치오차:",
                            f"{reference_position_error:.2f} mm",
                        )
                        print(
                            "실제 집기 깊이 목표 기준 coarse 회전오차:",
                            f"{reference_rotation_error:.2f} deg",
                        )

                        should_settle = (
                            selected_slot not in (2, 3)
                            and FINE_SETTLE_MIN_POSITION_MM
                            < reference_position_error
                            <= FINE_SETTLE_MAX_POSITION_MM
                            and reference_rotation_error
                            <= FINE_SETTLE_MAX_ROTATION_DEG
                        )

                        if (
                            selected_slot in (2, 3)
                            and reference_position_error
                            > FINE_SETTLE_MIN_POSITION_MM
                        ):
                            print(
                                f"슬롯 {selected_slot}은 X·Y 이동 시 Z가 함께 변하는 "
                                "자세 결합을 방지하기 위해 집기 자동 미세보정을 "
                                "생략합니다."
                            )

                        if should_settle:
                            actual_array = np.asarray(
                                final_coords,
                                dtype=np.float64,
                            )
                            reference_array = np.asarray(
                                grasp_reference_target,
                                dtype=np.float64,
                            )
                            previous_command = np.asarray(
                                grasp_preview_target,
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

                            if selected_slot == 1:
                                correction_xyz[2] = 0.0

                            if selected_slot == 4:
                                original_y_correction = float(
                                    correction_xyz[1]
                                )
                                correction_xyz[1] *= (
                                    SLOT4_GRASP_Y_FINE_GAIN
                                )
                                print(
                                    "슬롯 4 grasp Y 미세보정 축소:",
                                    f"{original_y_correction:.3f} → "
                                    f"{float(correction_xyz[1]):.3f}mm "
                                    f"(gain {SLOT4_GRASP_Y_FINE_GAIN:.2f})",
                                )

                            fine_target = previous_command.copy()
                            fine_target[:3] += correction_xyz

                            # XYZ 미세보정 중에도 앞에서 적용한
                            # 슬롯별 회전 보정값은 그대로 유지한다.
                            fine_target[3:] = previous_command[3:]

                            print(
                                "grasp 남은 오차벡터 XYZ(mm):",
                                np.round(
                                    reference_array[:3]
                                    - actual_array[:3],
                                    3,
                                ).tolist(),
                            )
                            print(
                                "grasp 미세 명령 XYZ 보정:",
                                np.round(
                                    correction_xyz,
                                    3,
                                ).tolist(),
                            )
                            print(
                                "grasp 미세 명령 목표:",
                                np.round(
                                    fine_target,
                                    3,
                                ).tolist(),
                            )

                            settle_response = (
                                robot_request_with_gui_pump(
                                    robot=robot,
                                    payload={
                                        "command": "MOVE_COORDS",
                                        "coords": (
                                            fine_target.tolist()
                                        ),
                                        "speed": (
                                            FINE_SETTLE_SPEED
                                        ),
                                        "mode": MOVE_MODE,
                                        "position_tolerance_mm": (
                                            FINE_MOVE_POSITION_TOLERANCE_MM
                                        ),
                                        "rotation_tolerance_deg": (
                                            FINE_MOVE_ROTATION_TOLERANCE_DEG
                                        ),
                                        "stable_required": (
                                            FINE_MOVE_STABLE_REQUIRED
                                        ),
                                        "timeout_sec": (
                                            FINE_MOVE_TIMEOUT_SEC
                                        ),
                                    },
                                    window_name=(
                                        "Slot Approach and Grasp Test"
                                    ),
                                    display_frame=display,
                                    status_text=(
                                        "grasp residual correction"
                                    ),
                                )
                            )

                            print(
                                "grasp 오차벡터 보정 결과:",
                                settle_response,
                            )

                            settle_final = (
                                settle_response.get(
                                    "final_coords_mm_deg"
                                )
                            )

                            if settle_final is not None:
                                (
                                    new_position_error,
                                    new_rotation_error,
                                ) = pose_error(
                                    [
                                        float(value)
                                        for value in settle_final
                                    ],
                                    grasp_reference_target,
                                )

                                if (
                                    new_position_error
                                    >= reference_position_error
                                ):
                                    print(
                                        "주의: 미세 보정 후 위치오차가 "
                                        "줄지 않았습니다."
                                    )

                                reference_position_error = (
                                    new_position_error
                                )
                                reference_rotation_error = (
                                    new_rotation_error
                                )

                        print(
                            "실제 집기 깊이 목표 기준 최종 위치오차:",
                            f"{reference_position_error:.2f} mm",
                        )
                        print(
                            "실제 집기 깊이 목표 기준 최종 회전오차:",
                            f"{reference_rotation_error:.2f} deg",
                        )

                    last_grasp_reference_target = [
                        float(value)
                        for value in grasp_reference_target
                    ]
                    last_grasp_move_slot = selected_slot

                    print()
                    print(
                        "집기 자세 도착. 집게는 열린 상태이며 닫기 동작은 "
                        "차단되어 있습니다."
                    )
                    print(
                        "타이어·집게·운반대 간섭과 중심·높이를 확인한 뒤 "
                        "c로 현재 pose를 출력하세요."
                    )
                else:
                    last_grasp_reference_target = None
                    last_grasp_move_slot = None
                    print(
                        "집기 자세 이동 실패. 집게를 닫지 마세요."
                    )

                grasp_preview_target = None
                grasp_reference_target = None
                grasp_preview_slot = None

            elif key == ord("b"):
                if selected_slot != AUTO_PICK_TEST_SLOT:
                    print(
                        f"자동 닫기·상승은 현재 슬롯 "
                        f"{AUTO_PICK_TEST_SLOT} 시험에서만 허용합니다."
                    )
                    pick_confirm_until = 0.0
                    continue

                if (
                    last_grasp_reference_target is None
                    or last_grasp_move_slot != AUTO_PICK_TEST_SLOT
                ):
                    print(
                        "먼저 슬롯 4에서 g → z 집기 자세 이동을 "
                        "성공시킨 뒤 b를 누르세요."
                    )
                    pick_confirm_until = 0.0
                    continue

                check_pose = robot.request("POSE")

                if not check_pose.get("ok"):
                    print("현재 POSE 읽기 실패:", check_pose)
                    pick_confirm_until = 0.0
                    continue

                check_coords = [
                    float(value)
                    for value in check_pose["coords_mm_deg"]
                ]
                (
                    pick_position_error,
                    pick_rotation_error,
                ) = pose_error(
                    check_coords,
                    last_grasp_reference_target,
                )

                print()
                print("슬롯 4 자동 집기 전 최종 안전 확인")
                print(
                    "집기 기준 위치오차:",
                    f"{pick_position_error:.2f}mm",
                )
                print(
                    "집기 기준 회전오차:",
                    f"{pick_rotation_error:.2f}deg",
                )

                if (
                    pick_position_error
                    > AUTO_PICK_MAX_POSITION_ERROR_MM
                    or pick_rotation_error
                    > AUTO_PICK_MAX_ROTATION_ERROR_DEG
                ):
                    print(
                        "자동 집기 차단: 현재 자세가 집기 기준에서 "
                        "너무 멀어졌습니다."
                    )
                    print(
                        "HOME 복귀 후 4 → m → p → x → g → z를 "
                        "다시 실행하세요."
                    )
                    pick_confirm_until = 0.0
                    continue

                now = time.monotonic()

                if now > pick_confirm_until:
                    pick_confirm_until = (
                        now + AUTO_PICK_CONFIRM_SEC
                    )
                    print(
                        "아직 실행하지 않았습니다. 집게 중심, 손, 케이블, "
                        "운반대 간섭을 다시 확인하세요."
                    )
                    print(
                        f"{AUTO_PICK_CONFIRM_SEC:.0f}초 안에 b를 "
                        "한 번 더 누르면 그리퍼를 닫고 "
                        f"Z축으로 {AUTO_LIFT_DISTANCE_MM:.0f}mm 상승합니다."
                    )
                    continue

                pick_confirm_until = 0.0
                open_confirm_until = 0.0

                print()
                print("=" * 70)
                print("슬롯 4 그리퍼 닫기 시작")
                print(
                    "주의: 손과 물체를 모두 로봇 동작 범위에서 "
                    "치운 상태여야 합니다."
                )
                print("=" * 70)

                close_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "GRIPPER_CLOSE",
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 4 gripper close"
                        ),
                    )
                )

                print(
                    "그리퍼 닫기 결과:",
                    close_response,
                )

                if not close_response.get("ok"):
                    print(
                        "그리퍼 닫기 실패. 상승하지 않습니다."
                    )
                    continue

                lift_start_pose = robot.request("POSE")

                if not lift_start_pose.get("ok"):
                    print(
                        "닫기 후 POSE 읽기 실패. 상승하지 않습니다."
                    )
                    continue

                lift_target = [
                    float(value)
                    for value in lift_start_pose[
                        "coords_mm_deg"
                    ]
                ]
                lift_target[2] += AUTO_LIFT_DISTANCE_MM

                print()
                print("=" * 70)
                print("슬롯 4 수직 상승 시작")
                print(
                    "현재 Z → 목표 Z:",
                    f"{lift_start_pose['coords_mm_deg'][2]:.3f} → "
                    f"{lift_target[2]:.3f}mm",
                )
                print(
                    "XY와 손목 자세는 현재값을 그대로 유지합니다."
                )
                print("=" * 70)

                lift_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_COORDS",
                            "coords": lift_target,
                            "speed": AUTO_LIFT_SPEED,
                            "mode": LINEAR_MOVE_MODE,
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 4 linear vertical lift 30mm"
                        ),
                    )
                )

                print(
                    "슬롯 4 상승 결과:",
                    lift_response,
                )

                if not lift_response.get("ok"):
                    print(
                        "상승 실패. 다음 이동을 차단합니다."
                    )
                    print(
                        "그리퍼는 자동으로 열지 않습니다. "
                        "타이어 상태를 확인한 뒤 물리 정지 또는 "
                        "안전한 조작으로 처리하세요."
                    )
                    print(
                        "STOP:",
                        robot.request("STOP"),
                    )
                    continue

                tire_lifted = True

                # 그리퍼를 닫으면 관절 하중과 유격 때문에 읽힌 pose가
                # 조금 변할 수 있다. 따라서 내려놓을 기준은 닫기 직전
                # 안전검사에서 읽은 실제 집기 pose를 저장한다.
                lift_origin_coords = [
                    float(value)
                    for value in check_coords
                ]
                last_grasp_reference_target = None
                last_grasp_move_slot = None

                print()
                print(
                    "슬롯 4 자동 집기·30mm 상승 완료."
                )
                print(
                    "타이어가 미끄러지거나 심하게 기울지 않는지 "
                    "눈으로 확인하세요."
                )
                print(
                    "공중에서 o를 누르지 마세요. "
                    "원래 슬롯에 내려놓으려면 d를 누르고, "
                    "안전 확인 후 8초 안에 d를 한 번 더 누르세요."
                )

            elif key == ord("d"):
                if not tire_lifted or lift_origin_coords is None:
                    print(
                        "현재 내려놓을 타이어 상승 기록이 없습니다. "
                        "먼저 슬롯 4에서 b를 두 번 눌러 상승을 완료하세요."
                    )
                    place_confirm_until = 0.0
                    continue

                now = time.monotonic()

                if now > place_confirm_until:
                    place_confirm_until = (
                        now + AUTO_PLACE_CONFIRM_SEC
                    )
                    print()
                    print(
                        "아직 하강하지 않았습니다. 타이어 아래 원래 슬롯에 "
                        "다른 물체가 없고 경로가 비어 있는지 확인하세요."
                    )
                    print(
                        f"{AUTO_PLACE_CONFIRM_SEC:.0f}초 안에 d를 "
                        "한 번 더 누르면 원래 집기 높이로 하강하고, "
                        "그리퍼를 연 뒤 Z축으로 30mm 후퇴합니다."
                    )
                    continue

                place_confirm_until = 0.0
                pick_confirm_until = 0.0
                open_confirm_until = 0.0

                place_target = [
                    float(value)
                    for value in lift_origin_coords
                ]

                for axis_index in range(3):
                    place_target[axis_index] += float(
                        PLACE_POSITION_OFFSET_MM[axis_index]
                    )

                pre_place_target = [
                    float(value)
                    for value in place_target
                ]
                pre_place_target[2] += PRE_PLACE_HEIGHT_MM

                print()
                print("=" * 70)
                print("슬롯 4 정밀 내려놓기 시작")
                print(
                    "닫기 전 집기 pose:",
                    np.round(
                        lift_origin_coords,
                        3,
                    ).tolist(),
                )
                print(
                    "내려놓기 전용 XYZ 보정(mm):",
                    np.round(
                        PLACE_POSITION_OFFSET_MM,
                        3,
                    ).tolist(),
                )
                print(
                    "10mm 상공 정렬 목표:",
                    np.round(
                        pre_place_target,
                        3,
                    ).tolist(),
                )
                print(
                    "최종 수직 하강 목표:",
                    np.round(
                        place_target,
                        3,
                    ).tolist(),
                )
                print(
                    "정밀 허용오차를 만족하지 못하면 "
                    "그리퍼를 열지 않습니다."
                )
                print("=" * 70)

                pre_place_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_COORDS",
                            "coords": pre_place_target,
                            "speed": AUTO_PLACE_SPEED,
                            "mode": LINEAR_MOVE_MODE,
                            "position_tolerance_mm": (
                                PRE_PLACE_POSITION_TOLERANCE_MM
                            ),
                            "rotation_tolerance_deg": (
                                PRE_PLACE_ROTATION_TOLERANCE_DEG
                            ),
                            "stable_required": (
                                PRE_PLACE_STABLE_REQUIRED
                            ),
                            "timeout_sec": 40.0,
                            "require_tolerance": True,
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 4 align 10mm above place"
                        ),
                    )
                )

                print(
                    "슬롯 4 상공 정렬 결과:",
                    pre_place_response,
                )

                if not pre_place_response.get("ok"):
                    print(
                        "상공 정렬 실패. 하강하거나 그리퍼를 열지 않습니다."
                    )
                    print(
                        "STOP:",
                        robot.request("STOP"),
                    )
                    continue

                place_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_COORDS",
                            "coords": place_target,
                            "speed": AUTO_PLACE_SPEED,
                            "mode": LINEAR_MOVE_MODE,
                            "position_tolerance_mm": (
                                PRECISE_PLACE_POSITION_TOLERANCE_MM
                            ),
                            "rotation_tolerance_deg": (
                                PRECISE_PLACE_ROTATION_TOLERANCE_DEG
                            ),
                            "stable_required": (
                                PRECISE_PLACE_STABLE_REQUIRED
                            ),
                            "timeout_sec": (
                                PRECISE_PLACE_TIMEOUT_SEC
                            ),
                            "require_tolerance": True,
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 4 return to original grasp height"
                        ),
                    )
                )

                print(
                    "슬롯 4 하강 결과:",
                    place_response,
                )

                if not place_response.get("ok"):
                    print(
                        "정밀 하강 실패. 그리퍼를 열지 않습니다."
                    )
                    print(
                        "STOP:",
                        robot.request("STOP"),
                    )
                    continue

                place_final = place_response.get(
                    "final_coords_mm_deg"
                )

                if place_final is None:
                    place_pose = robot.request("POSE")
                    place_final = place_pose.get(
                        "coords_mm_deg"
                    )

                if place_final is None:
                    print(
                        "하강 후 pose를 확인하지 못했습니다. "
                        "그리퍼를 열지 않습니다."
                    )
                    continue

                final_place_y_error = abs(
                    float(place_final[1])
                    - float(place_target[1])
                )

                print(
                    "최종 내려놓기 Y 오차:",
                    f"{final_place_y_error:.3f}mm",
                )

                if (
                    final_place_y_error
                    > PRECISE_PLACE_MAX_Y_ERROR_MM
                ):
                    print(
                        "Y 오차가 1mm를 초과하여 그리퍼 열기를 차단합니다."
                    )
                    print(
                        "타이어를 잡은 상태로 추가 동작하지 말고 "
                        "현재 로그를 확인하세요."
                    )
                    continue

                time.sleep(AUTO_PLACE_SETTLE_SEC)

                print(
                    "정밀 하강 완료. 그리퍼를 엽니다."
                )

                open_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "GRIPPER_OPEN",
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 4 release tire"
                        ),
                    )
                )

                print(
                    "그리퍼 열기 결과:",
                    open_response,
                )

                if not open_response.get("ok"):
                    print(
                        "그리퍼 열기 실패. 후퇴하지 않습니다."
                    )
                    continue

                release_pose = robot.request("POSE")

                if not release_pose.get("ok"):
                    print(
                        "열기 후 POSE 읽기 실패. 자동 후퇴하지 않습니다."
                    )
                    tire_lifted = False
                    lift_origin_coords = None
                    continue

                retreat_target = [
                    float(value)
                    for value in release_pose["coords_mm_deg"]
                ]
                retreat_target[2] += AUTO_LIFT_DISTANCE_MM

                print(
                    "그리퍼를 연 상태로 Z축 30mm 후퇴합니다."
                )

                retreat_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_COORDS",
                            "coords": retreat_target,
                            "speed": AUTO_RETREAT_SPEED,
                            "mode": LINEAR_MOVE_MODE,
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "slot 4 linear retreat after release"
                        ),
                    )
                )

                print(
                    "슬롯 4 후퇴 결과:",
                    retreat_response,
                )

                tire_lifted = False
                lift_origin_coords = None

                if retreat_response.get("ok"):
                    print()
                    print(
                        "슬롯 4 집기 → 상승 → 원위치 하강 → "
                        "열기 → 후퇴 완료."
                    )
                else:
                    print(
                        "그리퍼는 열렸지만 후퇴에 실패했습니다. "
                        "추가 이동하지 말고 현재 상태를 확인하세요."
                    )

            elif key == ord("o"):
                if tire_lifted:
                    print(
                        "그리퍼 열기 차단: 타이어가 공중에 있습니다. "
                        "원래 슬롯으로 내려놓으려면 d를 두 번 누르세요."
                    )
                    open_confirm_until = 0.0
                    continue

                now = time.monotonic()

                if now > open_confirm_until:
                    open_confirm_until = (
                        now + AUTO_OPEN_CONFIRM_SEC
                    )

                    print(
                        "아직 그리퍼를 열지 않았습니다."
                    )
                    print(
                        f"안전한 지지면에 내려놓았는지 확인한 뒤 "
                        f"{AUTO_OPEN_CONFIRM_SEC:.0f}초 안에 o를 "
                        "한 번 더 누르세요."
                    )
                    continue

                open_confirm_until = 0.0
                pick_confirm_until = 0.0

                open_response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "GRIPPER_OPEN",
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            "gripper open"
                        ),
                    )
                )

                print(
                    "그리퍼 열기 결과:",
                    open_response,
                )

                if open_response.get("ok"):
                    tire_lifted = False
                else:
                    print(
                        "그리퍼 열기 실패. 상태를 직접 확인하세요."
                    )

            elif key == ord("y"):
                if selected_slot != 2:
                    print(
                        "y 미세전진은 현재 슬롯 2에서만 허용합니다."
                    )
                    continue

                if (
                    last_grasp_reference_target is None
                    or last_grasp_move_slot != 2
                ):
                    print(
                        "먼저 슬롯 2에서 g → z 집기 coarse 이동을 "
                        "성공시킨 뒤 y를 누르세요."
                    )
                    continue

                start_pose = robot.request("POSE")
                if not start_pose.get("ok"):
                    print("현재 POSE 읽기 실패:", start_pose)
                    continue

                start_coords = [
                    float(value)
                    for value in start_pose["coords_mm_deg"]
                ]
                reference = np.asarray(
                    last_grasp_reference_target,
                    dtype=np.float64,
                )
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
                print(
                    "주의: 집게를 닫지 않고 주변에서 손을 치운 "
                    "상태에서만 실행합니다."
                )
                print("=" * 70)

                if (
                    initial_remaining_y
                    <= SLOT2_SINGLE_Y_TOLERANCE_MM
                    and initial_remaining_y >= 0.0
                ):
                    print(
                        "이미 목표 Y 허용오차 안입니다. "
                        "추가 이동하지 않습니다."
                    )
                    continue

                if initial_remaining_y < 0.0:
                    print(
                        "현재 Y가 목표보다 이미 앞쪽입니다. "
                        "자동으로 뒤로 이동하지 않습니다."
                    )
                    continue

                if (
                    initial_remaining_y
                    > SLOT2_SINGLE_Y_MAX_TOTAL_MM
                ):
                    print(
                        "미세전진 차단: 남은 Y 거리가 ",
                        f"{initial_remaining_y:.2f}mm로 "
                        f"허용값 {SLOT2_SINGLE_Y_MAX_TOTAL_MM:.1f}mm를 "
                        "초과했습니다.",
                    )
                    continue

                baseline = start_array.copy()
                final_coords = start_coords
                correction_ok = True

                for step_index in range(
                    1,
                    SLOT2_SINGLE_Y_MAX_STEPS + 1,
                ):
                    pose_now = robot.request("POSE")
                    if not pose_now.get("ok"):
                        print(
                            "미세전진 중 POSE 읽기 실패:",
                            pose_now,
                        )
                        correction_ok = False
                        break

                    current_coords = [
                        float(value)
                        for value in pose_now["coords_mm_deg"]
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

                    single_response = (
                        robot_request_with_gui_pump(
                            robot=robot,
                            payload={
                                "command": "MOVE_SINGLE_COORD",
                                "axis_id": 2,
                                "value": command_y,
                                "speed": SLOT2_SINGLE_Y_SPEED,
                            },
                            window_name=(
                                "Slot Approach and Grasp Test"
                            ),
                            display_frame=display,
                            status_text=(
                                f"slot 2 fine Y "
                                f"{step_index}/"
                                f"{SLOT2_SINGLE_Y_MAX_STEPS}"
                            ),
                        )
                    )

                    print(
                        "단일 Y축 이동 결과:",
                        single_response,
                    )

                    if not single_response.get("ok"):
                        print(
                            "Y축 미세전진 실패. 즉시 중단하고 "
                            "집게를 닫지 마세요."
                        )
                        correction_ok = False
                        break

                    returned = single_response.get(
                        "final_coords_mm_deg"
                    )
                    if returned is None:
                        print(
                            "Y축 이동 후 좌표를 받지 못했습니다."
                        )
                        correction_ok = False
                        break

                    final_coords = [
                        float(value) for value in returned
                    ]
                    final_array = np.asarray(
                        final_coords,
                        dtype=np.float64,
                    )

                    total_x_drift = abs(
                        final_array[0] - baseline[0]
                    )
                    total_z_drift = abs(
                        final_array[2] - baseline[2]
                    )
                    total_rotation_drift = max(
                        wrapped_angle_error_deg(
                            final_array[index],
                            baseline[index],
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
                        print("STOP:", robot.request("STOP"))
                        correction_ok = False
                        break

                final_pose = robot.request("POSE")
                if final_pose.get("ok"):
                    final_coords = [
                        float(value)
                        for value in final_pose["coords_mm_deg"]
                    ]

                (
                    final_reference_position_error,
                    final_reference_rotation_error,
                ) = pose_error(
                    final_coords,
                    last_grasp_reference_target,
                )
                final_remaining_y = float(
                    reference[1] - float(final_coords[1])
                )

                print()
                print("슬롯 2 Y축 미세전진 종료")
                print(
                    "최종 coords:",
                    np.round(final_coords, 3).tolist(),
                )
                print(
                    "목표까지 남은 Y:",
                    f"{final_remaining_y:.3f} mm",
                )
                print(
                    "실제 집기 목표 기준 최종 위치오차:",
                    f"{final_reference_position_error:.2f} mm",
                )
                print(
                    "실제 집기 목표 기준 최종 회전오차:",
                    f"{final_reference_rotation_error:.2f} deg",
                )

                if correction_ok:
                    print(
                        "수치와 실제 간섭을 눈으로 확인하세요. "
                        "그리퍼는 아직 닫지 마세요."
                    )

            elif key == ord("s"):
                print(
                    "STOP:",
                    robot.request("STOP"),
                )

            elif key == ord("c"):
                print(
                    "현재 POSE:",
                    robot.request("POSE"),
                )

            elif key == ord("q"):
                break

    finally:
        try:
            robot.request("STOP")
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        robot.close()

        print(
            "접근/집기 자세 시험 종료"
        )


if __name__ == "__main__":
    main()