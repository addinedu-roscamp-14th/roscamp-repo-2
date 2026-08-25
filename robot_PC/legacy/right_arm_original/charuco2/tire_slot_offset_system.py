import json
import math
import socket
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np


# ============================================================
# 환경 설정
# ============================================================

ROBOT_IP = "192.168.5.1"
ROBOT_PORT = 5000

CAMERA_DEVICE = "/dev/video2"
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

MODEL_FILE = Path(
    "/home/choiminjoon/slot_pose_registration/"
    "tire_grasp_model.json"
)

MARKER_LENGTH_M = 0.030
DICTIONARY_ID = aruco.DICT_4X4_50
TARGET_ID = 0

# OpenCV에서 사용한 마커 축:
# +X = 사진 기준 오른쪽
# +Y = 사진 기준 위쪽
# +Z = 마커 평면에서 카메라 쪽
#
# 사용자가 측정한 슬롯 1:
# 왼쪽 30 mm, 위쪽 35 mm
#
# 2~4번은 실제 자로 측정하기 전까지 None으로 두어 이동을 차단한다.
SLOT_OFFSETS_MM: Dict[int, Optional[np.ndarray]] = {
    1: np.array([-30.0, 35.0, 0.0], dtype=np.float64),
    2: None,
    3: None,
    4: None,
}

REFERENCE_SLOT = 1
# approach는 grasp의 수직 위가 아니라 로봇 베이스 쪽으로 빠진 자세로 만든다.
# 수직으로 50mm 올리면 팔의 최대 도달 반경을 벗어날 수 있다.
APPROACH_PULLBACK_MM = 40.0
APPROACH_LIFT_MM = 20.0

STABLE_BUFFER_SIZE = 12
MIN_STABLE_SAMPLES = 5
MAX_TRANSLATION_SPREAD_MM = 3.0
MAX_ROTATION_SPREAD_DEG = 2.0
MAX_REPROJECTION_ERROR_PX = 2.5
LOST_LIMIT = 5

APPROACH_SPEED = 4
GRASP_SPEED = 2
MOVE_MODE = 0
SAFE_Z_MARGIN_MM = 40.0

# 기존 반복 시험에서 확인한 명령 보정값.
# 새 모델을 저장한 뒤 슬롯 1에서 다시 검증한다.
# 새 ArUco-슬롯 중심 모델은 과거 수동 저장 좌표와 기준이 다르므로
# 기존 경험 보정값을 그대로 적용하지 않는다.
# 먼저 0 보정으로 재현성을 확인한 뒤 새로 측정해 보정한다.
APPROACH_COMMAND_COMPENSATION_BASE_MM = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float64,
)
GRASP_COMMAND_COMPENSATION_BASE_MM = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float64,
)

MAX_APPROACH_POSITION_ERROR_MM = 20.0
MAX_APPROACH_ROTATION_ERROR_DEG = 10.0
MAX_GRASP_TRANSLATION_STEP_MM = 120.0
MAX_GRASP_ROTATION_STEP_DEG = 35.0
PREVIEW_VALID_SEC = 120.0

# 중간 보간 경유점은 최종 작업 자세가 아니므로
# 위치는 충분히 가까운데 Euler 각도 한 축만 조금 벗어난 경우
# 다음 경유점으로 계속 진행한다.
INTERMEDIATE_SOFT_POSITION_MM = 10.0
INTERMEDIATE_SOFT_ROTATION_DEG = 15.0

WINDOW_NAME = "ArUco Tire Slot Geometry System"


# ============================================================
# 변환 유틸리티
# ============================================================

def make_transform(
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(
        R,
        dtype=np.float64,
    ).reshape(3, 3)
    T[:3, 3] = np.asarray(
        t,
        dtype=np.float64,
    ).reshape(3)
    return T


def translation_transform_mm(
    xyz_mm: np.ndarray,
) -> np.ndarray:
    return make_transform(
        np.eye(3, dtype=np.float64),
        np.asarray(
            xyz_mm,
            dtype=np.float64,
        ).reshape(3) / 1000.0,
    )


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
    t_m = np.array(
        [x, y, z],
        dtype=np.float64,
    ) / 1000.0
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
    return math.degrees(math.acos(value))


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
    return position_error_mm, rotation_error_deg


# ============================================================
# 파일 로드/저장
# ============================================================

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


def load_model() -> Optional[Dict[str, Any]]:
    if not MODEL_FILE.exists():
        return None

    try:
        model = json.loads(
            MODEL_FILE.read_text(
                encoding="utf-8",
            )
        )
        np.asarray(
            model["T_tire_gripper"],
            dtype=np.float64,
        ).reshape(4, 4)
        return model
    except Exception as exc:
        print("기존 모델 읽기 실패:", exc)
        return None


def save_model(
    T_tire_gripper: np.ndarray,
    robot_pose: Dict[str, Any],
    T_base_marker: np.ndarray,
) -> Dict[str, Any]:
    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = {
        "saved_at": datetime.now().isoformat(
            timespec="milliseconds"
        ),
        "reference_slot": REFERENCE_SLOT,
        "reference_slot_offset_mm": (
            SLOT_OFFSETS_MM[REFERENCE_SLOT]
        ).tolist(),
        "approach_pullback_mm": APPROACH_PULLBACK_MM,
        "approach_lift_mm": APPROACH_LIFT_MM,
        "T_tire_gripper": (
            T_tire_gripper.tolist()
        ),
        "tire_relative_gripper_xyz_mm": (
            T_tire_gripper[:3, 3]
            * 1000.0
        ).tolist(),
        "tire_relative_gripper_rpy_deg": (
            rotation_matrix_to_rpy_deg(
                T_tire_gripper[:3, :3]
            ).tolist()
        ),
        "teach_robot_coords_mm_deg": robot_pose[
            "coords_mm_deg"
        ],
        "teach_robot_angles_deg": robot_pose[
            "angles_deg"
        ],
        "teach_base_marker_xyz_mm": (
            T_base_marker[:3, 3]
            * 1000.0
        ).tolist(),
    }

    MODEL_FILE.write_text(
        json.dumps(
            model,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


# ============================================================
# 로봇 TCP 클라이언트
# ============================================================

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
            (line.strip() + "\n").encode(
                "utf-8"
            )
        )

        while "\n" not in self.buffer:
            chunk = self.sock.recv(4096)

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
    display_frame: np.ndarray,
    status_text: str,
) -> Dict[str, Any]:
    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def worker() -> None:
        try:
            result_box["response"] = (
                robot.request(payload)
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
            (FRAME_WIDTH - 10, 185),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            wait_frame,
            f"ROBOT MOVING{dots}",
            (25, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            wait_frame,
            status_text,
            (25, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(
            WINDOW_NAME,
            wait_frame,
        )
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


# ============================================================
# ArUco
# ============================================================

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
        return detector.detectMarkers(gray)

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

    R_mean = average_rotation(rotations)
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
        "translation_spread_mm": (
            translation_spread_mm
        ),
        "rotation_spread_deg": (
            rotation_spread_deg
        ),
        "reprojection_error_px": (
            reprojection_error_px
        ),
    }


def select_target_marker(
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


# ============================================================
# 슬롯 모델
# ============================================================

def require_slot_offset(
    slot_number: int,
) -> np.ndarray:
    offset = SLOT_OFFSETS_MM.get(
        slot_number
    )

    if offset is None:
        raise ValueError(
            f"슬롯 {slot_number} 중심 오프셋이 아직 입력되지 않았습니다."
        )

    return np.asarray(
        offset,
        dtype=np.float64,
    ).reshape(3)


def teach_tire_gripper_model(
    robot: RobotClient,
    T_base_marker: np.ndarray,
) -> Dict[str, Any]:
    slot_offset_mm = require_slot_offset(
        REFERENCE_SLOT
    )
    robot_pose = robot.request("POSE")

    if not robot_pose.get("ok"):
        raise RuntimeError(
            f"현재 POSE 읽기 실패: {robot_pose}"
        )

    T_base_gripper = (
        mycobot_coords_to_transform(
            robot_pose["coords_mm_deg"]
        )
    )
    T_marker_gripper = (
        np.linalg.inv(T_base_marker)
        @ T_base_gripper
    )
    T_marker_tire = translation_transform_mm(
        slot_offset_mm
    )
    T_tire_gripper = (
        np.linalg.inv(T_marker_tire)
        @ T_marker_gripper
    )

    return save_model(
        T_tire_gripper,
        robot_pose,
        T_base_marker,
    )


def compute_slot_targets(
    slot_number: int,
    T_base_marker: np.ndarray,
    model: Dict[str, Any],
) -> Dict[str, List[float]]:
    slot_offset_mm = require_slot_offset(
        slot_number
    )

    T_tire_gripper = np.asarray(
        model["T_tire_gripper"],
        dtype=np.float64,
    ).reshape(4, 4)

    T_marker_tire = translation_transform_mm(
        slot_offset_mm
    )
    T_marker_gripper_grasp = (
        T_marker_tire
        @ T_tire_gripper
    )

    T_base_gripper_grasp = (
        T_base_marker
        @ T_marker_gripper_grasp
    )

    # grasp 바로 위로 50mm 올리는 방식은 로봇에서 멀어지는 방향이 되어
    # 최대 도달 반경을 벗어날 수 있다.
    #
    # 새 approach:
    #   1) grasp XY에서 로봇 베이스 원점(0, 0) 쪽으로 40mm 후퇴
    #   2) base Z 방향으로 20mm 상승
    #   3) 손목 방향은 grasp와 동일하게 유지
    T_base_gripper_approach = (
        T_base_gripper_grasp.copy()
    )

    grasp_xy_m = (
        T_base_gripper_grasp[:2, 3]
    )
    grasp_radius_m = float(
        np.linalg.norm(grasp_xy_m)
    )

    if grasp_radius_m < 1e-9:
        raise ValueError(
            "grasp XY 반경이 0이라 approach 방향을 계산할 수 없습니다."
        )

    toward_base_xy = (
        -grasp_xy_m / grasp_radius_m
    )

    T_base_gripper_approach[
        0,
        3,
    ] += float(
        toward_base_xy[0]
        * APPROACH_PULLBACK_MM
        / 1000.0
    )
    T_base_gripper_approach[
        1,
        3,
    ] += float(
        toward_base_xy[1]
        * APPROACH_PULLBACK_MM
        / 1000.0
    )
    T_base_gripper_approach[
        2,
        3,
    ] += (
        APPROACH_LIFT_MM
        / 1000.0
    )

    grasp_reference = (
        transform_to_mycobot_coords(
            T_base_gripper_grasp
        )
    )
    approach_reference = (
        transform_to_mycobot_coords(
            T_base_gripper_approach
        )
    )

    grasp_command = [
        float(value)
        for value in grasp_reference
    ]
    approach_command = [
        float(value)
        for value in approach_reference
    ]

    for index in range(3):
        grasp_command[index] += float(
            GRASP_COMMAND_COMPENSATION_BASE_MM[
                index
            ]
        )
        approach_command[index] += float(
            APPROACH_COMMAND_COMPENSATION_BASE_MM[
                index
            ]
        )

    return {
        "grasp_reference": grasp_reference,
        "grasp_command": grasp_command,
        "approach_reference": (
            approach_reference
        ),
        "approach_command": (
            approach_command
        ),
    }


# ============================================================
# 메인
# ============================================================

def main() -> None:
    print("=" * 72)
    print("ArUco 슬롯 중심 기반 타이어 집기 시스템 v6")
    print("슬롯 1 중심: 왼쪽 30mm / 위쪽 35mm")
    print("중간 회전오차 소프트 허용 + 8단계 보간")
    print("=" * 72)

    T_gripper_camera = load_hand_eye(
        find_hand_eye_file()
    )
    K, D = load_camera_calibration(
        CAMERA_CALIBRATION_FILE
    )
    dictionary, params, detector = (
        create_detector()
    )

    model = load_model()

    if model is None:
        print("현재 저장된 중심 집기 모델: 없음")
    else:
        print(
            "현재 저장된 중심 집기 모델:",
            MODEL_FILE,
        )
        print(
            "저장 시각:",
            model.get("saved_at"),
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

    preview_targets: Optional[
        Dict[str, List[float]]
    ] = None
    preview_slot: Optional[int] = None
    preview_timestamp = 0.0

    print()
    print("공통")
    print("  1~4: 슬롯 선택")
    print("  m: 현재 ArUco 기준 저장")
    print("  c: 현재 로봇 pose")
    print("  q: 종료")
    print()
    print("중심 집기 모델 등록 — 처음 한 번만")
    print("  f: 서보 풀기")
    print("  e: 서보 잠그기")
    print("  k: 현재 중심 집기 자세를 모델로 저장")
    print()
    print("재현 시험")
    print("  p: approach/grasp 목표 미리보기")
    print("  x: approach 자세까지 이동")
    print("  g: grasp 안전검사 다시 출력")
    print("  z: grasp 자세까지만 저속 이동")
    print()
    print("z는 집게를 닫지 않습니다.")

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
            marker_index = (
                select_target_marker(ids)
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
                            pose["rvec"].reshape(
                                3,
                                1,
                            ),
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

            cv2.putText(
                display,
                (
                    "READY - press m"
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
                    f"slot:{selected_slot} | "
                    f"marker:{current_base_marker is not None} | "
                    f"model:{model is not None}"
                ),
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    "m marker | f/e manual | k teach | "
                    "p/x approach | g/z grasp"
                ),
                (10, FRAME_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )
            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
            ):
                selected_slot = int(
                    chr(key)
                )
                preview_targets = None
                preview_slot = None
                print(
                    "선택 슬롯:",
                    selected_slot,
                )

            elif key == ord("m"):
                stable_now = stable_marker_pose()

                if stable_now is None:
                    print(
                        "READY가 뜬 뒤 m을 누르세요."
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
                        stable_now,
                        T_gripper_camera,
                    )
                )

                preview_targets = None
                preview_slot = None
                pose_buffer.clear()
                lost_count = 0

                print()
                print(
                    "현재 ArUco 기준 저장 완료"
                )
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
                    "이후 운반대와 로봇 베이스를 "
                    "움직이지 마세요."
                )

            elif key == ord("f"):
                print(
                    "RELEASE:",
                    robot.request("RELEASE"),
                )
                print(
                    "로봇팔을 손으로 받치세요."
                )

            elif key == ord("e"):
                print(
                    "LOCK:",
                    robot.request("LOCK"),
                )

            elif key == ord("k"):
                if current_base_marker is None:
                    print(
                        "먼저 READY에서 m으로 "
                        "ArUco 기준을 저장하세요."
                    )
                    continue

                if selected_slot != REFERENCE_SLOT:
                    print(
                        "중심 집기 모델은 슬롯 1에서 "
                        "등록해야 합니다. 1을 누르세요."
                    )
                    continue

                try:
                    model = (
                        teach_tire_gripper_model(
                            robot,
                            current_base_marker,
                        )
                    )
                except Exception as exc:
                    print(
                        "모델 저장 실패:",
                        exc,
                    )
                    continue

                print()
                print("=" * 70)
                print(
                    "슬롯 1 중심 집기 모델 저장 완료"
                )
                print(
                    "저장 파일:",
                    MODEL_FILE,
                )
                print(
                    "tire→gripper XYZ(mm):",
                    np.round(
                        model[
                            "tire_relative_gripper_xyz_mm"
                        ],
                        3,
                    ).tolist(),
                )
                print(
                    "tire→gripper RPY(deg):",
                    np.round(
                        model[
                            "tire_relative_gripper_rpy_deg"
                        ],
                        3,
                    ).tolist(),
                )
                print("=" * 70)

            elif key == ord("p"):
                if current_base_marker is None:
                    print(
                        "먼저 READY에서 m으로 "
                        "ArUco 기준을 저장하세요."
                    )
                    continue

                if model is None:
                    print(
                        "중심 집기 모델이 없습니다. "
                        "슬롯 1에서 f→수동 정렬→e→k로 "
                        "먼저 저장하세요."
                    )
                    continue

                try:
                    preview_targets = (
                        compute_slot_targets(
                            selected_slot,
                            current_base_marker,
                            model,
                        )
                    )
                except Exception as exc:
                    print(
                        "목표 계산 실패:",
                        exc,
                    )
                    preview_targets = None
                    continue

                preview_slot = selected_slot
                preview_timestamp = time.monotonic()

                current_pose = robot.request(
                    "POSE"
                )

                print()
                print("=" * 72)
                print(
                    f"슬롯 {selected_slot} 목표 미리보기"
                )
                print(
                    "슬롯 중심 오프셋(mm):",
                    require_slot_offset(
                        selected_slot
                    ).tolist(),
                )
                print(
                    "현재:",
                    np.round(
                        current_pose.get(
                            "coords_mm_deg",
                            [],
                        ),
                        3,
                    ).tolist(),
                )
                print(
                    "자동 approach 방식:",
                    (
                        f"로봇 베이스 쪽 {APPROACH_PULLBACK_MM:.0f}mm "
                        f"+ 위 {APPROACH_LIFT_MM:.0f}mm"
                    ),
                )
                print(
                    "원래 approach:",
                    np.round(
                        preview_targets[
                            "approach_reference"
                        ],
                        3,
                    ).tolist(),
                )
                print(
                    "명령 approach:",
                    np.round(
                        preview_targets[
                            "approach_command"
                        ],
                        3,
                    ).tolist(),
                )
                print(
                    "원래 grasp:",
                    np.round(
                        preview_targets[
                            "grasp_reference"
                        ],
                        3,
                    ).tolist(),
                )
                print(
                    "명령 grasp:",
                    np.round(
                        preview_targets[
                            "grasp_command"
                        ],
                        3,
                    ).tolist(),
                )
                print(
                    "확인 후 x를 누르면 "
                    "approach까지만 이동합니다."
                )
                print("=" * 72)

            elif key == ord("x"):
                if (
                    preview_targets is None
                    or preview_slot
                    != selected_slot
                ):
                    print(
                        "먼저 p로 목표를 계산하세요."
                    )
                    continue

                if (
                    time.monotonic()
                    - preview_timestamp
                    > PREVIEW_VALID_SEC
                ):
                    print(
                        "미리보기가 오래되었습니다. "
                        "p를 다시 누르세요."
                    )
                    preview_targets = None
                    continue

                current_pose = robot.request(
                    "POSE"
                )

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
                target = [
                    float(value)
                    for value in preview_targets[
                        "approach_command"
                    ]
                ]

                # 목표 XY까지 현재 손목 각도를 그대로 유지하면
                # 먼 거리에서 역기구학이 풀리지 않을 수 있다.
                # 위치와 각도를 동시에 조금씩 바꾸는
                # 4단계 보간 경로를 사용한다.
                def interpolate_angle_deg(
                    start_deg: float,
                    end_deg: float,
                    ratio: float,
                ) -> float:
                    delta = (
                        float(end_deg)
                        - float(start_deg)
                        + 180.0
                    ) % 360.0 - 180.0

                    return (
                        float(start_deg)
                        + delta * ratio
                    )

                waypoints = []

                interpolation_ratios = (
                    0.125,
                    0.250,
                    0.375,
                    0.500,
                    0.625,
                    0.750,
                    0.875,
                    1.000,
                )

                for step_index, ratio in enumerate(
                    interpolation_ratios,
                    start=1,
                ):
                    coords = [
                        current[index]
                        + (
                            target[index]
                            - current[index]
                        )
                        * ratio
                        for index in range(3)
                    ]

                    coords.extend(
                        [
                            interpolate_angle_deg(
                                current[index],
                                target[index],
                                ratio,
                            )
                            for index in range(3, 6)
                        ]
                    )

                    name = (
                        "계산된 approach"
                        if ratio == 1.00
                        else (
                            f"보간 경유점 "
                            f"{step_index}/7"
                        )
                    )

                    waypoints.append(
                        (
                            name,
                            coords,
                        )
                    )

                all_ok = True

                total_steps = len(waypoints)

                for index, (
                    name,
                    coords,
                ) in enumerate(
                    waypoints,
                    start=1,
                ):
                    print()
                    print(
                        f"[{index}/{total_steps}] {name}"
                    )

                    response = (
                        robot_request_with_gui_pump(
                            robot=robot,
                            payload={
                                "command": (
                                    "MOVE_COORDS"
                                ),
                                "coords": coords,
                                "speed": (
                                    APPROACH_SPEED
                                ),
                                "mode": MOVE_MODE,
                            },
                            display_frame=display,
                            status_text=(
                                f"{index}/{total_steps} {name}"
                            ),
                        )
                    )

                    print("결과:", response)

                    if not response.get("ok"):
                        final_position_error = float(
                            response.get(
                                "final_position_error_mm",
                                float("inf"),
                            )
                        )
                        final_rotation_error = float(
                            response.get(
                                "final_rotation_error_deg",
                                float("inf"),
                            )
                        )

                        is_final_waypoint = (
                            index == total_steps
                        )

                        soft_intermediate_ok = (
                            not is_final_waypoint
                            and final_position_error
                            <= INTERMEDIATE_SOFT_POSITION_MM
                            and final_rotation_error
                            <= INTERMEDIATE_SOFT_ROTATION_DEG
                        )

                        if soft_intermediate_ok:
                            print(
                                "중간 경유점 소프트 통과:",
                                f"위치오차 {final_position_error:.2f}mm,",
                                f"회전오차 {final_rotation_error:.2f}deg",
                            )
                            print(
                                "최종 approach가 아니므로 "
                                "다음 보간 경유점으로 계속 진행합니다."
                            )
                        else:
                            print(
                                "이동 실패. 다음 단계를 "
                                "차단합니다."
                            )
                            all_ok = False
                            break

                    time.sleep(0.8)

                if all_ok:
                    print()
                    print(
                        "approach 자세 도착."
                    )
                    print(
                        "중심과 간섭을 확인한 뒤 "
                        "g를 누르세요."
                    )

            elif key == ord("g"):
                if (
                    preview_targets is None
                    or preview_slot
                    != selected_slot
                ):
                    print(
                        "먼저 p로 현재 슬롯 목표를 "
                        "계산하세요."
                    )
                    continue

                current_pose = robot.request(
                    "POSE"
                )

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
                approach_reference = (
                    preview_targets[
                        "approach_reference"
                    ]
                )
                grasp_command = (
                    preview_targets[
                        "grasp_command"
                    ]
                )

                (
                    approach_pos_error,
                    approach_rot_error,
                ) = pose_error(
                    current,
                    approach_reference,
                )
                (
                    grasp_step,
                    grasp_rotation_step,
                ) = pose_error(
                    current,
                    grasp_command,
                )

                print()
                print("=" * 72)
                print(
                    f"슬롯 {selected_slot} grasp 안전검사"
                )
                print(
                    "현재→원래 approach 위치오차:",
                    f"{approach_pos_error:.2f}mm",
                )
                print(
                    "현재→원래 approach 회전오차:",
                    f"{approach_rot_error:.2f}deg",
                )
                print(
                    "현재→명령 grasp 이동거리:",
                    f"{grasp_step:.2f}mm",
                )
                print(
                    "현재→명령 grasp 회전변화:",
                    f"{grasp_rotation_step:.2f}deg",
                )

                safe = True

                if (
                    approach_pos_error
                    > MAX_APPROACH_POSITION_ERROR_MM
                    or approach_rot_error
                    > MAX_APPROACH_ROTATION_ERROR_DEG
                ):
                    safe = False
                    print(
                        "차단: 현재 자세가 approach "
                        "근처가 아닙니다."
                    )

                if (
                    grasp_step
                    > MAX_GRASP_TRANSLATION_STEP_MM
                    or grasp_rotation_step
                    > MAX_GRASP_ROTATION_STEP_DEG
                ):
                    safe = False
                    print(
                        "차단: grasp 이동량이 "
                        "안전 기준을 넘었습니다."
                    )

                if safe:
                    print(
                        "통과. 집게가 열린 상태에서만 "
                        "z를 누르세요."
                    )
                print("=" * 72)

            elif key == ord("z"):
                if (
                    preview_targets is None
                    or preview_slot
                    != selected_slot
                ):
                    print(
                        "먼저 p→x→g 순서로 "
                        "진행하세요."
                    )
                    continue

                current_pose = robot.request(
                    "POSE"
                )

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
                approach_reference = (
                    preview_targets[
                        "approach_reference"
                    ]
                )
                grasp_command = (
                    preview_targets[
                        "grasp_command"
                    ]
                )
                grasp_reference = (
                    preview_targets[
                        "grasp_reference"
                    ]
                )

                (
                    approach_pos_error,
                    approach_rot_error,
                ) = pose_error(
                    current,
                    approach_reference,
                )
                (
                    grasp_step,
                    grasp_rotation_step,
                ) = pose_error(
                    current,
                    grasp_command,
                )

                if (
                    approach_pos_error
                    > MAX_APPROACH_POSITION_ERROR_MM
                    or approach_rot_error
                    > MAX_APPROACH_ROTATION_ERROR_DEG
                    or grasp_step
                    > MAX_GRASP_TRANSLATION_STEP_MM
                    or grasp_rotation_step
                    > MAX_GRASP_ROTATION_STEP_DEG
                ):
                    print(
                        "z 이동 차단. g 안전검사를 "
                        "다시 확인하세요."
                    )
                    continue

                print()
                print(
                    "grasp 자세로 저속 이동합니다."
                )
                print(
                    "이번 코드에서는 집게를 닫지 않습니다."
                )

                response = (
                    robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": (
                                "MOVE_COORDS"
                            ),
                            "coords": (
                                grasp_command
                            ),
                            "speed": GRASP_SPEED,
                            "mode": MOVE_MODE,
                        },
                        display_frame=display,
                        status_text=(
                            f"slot {selected_slot} grasp"
                        ),
                    )
                )

                print(
                    "grasp 이동 결과:",
                    response,
                )

                if response.get("ok"):
                    final_coords = response.get(
                        "final_coords_mm_deg"
                    )

                    if final_coords is not None:
                        (
                            reference_pos_error,
                            reference_rot_error,
                        ) = pose_error(
                            [
                                float(value)
                                for value in final_coords
                            ],
                            grasp_reference,
                        )

                        print(
                            "원래 중심 grasp 기준 위치오차:",
                            f"{reference_pos_error:.2f}mm",
                        )
                        print(
                            "원래 중심 grasp 기준 회전오차:",
                            f"{reference_rot_error:.2f}deg",
                        )

                    print(
                        "도착. 아직 집게를 닫지 마세요."
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
        print("프로그램 종료")


if __name__ == "__main__":
    main()