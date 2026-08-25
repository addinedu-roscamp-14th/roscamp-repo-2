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

REGISTRATION_FILE = Path(
    "/home/choiminjoon/slot_pose_registration/"
    "slot_registration.json"
)

MARKER_LENGTH_M = 0.030
DICTIONARY_ID = aruco.DICT_4X4_50
TARGET_ID = 0

STABLE_BUFFER_SIZE = 12
MIN_STABLE_SAMPLES = 5
MAX_TRANSLATION_SPREAD_MM = 3.0
MAX_ROTATION_SPREAD_DEG = 2.0
MAX_REPROJECTION_ERROR_PX = 2.5
LOST_LIMIT = 5

TEST_SPEED = 4
GRASP_TEST_SPEED = 2
MOVE_MODE = 0
SAFE_Z_MARGIN_MM = 40.0

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

# 실제로 도달하려는 집기 깊이 기준.
# 슬롯 1은 등록 자세보다 Z를 2.0mm 더 낮게 잡는다.
GRASP_PHYSICAL_DEPTH_OFFSET_MM = {
    1: -2.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
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
    1: np.array([0.9, 5.8, 7.9], dtype=np.float64),
    2: np.array([0.0, 0.0, 0.0], dtype=np.float64),
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
    # 슬롯 1: v7보다 명령 Z를 1.0mm 더 낮춘다.
    # 물리 목표 깊이는 별도의 GRASP_PHYSICAL_DEPTH_OFFSET_MM로 판단한다.
    1: np.array([3.0, 0.0, 5.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    3: np.array([0.0, 0.0, 0.0], dtype=np.float64),
    4: np.array([0.0, 0.0, 0.0], dtype=np.float64),
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
    print("중심 재등록 자세 재현 시험 깊이보정 v8")
    print("슬롯 1 명령 Z 1mm 추가 하향 / 실제 깊이 목표 -2mm")
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

    print()
    print("1~4: 슬롯 선택")
    print("m: 현재 마커 저장")
    print("p: 접근 목표 미리보기")
    print("x: 접근 자세로 단계 이동")
    print("g: 저장된 집기 목표 미리보기")
    print("z: 접근 자세에서 집기 자세로 저속 이동")
    print("s: 정지 요청(프로그램이 키를 받을 수 있을 때)")
    print("c: 현재 로봇 pose")
    print("q: 종료")
    print()
    print("주의: z 이동 중에는 키 입력 처리가 멈출 수 있습니다.")
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

            error_text = (
                f"{current_error_px:.3f}px"
                if current_error_px is not None
                else "NONE"
            )

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
                "p/x approach | g preview grasp | z move grasp",
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
                print(
                    "선택 슬롯:",
                    selected_slot,
                )

            elif key == ord("m"):
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

                preview_target = None
                preview_slot = None
                grasp_preview_target = None
                grasp_reference_target = None
                grasp_preview_slot = None
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
                    "이제 트럭을 움직이지 마세요."
                )

            elif key == ord("p"):
                if current_base_marker is None:
                    print(
                        "먼저 READY에서 m으로 "
                        "현재 마커 위치를 저장하세요."
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
                preview_target = [
                    float(value)
                    for value in preview_reference_target
                ]

                compensation = (
                    APPROACH_POSITION_COMPENSATION_MM[
                        selected_slot
                    ]
                )

                for axis_index in range(3):
                    preview_target[axis_index] += float(
                        compensation[axis_index]
                    )

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

                # 한 번에 목표로 가지 않고 안전 높이를 거쳐 이동한다.
                # 중요:
                # 목표 각도를 높은 Z에서 먼저 적용하면 IK가 풀리지 않는 경우가 있다.
                # 그래서 XY 이동 중에는 현재 손목 각도를 유지하고,
                # 마지막 하강에서 저장된 목표 각도로 전환한다.
                transition_z = max(
                    target[2] + SAFE_Z_MARGIN_MM,
                    min(current[2], target[2] + 60.0),
                )

                midpoint = [
                    (current[0] + target[0]) / 2.0,
                    (current[1] + target[1]) / 2.0,
                    current[2],
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

                for step_index, (name, coords) in enumerate(
                    waypoints,
                    start=1,
                ):
                    print()
                    print(
                        f"[{step_index}/3] {name} 이동 요청"
                    )

                    response = robot_request_with_gui_pump(
                        robot=robot,
                        payload={
                            "command": "MOVE_COORDS",
                            "coords": coords,
                            "speed": TEST_SPEED,
                            "mode": MOVE_MODE,
                        },
                        window_name=(
                            "Slot Approach and Grasp Test"
                        ),
                        display_frame=display,
                        status_text=(
                            f"{step_index}/3 {name}"
                        ),
                    )

                    print(
                        f"[{step_index}/3] 결과:",
                        response,
                    )

                    if not response.get("ok"):
                        print(
                            "단계 이동 실패. 다음 단계는 실행하지 않습니다."
                        )
                        all_ok = False
                        break

                    time.sleep(0.8)

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

                grasp_desired_reference = [
                    float(value)
                    for value in grasp_reference
                ]
                grasp_desired_reference[2] += float(
                    GRASP_PHYSICAL_DEPTH_OFFSET_MM[
                        selected_slot
                    ]
                )

                grasp_command = [
                    float(value)
                    for value in grasp_reference
                ]

                grasp_compensation = (
                    GRASP_POSITION_COMPENSATION_MM[
                        selected_slot
                    ]
                )

                for axis_index in range(3):
                    grasp_command[axis_index] += float(
                        grasp_compensation[
                            axis_index
                        ]
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
                            FINE_SETTLE_MIN_POSITION_MM
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

                            fine_target = previous_command.copy()
                            fine_target[:3] += correction_xyz
                            fine_target[3:] = reference_array[3:]

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

                    print()
                    print(
                        "집기 자세 도착. 집게는 아직 닫지 마세요."
                    )
                    print(
                        "타이어·집게·운반대 간섭을 눈으로 확인하고 "
                        "c로 현재 pose를 출력하세요."
                    )
                else:
                    print(
                        "집기 자세 이동 실패. 집게를 닫지 마세요."
                    )

                grasp_preview_target = None
                grasp_reference_target = None
                grasp_preview_slot = None

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