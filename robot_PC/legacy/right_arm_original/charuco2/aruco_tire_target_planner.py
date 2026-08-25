import json
import math
import socket
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np


# ============================================================
# 사용자 설정
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

# 트럭의 단일 ArUco 마커 설정
MARKER_LENGTH_M = 0.030
DICTIONARY_ID = aruco.DICT_4X4_50

# 특정 ID만 사용하려면 숫자로 지정한다.
# 화면에 트럭 마커 하나만 보이면 None으로 둬도 된다.
TARGET_ID: Optional[int] = None


# ============================================================
# 반드시 실제로 측정해서 수정할 값
# ============================================================

# ArUco 중심에서 타이어 중심까지의 거리 [X, Y, Z], 단위 mm.
#
# 화면에 그려지는 마커 좌표축을 기준으로 한다.
# +X: 빨간 축 방향
# +Y: 초록 축 방향
# +Z: 파란 축 방향
#
# 아직 측정하지 않았다면 0으로 두고 실행해도 되지만,
# 그 경우 타이어 중심 결과는 ArUco 중심과 똑같이 나온다.
MARKER_TO_TIRE_OFFSET_MM = np.array(
    [35.0, 30.0, 0.0],
    dtype=np.float64,
)

# 타이어 중심에서 안전하게 떨어진 접근 후보 거리
APPROACH_DISTANCE_MM = 80.0


# ============================================================
# 측정 안정 조건
# ============================================================

STABLE_BUFFER_SIZE = 15
MIN_STABLE_SAMPLES = 8

MAX_TRANSLATION_SPREAD_MM = 2.0
MAX_ROTATION_SPREAD_DEG = 1.0
MAX_REPROJECTION_ERROR_PX = 1.5

SAVE_DIR = Path(
    "/home/choiminjoon/aruco_tire_target_results"
)


# ============================================================
# 변환 행렬
# ============================================================

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def make_translation_transform(
    x_m: float,
    y_m: float,
    z_m: float,
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [x_m, y_m, z_m]
    return T


def rot_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=np.float64,
    )


def rot_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=np.float64,
    )


def rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        dtype=np.float64,
    )


def mycobot_coords_to_transform(
    coords_mm_deg: List[float],
) -> np.ndarray:
    """
    MyCobot get_coords():
        [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]

    Hand-Eye 캘리브레이션 때 사용한 것과 같은 회전 순서:
        R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    if len(coords_mm_deg) != 6:
        raise ValueError("로봇 coords는 숫자 6개여야 합니다.")

    x, y, z, rx_deg, ry_deg, rz_deg = [
        float(value) for value in coords_mm_deg
    ]

    rx, ry, rz = np.deg2rad(
        [rx_deg, ry_deg, rz_deg]
    )

    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    t_m = np.array([x, y, z], dtype=np.float64) / 1000.0

    return make_transform(R, t_m)


def average_rotation(rotations: List[np.ndarray]) -> np.ndarray:
    M = np.mean(np.stack(rotations, axis=0), axis=0)
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
    cosine = (np.trace(R_delta) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_matrix_to_rpy_zyx_deg(
    R: np.ndarray,
) -> np.ndarray:
    """
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    반환: [roll, pitch, yaw] degree
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.rad2deg([roll, pitch, yaw])


# ============================================================
# 파일 로딩
# ============================================================

def find_hand_eye_file() -> Path:
    for path in HAND_EYE_FILE_CANDIDATES:
        if path.exists():
            return path

    paths = "\n".join(
        f"- {path}" for path in HAND_EYE_FILE_CANDIDATES
    )

    raise FileNotFoundError(
        "Eye-in-Hand 파일을 찾지 못했습니다.\n"
        f"{paths}"
    )


def load_hand_eye_transform(
    path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)

    if "T_gripper_camera" in data:
        T_gripper_camera = np.asarray(
            data["T_gripper_camera"],
            dtype=np.float64,
        ).reshape(4, 4)

    elif (
        "R_cam2gripper" in data
        and "t_cam2gripper_m" in data
    ):
        R = np.asarray(
            data["R_cam2gripper"],
            dtype=np.float64,
        ).reshape(3, 3)

        t = np.asarray(
            data["t_cam2gripper_m"],
            dtype=np.float64,
        ).reshape(3)

        T_gripper_camera = make_transform(R, t)

    else:
        raise KeyError(
            "Eye-in-Hand npz에 T_gripper_camera 또는 "
            "R_cam2gripper/t_cam2gripper_m가 없습니다."
        )

    metadata: Dict[str, Any] = {}

    for key in (
        "method",
        "translation_rms_mm",
        "rotation_rms_deg",
        "sample_count",
    ):
        if key in data:
            value = data[key]
            try:
                metadata[key] = value.item()
            except Exception:
                metadata[key] = np.asarray(value).tolist()

    return T_gripper_camera, metadata


def load_camera_calibration(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"카메라 보정 파일이 없습니다: {path}"
        )

    data = np.load(path, allow_pickle=False)

    camera_matrix = np.asarray(
        data["camera_matrix"],
        dtype=np.float64,
    ).reshape(3, 3)

    dist_coeffs = np.asarray(
        data["dist_coeffs"],
        dtype=np.float64,
    )

    if "image_width" in data and "image_height" in data:
        image_width = int(data["image_width"])
        image_height = int(data["image_height"])

        if (
            image_width != FRAME_WIDTH
            or image_height != FRAME_HEIGHT
        ):
            raise RuntimeError(
                "카메라 보정 해상도와 실행 해상도가 다릅니다. "
                f"보정={image_width}x{image_height}, "
                f"실행={FRAME_WIDTH}x{FRAME_HEIGHT}"
            )

    return camera_matrix, dist_coeffs


# ============================================================
# JetCobot 서버 통신
# ============================================================

class RobotClient:
    def __init__(self, ip: str, port: int) -> None:
        self.sock = socket.create_connection(
            (ip, port),
            timeout=5.0,
        )
        self.sock.settimeout(20.0)
        self.sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        self.buffer = ""

    def request(self, command: str) -> Dict[str, Any]:
        self.sock.sendall(
            (command.strip() + "\n").encode("utf-8")
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

        line, self.buffer = self.buffer.split("\n", 1)
        return json.loads(line)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# ============================================================
# ArUco 검출 및 자세 추정
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

    if hasattr(aruco, "ArucoDetector"):
        return (
            dictionary,
            aruco.ArucoDetector(dictionary, params),
            params,
        )

    return dictionary, None, params


def detect_markers(
    gray: np.ndarray,
    dictionary,
    detector,
    detector_params,
):
    if detector is not None:
        return detector.detectMarkers(gray)

    return aruco.detectMarkers(
        gray,
        dictionary,
        parameters=detector_params,
    )


def marker_object_points(
    marker_length_m: float,
) -> np.ndarray:
    half = marker_length_m / 2.0

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
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[Dict[str, Any]]:
    image_points = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(4, 2)

    object_points = marker_object_points(
        MARKER_LENGTH_M
    )

    flags = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=flags,
    )

    if not ok:
        return None

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected = np.asarray(
        projected,
        dtype=np.float64,
    ).reshape(4, 2)

    reprojection_error_px = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (projected - image_points) ** 2,
                    axis=1,
                )
            )
        )
    )

    R_camera_marker, _ = cv2.Rodrigues(rvec)
    t_camera_marker_m = np.asarray(
        tvec,
        dtype=np.float64,
    ).reshape(3)

    if (
        not np.all(np.isfinite(t_camera_marker_m))
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
        "reprojection_error_px": reprojection_error_px,
    }


def select_marker_index(
    corners,
    ids,
) -> Optional[int]:
    if ids is None or len(ids) == 0:
        return None

    flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)

    if TARGET_ID is not None:
        matches = np.where(flat_ids == TARGET_ID)[0]

        if len(matches) == 0:
            return None

        return int(matches[0])

    areas = []

    for marker_corners in corners:
        points = np.asarray(
            marker_corners,
            dtype=np.float32,
        ).reshape(4, 2)

        areas.append(abs(cv2.contourArea(points)))

    return int(np.argmax(areas))


# ============================================================
# 안정 판정
# ============================================================

PoseRecord = Tuple[np.ndarray, np.ndarray, float]

pose_buffer: Deque[PoseRecord] = deque(
    maxlen=STABLE_BUFFER_SIZE
)


def get_stable_marker_pose() -> Optional[Dict[str, Any]]:
    if len(pose_buffer) < MIN_STABLE_SAMPLES:
        return None

    records = list(pose_buffer)[-MIN_STABLE_SAMPLES:]

    rotations = [record[0] for record in records]
    translations = np.stack(
        [record[1] for record in records],
        axis=0,
    )
    reprojection_errors = np.asarray(
        [record[2] for record in records],
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
                    (translations - t_median) ** 2,
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
                        rotation,
                    ) ** 2
                    for rotation in rotations
                ]
            )
        )
    )

    median_reprojection_error_px = float(
        np.median(reprojection_errors)
    )

    if (
        translation_spread_mm
        > MAX_TRANSLATION_SPREAD_MM
        or rotation_spread_deg
        > MAX_ROTATION_SPREAD_DEG
        or median_reprojection_error_px
        > MAX_REPROJECTION_ERROR_PX
    ):
        return None

    return {
        "R_camera_marker": R_mean,
        "t_camera_marker_m": t_median,
        "translation_spread_mm": translation_spread_mm,
        "rotation_spread_deg": rotation_spread_deg,
        "reprojection_error_px": (
            median_reprojection_error_px
        ),
    }


# ============================================================
# 목표 계산
# ============================================================

def calculate_targets(
    robot_pose: Dict[str, Any],
    stable_pose: Dict[str, Any],
    T_gripper_camera: np.ndarray,
) -> Dict[str, Any]:
    coords = robot_pose.get("coords_mm_deg")

    if coords is None:
        raise RuntimeError(
            f"로봇 좌표가 없습니다: {robot_pose}"
        )

    T_base_gripper = mycobot_coords_to_transform(
        coords
    )

    T_camera_marker = make_transform(
        stable_pose["R_camera_marker"],
        stable_pose["t_camera_marker_m"],
    )

    T_base_marker = (
        T_base_gripper
        @ T_gripper_camera
        @ T_camera_marker
    )

    offset_m = (
        np.asarray(
            MARKER_TO_TIRE_OFFSET_MM,
            dtype=np.float64,
        )
        / 1000.0
    )

    T_marker_tire = make_translation_transform(
        offset_m[0],
        offset_m[1],
        offset_m[2],
    )

    T_base_tire = (
        T_base_marker
        @ T_marker_tire
    )

    approach_distance_m = (
        APPROACH_DISTANCE_MM / 1000.0
    )

    # 타이어 좌표계는 마커와 같은 방향으로 둔다.
    T_tire_approach_plus_z = (
        make_translation_transform(
            0.0,
            0.0,
            approach_distance_m,
        )
    )

    T_tire_approach_minus_z = (
        make_translation_transform(
            0.0,
            0.0,
            -approach_distance_m,
        )
    )

    T_base_approach_plus_z = (
        T_base_tire
        @ T_tire_approach_plus_z
    )

    T_base_approach_minus_z = (
        T_base_tire
        @ T_tire_approach_minus_z
    )

    return {
        "timestamp": datetime.now().isoformat(
            timespec="milliseconds"
        ),
        "robot_coords_mm_deg": [
            float(value) for value in coords
        ],
        "marker_to_tire_offset_mm": (
            MARKER_TO_TIRE_OFFSET_MM.tolist()
        ),
        "approach_distance_mm": (
            float(APPROACH_DISTANCE_MM)
        ),
        "quality": {
            "reprojection_error_px": float(
                stable_pose["reprojection_error_px"]
            ),
            "translation_spread_mm": float(
                stable_pose["translation_spread_mm"]
            ),
            "rotation_spread_deg": float(
                stable_pose["rotation_spread_deg"]
            ),
        },
        "T_base_marker": T_base_marker.tolist(),
        "T_base_tire": T_base_tire.tolist(),
        "T_base_approach_plus_z": (
            T_base_approach_plus_z.tolist()
        ),
        "T_base_approach_minus_z": (
            T_base_approach_minus_z.tolist()
        ),
        "base_marker_xyz_mm": (
            T_base_marker[:3, 3] * 1000.0
        ).tolist(),
        "base_tire_xyz_mm": (
            T_base_tire[:3, 3] * 1000.0
        ).tolist(),
        "base_approach_plus_z_xyz_mm": (
            T_base_approach_plus_z[:3, 3]
            * 1000.0
        ).tolist(),
        "base_approach_minus_z_xyz_mm": (
            T_base_approach_minus_z[:3, 3]
            * 1000.0
        ).tolist(),
        "base_marker_rpy_deg": (
            rotation_matrix_to_rpy_zyx_deg(
                T_base_marker[:3, :3]
            ).tolist()
        ),
    }


def print_result(result: Dict[str, Any]) -> None:
    print()
    print("=" * 66)
    print("ArUco → 타이어 중심/안전 접근점 계산 결과")
    print("-" * 66)

    print(
        "ArUco 중심 base XYZ(mm):",
        np.round(
            result["base_marker_xyz_mm"],
            3,
        ),
    )

    print(
        "입력한 marker→tire offset(mm):",
        np.round(
            result["marker_to_tire_offset_mm"],
            3,
        ),
    )

    print(
        "타이어 중심 base XYZ(mm):",
        np.round(
            result["base_tire_xyz_mm"],
            3,
        ),
    )

    print()
    print(
        f"안전 접근 거리: "
        f"{result['approach_distance_mm']:.1f}mm"
    )

    print(
        "접근 후보 A: 타이어에서 marker +Z 방향",
        np.round(
            result["base_approach_plus_z_xyz_mm"],
            3,
        ),
    )

    print(
        "접근 후보 B: 타이어에서 marker -Z 방향",
        np.round(
            result["base_approach_minus_z_xyz_mm"],
            3,
        ),
    )

    quality = result["quality"]

    print()
    print(
        "측정 품질: "
        f"reproj={quality['reprojection_error_px']:.3f}px, "
        f"translation_spread="
        f"{quality['translation_spread_mm']:.3f}mm, "
        f"rotation_spread="
        f"{quality['rotation_spread_deg']:.3f}deg"
    )

    print("-" * 66)
    print(
        "이 프로그램은 좌표만 계산합니다. "
        "로봇을 움직이지 않습니다."
    )
    print(
        "후보 A와 B 중 트럭 바깥쪽에 있는 접근점을 "
        "실제 배치로 확인해야 합니다."
    )
    print("=" * 66)


# ============================================================
# 메인
# ============================================================

def main() -> None:
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hand_eye_file = find_hand_eye_file()

    T_gripper_camera, hand_eye_metadata = (
        load_hand_eye_transform(hand_eye_file)
    )

    camera_matrix, dist_coeffs = (
        load_camera_calibration(
            CAMERA_CALIBRATION_FILE
        )
    )

    dictionary, detector, detector_params = (
        create_detector()
    )

    print("Eye-in-Hand 파일:", hand_eye_file)
    print("Eye-in-Hand 정보:", hand_eye_metadata)
    print("카메라 보정 파일:", CAMERA_CALIBRATION_FILE)
    print(
        "ArUco:",
        "DICT_4X4_50,",
        f"marker length={MARKER_LENGTH_M * 1000:.1f}mm,",
        "TARGET_ID=",
        TARGET_ID,
    )
    print(
        "marker→tire offset(mm):",
        MARKER_TO_TIRE_OFFSET_MM,
    )
    print(
        "안전 접근 거리(mm):",
        APPROACH_DISTANCE_MM,
    )

    print("JetCobot 서버 연결 중...")

    robot = RobotClient(
        ROBOT_IP,
        ROBOT_PORT,
    )

    print("서버 응답:", robot.request("PING"))

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
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

    print()
    print("=" * 66)
    print("이 프로그램에서는 로봇이 움직이지 않습니다.")
    print("READY가 뜨면 c를 눌러 목표를 계산하세요.")
    print("s: 마지막 결과 저장, r: 초기화, p: robot pose, q: 종료")
    print("=" * 66)

    last_result: Optional[Dict[str, Any]] = None
    selected_id_text = "NONE"

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("카메라 프레임 읽기 실패")
                break

            display = frame.copy()
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            corners, ids, rejected = detect_markers(
                gray,
                dictionary,
                detector,
                detector_params,
            )

            selected_index = select_marker_index(
                corners,
                ids,
            )

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(
                    display,
                    corners,
                    ids,
                )

            if selected_index is not None:
                selected_id = int(
                    np.asarray(ids).reshape(-1)[
                        selected_index
                    ]
                )

                selected_id_text = str(selected_id)

                pose = estimate_marker_pose(
                    corners[selected_index],
                    camera_matrix,
                    dist_coeffs,
                )

                if pose is not None:
                    if (
                        pose["reprojection_error_px"]
                        <= MAX_REPROJECTION_ERROR_PX
                    ):
                        pose_buffer.append(
                            (
                                pose["R_camera_marker"],
                                pose["t_camera_marker_m"],
                                pose[
                                    "reprojection_error_px"
                                ],
                            )
                        )

                    try:
                        cv2.drawFrameAxes(
                            display,
                            camera_matrix,
                            dist_coeffs,
                            pose["rvec"].reshape(3, 1),
                            pose[
                                "t_camera_marker_m"
                            ].reshape(3, 1),
                            MARKER_LENGTH_M,
                            3,
                        )
                    except cv2.error:
                        pass

                    xyz_mm = (
                        pose["t_camera_marker_m"]
                        * 1000.0
                    )

                    cv2.putText(
                        display,
                        (
                            f"camera XYZ mm: "
                            f"{xyz_mm[0]:.1f}, "
                            f"{xyz_mm[1]:.1f}, "
                            f"{xyz_mm[2]:.1f}"
                        ),
                        (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        display,
                        (
                            "reproj: "
                            f"{pose['reprojection_error_px']:.3f}px"
                        ),
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            else:
                selected_id_text = "NONE"
                pose_buffer.clear()

            stable_pose = get_stable_marker_pose()

            stable_text = (
                "READY - press c"
                if stable_pose is not None
                else (
                    f"stabilizing "
                    f"{len(pose_buffer)}/"
                    f"{MIN_STABLE_SAMPLES}"
                )
            )

            stable_color = (
                (0, 255, 0)
                if stable_pose is not None
                else (0, 165, 255)
            )

            cv2.putText(
                display,
                f"selected marker ID: {selected_id_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                stable_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                stable_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "c: calculate  s: save  r: reset  p: pose  q: quit",
                (10, FRAME_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "ArUco Tire Target Planner",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                pose_buffer.clear()
                print("측정 버퍼 초기화")

            if key == ord("p"):
                print(
                    "현재 로봇 POSE:",
                    robot.request("POSE"),
                )

            if key == ord("c"):
                stable_pose = get_stable_marker_pose()

                if stable_pose is None:
                    print(
                        "아직 측정값이 안정되지 않았습니다. "
                        "READY가 뜬 뒤 c를 누르세요."
                    )
                    continue

                robot_pose = robot.request("POSE")

                if not robot_pose.get("ok"):
                    print(
                        "로봇 pose 읽기 실패:",
                        robot_pose,
                    )
                    continue

                last_result = calculate_targets(
                    robot_pose,
                    stable_pose,
                    T_gripper_camera,
                )

                print_result(last_result)
                pose_buffer.clear()

            if key == ord("s"):
                if last_result is None:
                    print(
                        "저장할 결과가 없습니다. "
                        "먼저 READY 상태에서 c를 누르세요."
                    )
                    continue

                path = (
                    SAVE_DIR
                    / (
                        "tire_target_"
                        + datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + ".json"
                    )
                )

                path.write_text(
                    json.dumps(
                        last_result,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print("결과 저장:", path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.close()
        print("타이어 목표 계산 프로그램 종료")


if __name__ == "__main__":
    main()