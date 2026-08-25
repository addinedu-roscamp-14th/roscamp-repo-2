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
# 설정
# ============================================================

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

MARKER_LENGTH_M = 0.030
DICTIONARY_ID = aruco.DICT_4X4_50

# 마커가 여러 개 보이면 실제 운반대 마커 ID를 숫자로 지정한다.
TARGET_ID: Optional[int] = 0

STABLE_BUFFER_SIZE = 12
MIN_STABLE_SAMPLES = 5
MAX_TRANSLATION_SPREAD_MM = 3.0
MAX_ROTATION_SPREAD_DEG = 2.0
MAX_REPROJECTION_ERROR_PX = 2.5
LOST_LIMIT = 5

OUTPUT_DIR = Path(
    "/home/choiminjoon/slot_pose_registration"
)
OUTPUT_FILE = OUTPUT_DIR / "slot_registration.json"


# ============================================================
# 좌표 변환
# ============================================================

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
    if len(coords_mm_deg) != 6:
        raise ValueError("로봇 coords는 숫자 6개여야 합니다.")

    x, y, z, rx_deg, ry_deg, rz_deg = [
        float(value) for value in coords_mm_deg
    ]

    rx, ry, rz = np.deg2rad(
        [rx_deg, ry_deg, rz_deg]
    )

    # Hand-Eye 캘리브레이션에서 사용한 것과 같은 변환
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
    value = float(
        np.clip(
            (np.trace(R_delta) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(value))


def rotation_matrix_to_rpy_deg(
    R: np.ndarray,
) -> np.ndarray:
    sy = math.sqrt(
        R[0, 0] ** 2 + R[1, 0] ** 2
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

    return np.rad2deg([roll, pitch, yaw])


# ============================================================
# 파일 로딩
# ============================================================

def find_hand_eye_file() -> Path:
    for path in HAND_EYE_FILE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Eye-in-Hand 파일을 찾지 못했습니다:\n"
        + "\n".join(
            str(path)
            for path in HAND_EYE_FILE_CANDIDATES
        )
    )


def load_hand_eye(
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
            "Eye-in-Hand 파일에 필요한 변환이 없습니다."
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

    K = np.asarray(
        data["camera_matrix"],
        dtype=np.float64,
    ).reshape(3, 3)

    D = np.asarray(
        data["dist_coeffs"],
        dtype=np.float64,
    )

    if (
        "image_width" in data
        and "image_height" in data
    ):
        width = int(data["image_width"])
        height = int(data["image_height"])

        if (
            width != FRAME_WIDTH
            or height != FRAME_HEIGHT
        ):
            raise RuntimeError(
                f"보정 해상도 {width}x{height}와 "
                f"실행 해상도 {FRAME_WIDTH}x{FRAME_HEIGHT}가 다릅니다."
            )

    return K, D


# ============================================================
# 서버 통신
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
        self.sock.settimeout(20.0)
        self.sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        self.buffer = ""

    def request(
        self,
        command: str,
    ) -> Dict[str, Any]:
        self.sock.sendall(
            (
                command.strip()
                + "\n"
            ).encode("utf-8")
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


# ============================================================
# ArUco 검출
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
        aruco.ArucoDetector(dictionary, params)
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
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
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

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K,
                D,
                rvec,
                tvec,
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
        "error_px": error_px,
    }


def select_marker_index(
    corners,
    ids,
) -> Optional[int]:
    if ids is None or len(ids) == 0:
        return None

    flat_ids = np.asarray(
        ids,
        dtype=np.int32,
    ).reshape(-1)

    if TARGET_ID is not None:
        matches = np.where(
            flat_ids == TARGET_ID
        )[0]

        return (
            int(matches[0])
            if len(matches)
            else None
        )

    areas = []

    for marker_corners in corners:
        points = np.asarray(
            marker_corners,
            dtype=np.float32,
        ).reshape(4, 2)

        areas.append(
            abs(cv2.contourArea(points))
        )

    return int(np.argmax(areas))


# ============================================================
# 안정화
# ============================================================

PoseRecord = Tuple[np.ndarray, np.ndarray, float]

pose_buffer: Deque[PoseRecord] = deque(
    maxlen=STABLE_BUFFER_SIZE
)


def stable_marker_pose() -> Optional[Dict[str, Any]]:
    if len(pose_buffer) < MIN_STABLE_SAMPLES:
        return None

    records = list(pose_buffer)[
        -MIN_STABLE_SAMPLES:
    ]

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


# ============================================================
# 등록 데이터
# ============================================================

def new_registration() -> Dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "marker_reference": None,
        "slots": {
            "1": {},
            "2": {},
            "3": {},
            "4": {},
        },
    }


def load_registration() -> Dict[str, Any]:
    if not OUTPUT_FILE.exists():
        return new_registration()

    return json.loads(
        OUTPUT_FILE.read_text(
            encoding="utf-8",
        )
    )


def save_registration(
    registration: Dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    registration["updated_at"] = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            registration,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("등록 파일 저장:", OUTPUT_FILE)


def transform_summary(
    T: np.ndarray,
) -> Dict[str, Any]:
    return {
        "xyz_mm": (
            T[:3, 3] * 1000.0
        ).tolist(),
        "rpy_deg": (
            rotation_matrix_to_rpy_deg(
                T[:3, :3]
            ).tolist()
        ),
        "matrix": T.tolist(),
    }


def print_registration(
    registration: Dict[str, Any],
) -> None:
    print()
    print("=" * 66)
    print("현재 슬롯 등록 상태")

    marker_reference = registration.get(
        "marker_reference"
    )

    print(
        "마커 기준:",
        "저장됨"
        if marker_reference
        else "없음",
    )

    for slot_number in ("1", "2", "3", "4"):
        slot = registration["slots"].get(
            slot_number,
            {},
        )

        print(
            f"슬롯 {slot_number}: "
            f"approach={'저장' if 'approach' in slot else '-'}, "
            f"grasp={'저장' if 'grasp' in slot else '-'}"
        )

    print("=" * 66)


def save_marker_reference(
    registration: Dict[str, Any],
    robot: RobotClient,
    stable_pose: Dict[str, Any],
    T_gripper_camera: np.ndarray,
) -> None:
    robot_pose = robot.request("POSE")

    if not robot_pose.get("ok"):
        print(
            "로봇 POSE 읽기 실패:",
            robot_pose,
        )
        return

    T_base_gripper = mycobot_coords_to_transform(
        robot_pose["coords_mm_deg"]
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

    registration["marker_reference"] = {
        "saved_at": datetime.now().isoformat(
            timespec="milliseconds"
        ),
        "marker_id": TARGET_ID,
        "robot_coords_mm_deg": robot_pose[
            "coords_mm_deg"
        ],
        "quality": {
            "reprojection_error_px": stable_pose[
                "reprojection_error_px"
            ],
            "translation_spread_mm": stable_pose[
                "translation_spread_mm"
            ],
            "rotation_spread_deg": stable_pose[
                "rotation_spread_deg"
            ],
        },
        "T_base_marker": T_base_marker.tolist(),
        "base_marker_xyz_mm": (
            T_base_marker[:3, 3]
            * 1000.0
        ).tolist(),
        "base_marker_rpy_deg": (
            rotation_matrix_to_rpy_deg(
                T_base_marker[:3, :3]
            ).tolist()
        ),
    }

    save_registration(registration)

    print()
    print("운반대 ArUco 기준 자세 저장 완료")
    print(
        "base XYZ(mm):",
        np.round(
            T_base_marker[:3, 3]
            * 1000.0,
            3,
        ),
    )
    print(
        "이제부터 운반대와 마커를 움직이지 마세요."
    )


def save_slot_pose(
    registration: Dict[str, Any],
    robot: RobotClient,
    selected_slot: int,
    pose_name: str,
) -> None:
    marker_reference = registration.get(
        "marker_reference"
    )

    if not marker_reference:
        print(
            "먼저 마커가 보이는 상태에서 m을 눌러 "
            "마커 기준을 저장하세요."
        )
        return

    robot_pose = robot.request("POSE")

    if not robot_pose.get("ok"):
        print(
            "로봇 POSE 읽기 실패:",
            robot_pose,
        )
        return

    T_base_marker = np.asarray(
        marker_reference["T_base_marker"],
        dtype=np.float64,
    ).reshape(4, 4)

    T_base_gripper = mycobot_coords_to_transform(
        robot_pose["coords_mm_deg"]
    )

    T_marker_gripper = (
        np.linalg.inv(T_base_marker)
        @ T_base_gripper
    )

    pose_entry = {
        "saved_at": datetime.now().isoformat(
            timespec="milliseconds"
        ),
        "robot_coords_mm_deg": robot_pose[
            "coords_mm_deg"
        ],
        "robot_angles_deg": robot_pose[
            "angles_deg"
        ],
        "T_base_gripper": (
            T_base_gripper.tolist()
        ),
        "T_marker_gripper": (
            T_marker_gripper.tolist()
        ),
        "marker_relative_gripper_xyz_mm": (
            T_marker_gripper[:3, 3]
            * 1000.0
        ).tolist(),
        "marker_relative_gripper_rpy_deg": (
            rotation_matrix_to_rpy_deg(
                T_marker_gripper[:3, :3]
            ).tolist()
        ),
    }

    slot_key = str(selected_slot)
    registration["slots"].setdefault(
        slot_key,
        {},
    )[pose_name] = pose_entry

    save_registration(registration)

    korean_name = (
        "집기"
        if pose_name == "grasp"
        else "접근"
    )

    print()
    print(
        f"슬롯 {selected_slot} {korean_name} 자세 저장 완료"
    )
    print(
        "마커 기준 gripper XYZ(mm):",
        np.round(
            pose_entry[
                "marker_relative_gripper_xyz_mm"
            ],
            3,
        ),
    )
    print(
        "마커 기준 gripper RPY(deg):",
        np.round(
            pose_entry[
                "marker_relative_gripper_rpy_deg"
            ],
            3,
        ),
    )


# ============================================================
# 메인
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hand_eye_file = find_hand_eye_file()
    T_gripper_camera, metadata = load_hand_eye(
        hand_eye_file
    )
    K, D = load_camera_calibration(
        CAMERA_CALIBRATION_FILE
    )

    dictionary, params, detector = (
        create_detector()
    )

    registration = load_registration()
    selected_slot = 1

    print("Eye-in-Hand 파일:", hand_eye_file)
    print("Eye-in-Hand 정보:", metadata)
    print("등록 파일:", OUTPUT_FILE)
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
    print("=" * 70)
    print("1단계: 마커가 보이고 READY일 때 m")
    print("2단계: 운반대 고정")
    print("3단계: 1~4로 슬롯 선택")
    print("4단계: f → 수동 이동 → e → g 또는 a")
    print()
    print("m: marker  1~4: slot  g: grasp  a: approach")
    print("f: release  e: lock  p: pose  v: view  q: quit")
    print("=" * 70)

    print_registration(registration)

    # 한두 프레임 놓쳤다고 바로 안정화 버퍼를 지우지 않기 위한 카운터
    lost_count = 0
    current_error_px: Optional[float] = None

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

            corners, ids, _ = detect_markers(
                gray,
                dictionary,
                params,
                detector,
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

            marker_id_text = "NONE"
            current_error_px = None

            if selected_index is not None:
                lost_count = 0

                marker_id = int(
                    np.asarray(ids).reshape(-1)[
                        selected_index
                    ]
                )
                marker_id_text = str(marker_id)

                current_pose = estimate_marker_pose(
                    corners[selected_index],
                    K,
                    D,
                )

                if current_pose is not None:
                    current_error_px = float(
                        current_pose["error_px"]
                    )

                    if (
                        current_error_px
                        <= MAX_REPROJECTION_ERROR_PX
                    ):
                        pose_buffer.append(
                            (
                                current_pose[
                                    "R_camera_marker"
                                ],
                                current_pose[
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
                            current_pose[
                                "rvec"
                            ].reshape(3, 1),
                            current_pose[
                                "t_camera_marker_m"
                            ].reshape(3, 1),
                            MARKER_LENGTH_M
                            * 0.7,
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

            status = (
                "READY - press m to save marker"
                if ready
                else (
                    f"stabilizing "
                    f"{len(pose_buffer)}/"
                    f"{MIN_STABLE_SAMPLES}"
                )
            )

            marker_saved = (
                registration.get(
                    "marker_reference"
                )
                is not None
            )

            cv2.putText(
                display,
                f"marker ID: {marker_id_text}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                status,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (
                    (0, 255, 0)
                    if ready
                    else (0, 165, 255)
                ),
                2,
                cv2.LINE_AA,
            )

            error_text = (
                f"{current_error_px:.3f}px"
                if current_error_px is not None
                else "NONE"
            )

            cv2.putText(
                display,
                (
                    f"reproj: {error_text} | "
                    f"lost: {lost_count}/{LOST_LIMIT}"
                ),
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"marker saved: {marker_saved} | "
                    f"selected slot: {selected_slot}"
                ),
                (10, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    "m marker | 1-4 slot | "
                    "g grasp | a approach"
                ),
                (10, FRAME_HEIGHT - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    "f release | e lock | "
                    "v view | q quit"
                ),
                (10, FRAME_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Slot Pose Registration",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                try:
                    print(
                        "종료 전 LOCK:",
                        robot.request("LOCK"),
                    )
                except Exception as exc:
                    print("LOCK 실패:", exc)
                break

            if key == ord("m"):
                stable = stable_marker_pose()

                if stable is None:
                    print(
                        "아직 마커가 안정되지 않았습니다. "
                        "READY 이후 m을 누르세요."
                    )
                else:
                    save_marker_reference(
                        registration,
                        robot,
                        stable,
                        T_gripper_camera,
                    )
                    pose_buffer.clear()

            if key in (
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
            ):
                selected_slot = int(chr(key))
                print(
                    f"선택 슬롯: {selected_slot}"
                )

            if key == ord("f"):
                print()
                print(
                    "주의: 서보 해제 시 로봇팔이 떨어질 수 있습니다."
                )
                print(
                    "로봇팔을 손으로 받친 상태에서 진행하세요."
                )
                print(
                    "RELEASE:",
                    robot.request("RELEASE"),
                )

            if key == ord("e"):
                print(
                    "LOCK:",
                    robot.request("LOCK"),
                )
                time.sleep(1.0)

            if key == ord("g"):
                save_slot_pose(
                    registration,
                    robot,
                    selected_slot,
                    "grasp",
                )

            if key == ord("a"):
                save_slot_pose(
                    registration,
                    robot,
                    selected_slot,
                    "approach",
                )

            if key == ord("p"):
                print(
                    "현재 로봇 POSE:",
                    robot.request("POSE"),
                )

            if key == ord("v"):
                print_registration(
                    registration
                )

            if key == ord("r"):
                pose_buffer.clear()
                print(
                    "ArUco 측정 버퍼 초기화"
                )

    finally:
        try:
            robot.request("LOCK")
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        robot.close()

        print(
            "슬롯 자세 등록 프로그램 종료"
        )


if __name__ == "__main__":
    main()