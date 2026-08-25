import json
import math
import socket
import time
import traceback
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

CALIBRATION_FILE = Path(
    "/home/choiminjoon/charuco2/camera_calibration.npz"
)

# 검증 데이터는 실행할 때마다 새 세션 폴더에 저장한다.
VALIDATION_ROOT = Path(
    "/home/choiminjoon/handeye_validation_v2"
)
DATASET_DIR = VALIDATION_ROOT / datetime.now().strftime(
    "session_%Y%m%d_%H%M%S"
)

SQUARES_X = 6
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022
DICTIONARY_ID = aruco.DICT_4X4_50

TOTAL_CHARUCO_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)  # 20
MIN_CHARUCO_CORNERS = 12

MAX_REPROJECTION_ERROR_PX = 1.50

POSE_BUFFER_SIZE = 18
MIN_STABLE_POSES = 8
MAX_TRANSLATION_SPREAD_MM = 1.80
MAX_ROTATION_SPREAD_DEG = 1.00

MOVE_SETTLE_SEC = 2.2
HOME_SETTLE_SEC = 1.5
POSE_CAPTURE_TIMEOUT_SEC = 15.0

# HOME 기준 작은 관절 오프셋들.
# 각 목표는 누적이 아니라 항상 HOME + offset이다.
AUTO_OFFSETS_DEG = [
    # 보드가 화면 밖으로 나가지 않도록 기존 검증 각도의 약 절반으로 축소
    # 각 목표는 누적 이동이 아니라 항상 HOME + 아래 offset이다.
    [1.0, -1.0, 0.5, 2.0, -1.5, 3.0],
    [-1.0, 1.0, -0.5, -2.0, 1.5, -3.0],
    [1.5, -2.0, 1.0, 2.0, 2.0, -2.5],
    [-1.5, 2.0, -1.0, -2.0, -2.0, 2.5],
    [0.5, 2.5, -1.5, -3.0, 1.0, 3.5],
]


# ============================================================
# 수학 및 통신
# ============================================================
def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def mycobot_coords_to_transform(coords_mm_deg: List[float]) -> np.ndarray:
    x, y, z, rx_deg, ry_deg, rz_deg = [float(v) for v in coords_mm_deg]
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
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


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = R_a.T @ R_b
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def valid_array(data: Any) -> bool:
    return data is not None and np.asarray(data).size > 0


class RobotClient:
    def __init__(self, ip: str, port: int) -> None:
        self.sock = socket.create_connection((ip, port), timeout=5.0)
        self.sock.settimeout(20.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffer = ""

    def request(self, command: str) -> Dict[str, Any]:
        self.sock.sendall((command.strip() + "\n").encode("utf-8"))

        while "\n" not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("로봇 서버 연결이 종료되었습니다.")
            self.buffer += chunk.decode("utf-8", errors="ignore")

        line, self.buffer = self.buffer.split("\n", 1)
        return json.loads(line)

    def close(self) -> None:
        try:
            self.request("QUIT")
        except Exception:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ============================================================
# 카메라 보정 및 ChArUco
# ============================================================
if not CALIBRATION_FILE.exists():
    raise FileNotFoundError(f"보정 파일 없음: {CALIBRATION_FILE}")

calib = np.load(CALIBRATION_FILE, allow_pickle=False)
camera_matrix = np.asarray(calib["camera_matrix"], dtype=np.float64).reshape(3, 3)
dist_coeffs = np.asarray(calib["dist_coeffs"], dtype=np.float64)

calibration_width = int(calib["image_width"]) if "image_width" in calib else FRAME_WIDTH
calibration_height = int(calib["image_height"]) if "image_height" in calib else FRAME_HEIGHT

if (calibration_width, calibration_height) != (FRAME_WIDTH, FRAME_HEIGHT):
    raise RuntimeError(
        f"보정 해상도 {calibration_width}x{calibration_height}와 "
        f"실행 해상도 {FRAME_WIDTH}x{FRAME_HEIGHT}가 다릅니다."
    )

dictionary = aruco.getPredefinedDictionary(DICTIONARY_ID)

try:
    board = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )
except (AttributeError, TypeError):
    board = aruco.CharucoBoard_create(
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )

detector_params = aruco.DetectorParameters()
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 53
detector_params.adaptiveThreshWinSizeStep = 10
detector_params.minMarkerPerimeterRate = 0.02

if hasattr(aruco, "CORNER_REFINE_NONE"):
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE

marker_detector = aruco.ArucoDetector(dictionary, detector_params)

modern_charuco_detector = None
if hasattr(aruco, "CharucoDetector"):
    try:
        charuco_params = aruco.CharucoParameters()
        if hasattr(charuco_params, "cameraMatrix"):
            charuco_params.cameraMatrix = camera_matrix
        if hasattr(charuco_params, "distCoeffs"):
            charuco_params.distCoeffs = dist_coeffs

        modern_charuco_detector = aruco.CharucoDetector(
            board,
            charuco_params,
            detector_params,
        )
    except Exception:
        modern_charuco_detector = None


def detect_charuco(gray: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    if modern_charuco_detector is not None:
        return modern_charuco_detector.detectBoard(gray)

    marker_corners, marker_ids, _ = marker_detector.detectMarkers(gray)

    if not valid_array(marker_ids):
        return None, None, marker_corners, marker_ids

    count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
        camera_matrix,
        dist_coeffs,
    )

    if count is None or int(count) <= 0:
        return None, None, marker_corners, marker_ids

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def charuco_object_image_points(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(board, "matchImagePoints"):
        obj_points, img_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )
        return (
            np.asarray(obj_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(img_points, dtype=np.float64).reshape(-1, 2),
        )

    all_corners = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float64,
    ).reshape(-1, 3)

    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    img_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    return all_corners[ids], img_points


def estimate_board_pose(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
) -> Optional[Dict[str, Any]]:
    corner_count = len(np.asarray(charuco_ids).reshape(-1))
    if corner_count < 4:
        return None

    obj_points, img_points = charuco_object_image_points(
        charuco_corners,
        charuco_ids,
    )

    ok, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return None

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(
        obj_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    error_px = float(
        np.sqrt(np.mean(np.sum((projected - img_points) ** 2, axis=1)))
    )

    R_target2cam, _ = cv2.Rodrigues(rvec)
    t_target2cam = np.asarray(tvec, dtype=np.float64).reshape(3)

    if not np.all(np.isfinite(t_target2cam)) or t_target2cam[2] <= 0:
        return None

    return {
        "R_target2cam": R_target2cam,
        "t_target2cam_m": t_target2cam,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
        "error_px": error_px,
        "corner_count": corner_count,
    }


# ============================================================
# 데이터 저장 및 안정 판정
# ============================================================
DATASET_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATASET_DIR / "raw"
ANNOTATED_DIR = DATASET_DIR / "annotated"
RAW_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)


def sample_files() -> List[Path]:
    return sorted(DATASET_DIR.glob("sample_*.json"))


def next_sample_index() -> int:
    indices = []
    for path in sample_files():
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(indices, default=0) + 1


PoseRecord = Tuple[np.ndarray, np.ndarray, float, int]
pose_buffer: Deque[PoseRecord] = deque(maxlen=POSE_BUFFER_SIZE)


def stable_pose_from_buffer() -> Optional[Dict[str, Any]]:
    if len(pose_buffer) < MIN_STABLE_POSES:
        return None

    records = list(pose_buffer)[-MIN_STABLE_POSES:]
    rotations = [record[0] for record in records]
    translations = np.stack([record[1] for record in records], axis=0)
    errors = np.asarray([record[2] for record in records], dtype=np.float64)
    corner_counts = [record[3] for record in records]

    R_mean = average_rotation(rotations)
    t_median = np.median(translations, axis=0)

    trans_spread_mm = float(
        np.sqrt(np.mean(np.sum((translations - t_median) ** 2, axis=1)))
        * 1000.0
    )

    rot_spread_deg = float(
        np.sqrt(
            np.mean(
                [
                    rotation_angle_deg(R_mean, R) ** 2
                    for R in rotations
                ]
            )
        )
    )

    if (
        min(corner_counts) < MIN_CHARUCO_CORNERS
        or float(np.median(errors)) > MAX_REPROJECTION_ERROR_PX
        or trans_spread_mm > MAX_TRANSLATION_SPREAD_MM
        or rot_spread_deg > MAX_ROTATION_SPREAD_DEG
    ):
        return None

    return {
        "R_target2cam": R_mean,
        "t_target2cam_m": t_median,
        "corner_count": int(min(corner_counts)),
        "median_error_px": float(np.median(errors)),
        "translation_spread_mm": trans_spread_mm,
        "rotation_spread_deg": rot_spread_deg,
    }


def save_sample(
    frame: np.ndarray,
    annotated: np.ndarray,
    robot_pose: Dict[str, Any],
    stable_pose: Dict[str, Any],
    offset_deg: List[float],
    target_angles_deg: List[float],
) -> Path:
    index = next_sample_index()

    raw_path = RAW_DIR / f"sample_{index:03d}.jpg"
    annotated_path = ANNOTATED_DIR / f"sample_{index:03d}.jpg"
    json_path = DATASET_DIR / f"sample_{index:03d}.json"

    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(annotated_path), annotated)

    coords = robot_pose["coords_mm_deg"]
    T_base_gripper = mycobot_coords_to_transform(coords)

    payload = {
        "index": index,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "capture_mode": "automatic_joint_offsets",
        "board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_LENGTH_M,
            "marker_length_m": MARKER_LENGTH_M,
            "dictionary": "DICT_4X4_50",
        },
        "automatic_motion": {
            "offset_deg": offset_deg,
            "target_angles_deg": target_angles_deg,
        },
        "robot": {
            "coords_mm_deg": coords,
            "angles_deg": robot_pose["angles_deg"],
            "T_base_gripper": T_base_gripper.tolist(),
        },
        "camera_target": {
            "R_target2cam": stable_pose["R_target2cam"].tolist(),
            "t_target2cam_m": stable_pose["t_target2cam_m"].tolist(),
        },
        "quality": {
            "charuco_corner_count": stable_pose["corner_count"],
            "median_reprojection_error_px": stable_pose["median_error_px"],
            "translation_spread_mm": stable_pose["translation_spread_mm"],
            "rotation_spread_deg": stable_pose["rotation_spread_deg"],
        },
        "raw_image": str(raw_path),
        "annotated_image": str(annotated_path),
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return json_path


# ============================================================
# 메인
# ============================================================
def main() -> None:
    robot = RobotClient(ROBOT_IP, ROBOT_PORT)

    print("PING:", robot.request("PING"))
    status = robot.request("STATUS")
    print("STATUS:", status)

    home_response = robot.request("GET_HOME")
    if not home_response.get("ok"):
        raise RuntimeError(f"HOME 읽기 실패: {home_response}")

    home_angles = [
        float(value)
        for value in home_response["home_angles_deg"]
    ]

    print("HOME angles:", home_angles)
    print("검증 세션 저장 폴더:", DATASET_DIR)

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not cap.isOpened():
        robot.close()
        raise RuntimeError(f"카메라를 열 수 없습니다: {CAMERA_DEVICE}")

    time.sleep(1.0)
    for _ in range(10):
        cap.read()

    aborted = False
    running = False
    current_pose_number = 0
    current_pose_total = 0
    last_frame = None
    last_display = None
    last_pose = None
    last_corner_count = 0

    def read_and_draw(
        message: str = "",
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
        nonlocal last_frame, last_display, last_pose, last_corner_count

        ok, frame = cap.read()
        if not ok:
            return False, None, None, None

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners = charuco_ids = marker_corners = marker_ids = None

        try:
            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
            ) = detect_charuco(gray)
        except Exception as exc:
            cv2.putText(
                display,
                f"DETECT ERROR: {exc}",
                (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if valid_array(marker_ids):
            try:
                aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            except cv2.error:
                pass

        pose = None
        corner_count = 0

        if valid_array(charuco_ids):
            corner_count = len(np.asarray(charuco_ids).reshape(-1))

            try:
                aruco.drawDetectedCornersCharuco(
                    display,
                    charuco_corners,
                    charuco_ids,
                    (0, 255, 0),
                )
            except cv2.error:
                points = np.asarray(charuco_corners).reshape(-1, 2)
                for px, py in points:
                    cv2.circle(
                        display,
                        (int(round(px)), int(round(py))),
                        4,
                        (0, 255, 0),
                        -1,
                    )

            pose = estimate_board_pose(charuco_corners, charuco_ids)

            if pose is not None:
                try:
                    cv2.drawFrameAxes(
                        display,
                        camera_matrix,
                        dist_coeffs,
                        pose["rvec"].reshape(3, 1),
                        pose["t_target2cam_m"].reshape(3, 1),
                        0.06,
                        2,
                    )
                except cv2.error:
                    pass

        cv2.putText(
            display,
            f"samples: {len(sample_files())}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            f"corners: {corner_count}/{TOTAL_CHARUCO_CORNERS}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if pose is not None:
            cv2.putText(
                display,
                f"reprojection: {pose['error_px']:.3f}px",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if running:
            cv2.putText(
                display,
                f"AUTO {current_pose_number}/{current_pose_total}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                display,
                "t: 2 pose test   a: 5 pose validate   q: quit",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if message:
            cv2.putText(
                display,
                message,
                (10, 152),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        last_frame = frame.copy()
        last_display = display.copy()
        last_pose = pose
        last_corner_count = corner_count

        cv2.imshow("Automatic Eye-in-Hand Capture", display)

        return True, pose, frame, display

    def check_abort_key() -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def wait_seconds(seconds: float, message: str) -> bool:
        end_time = time.monotonic() + seconds

        while time.monotonic() < end_time:
            ok, _, _, _ = read_and_draw(message)
            if not ok:
                return False
            if check_abort_key():
                return False

        return True

    def capture_stable_pose(
        offset: List[float],
        target: List[float],
    ) -> bool:
        """
        기존 버전처럼 위치·회전 spread가 아주 작아질 때까지 기다리지 않는다.

        로봇이 멈춘 상태에서:
        - 코너 12개 이상
        - 재투영 오차 1.5px 이하
        인 프레임을 최소 5개 모은다.

        모은 프레임 중 코너 수가 많고 재투영 오차가 낮은 프레임을 사진으로 저장하고,
        보드 자세는 유효 프레임들의 평균/중앙값을 사용한다.
        """
        minimum_valid_frames = 5
        valid_records = []
        best_candidate = None
        deadline = time.monotonic() + POSE_CAPTURE_TIMEOUT_SEC

        while time.monotonic() < deadline:
            ok, pose, frame, display = read_and_draw(
                f"COLLECTING GOOD FRAMES {len(valid_records)}/{minimum_valid_frames}"
            )

            if not ok:
                return False

            if check_abort_key():
                return False

            if (
                pose is None
                or pose["corner_count"] < MIN_CHARUCO_CORNERS
                or pose["error_px"] > MAX_REPROJECTION_ERROR_PX
            ):
                continue

            record = {
                "R": pose["R_target2cam"].copy(),
                "t": pose["t_target2cam_m"].copy(),
                "error_px": float(pose["error_px"]),
                "corner_count": int(pose["corner_count"]),
                "frame": frame.copy(),
                "display": display.copy(),
            }
            valid_records.append(record)

            # 1순위: 코너 수가 많음, 2순위: 재투영 오차가 작음
            if best_candidate is None:
                best_candidate = record
            else:
                current_key = (
                    record["corner_count"],
                    -record["error_px"],
                )
                best_key = (
                    best_candidate["corner_count"],
                    -best_candidate["error_px"],
                )
                if current_key > best_key:
                    best_candidate = record

            if len(valid_records) >= minimum_valid_frames:
                # 최근 데이터 중 품질이 좋은 최대 12개만 사용
                selected = sorted(
                    valid_records,
                    key=lambda item: (
                        -item["corner_count"],
                        item["error_px"],
                    ),
                )[:12]

                rotations = [item["R"] for item in selected]
                translations = np.stack(
                    [item["t"] for item in selected],
                    axis=0,
                )
                errors = np.asarray(
                    [item["error_px"] for item in selected],
                    dtype=np.float64,
                )
                corner_counts = [
                    item["corner_count"] for item in selected
                ]

                R_mean = average_rotation(rotations)
                t_median = np.median(translations, axis=0)

                trans_spread_mm = float(
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

                rot_spread_deg = float(
                    np.sqrt(
                        np.mean(
                            [
                                rotation_angle_deg(R_mean, R) ** 2
                                for R in rotations
                            ]
                        )
                    )
                )

                stable_pose = {
                    "R_target2cam": R_mean,
                    "t_target2cam_m": t_median,
                    "corner_count": int(min(corner_counts)),
                    "median_error_px": float(np.median(errors)),
                    "translation_spread_mm": trans_spread_mm,
                    "rotation_spread_deg": rot_spread_deg,
                }

                robot_pose = robot.request("POSE")

                if not robot_pose.get("ok"):
                    print("POSE 읽기 실패:", robot_pose)
                    return False

                path = save_sample(
                    best_candidate["frame"],
                    best_candidate["display"],
                    robot_pose,
                    stable_pose,
                    offset,
                    target,
                )

                print(
                    f"저장 완료: {path.name} | "
                    f"valid_frames={len(valid_records)} | "
                    f"corners={stable_pose['corner_count']} | "
                    f"reproj={stable_pose['median_error_px']:.3f}px | "
                    f"trans_spread={trans_spread_mm:.3f}mm | "
                    f"rot_spread={rot_spread_deg:.3f}deg"
                )

                return True

        print(
            "이 자세에서 사용할 수 있는 프레임이 부족해서 건너뜀:",
            offset,
            f"| valid_frames={len(valid_records)}",
        )
        return True

    def run_sequence(offsets: List[List[float]]) -> bool:
        nonlocal running, current_pose_number, current_pose_total

        running = True
        current_pose_total = len(offsets)

        saved_before = len(sample_files())
        skipped = 0

        try:
            for index, offset in enumerate(offsets, start=1):
                current_pose_number = index

                target = [
                    home_angles[j] + float(offset[j])
                    for j in range(6)
                ]

                print()
                print("==================================================")
                print(f"AUTO POSE {index}/{len(offsets)}")
                print("offset:", offset)
                print("target:", [round(v, 2) for v in target])
                print("==================================================")

                move_command = "MOVE," + ",".join(
                    f"{value:.3f}" for value in target
                )

                move_result = robot.request(move_command)

                if not move_result.get("ok"):
                    print("이동 실패:", move_result)
                    robot.request("STOP")
                    robot.request("HOME")
                    return False

                if not wait_seconds(MOVE_SETTLE_SEC, "SETTLING AFTER MOVE"):
                    return False

                saved_count_before_pose = len(sample_files())

                if not capture_stable_pose(offset, target):
                    return False

                if len(sample_files()) == saved_count_before_pose:
                    skipped += 1

                home_result = robot.request("HOME")

                if not home_result.get("ok"):
                    print("HOME 복귀 실패:", home_result)
                    return False

                if not wait_seconds(HOME_SETTLE_SEC, "RETURNING HOME"):
                    return False

            saved_after = len(sample_files())

            print()
            print("==================================================")
            print("검증 촬영 완료")
            print("이번 저장 수:", saved_after - saved_before)
            print("건너뛴 자세:", skipped)
            print("전체 데이터 수:", saved_after)
            print("==================================================")

            return True

        finally:
            running = False
            current_pose_number = 0
            current_pose_total = 0
            try:
                robot.request("HOME")
            except Exception:
                pass

    print()
    print("==================================================")
    print("Eye-in-Hand 별도 검증 촬영 준비 - 최적 프레임 저장 버전")
    print("t : 앞의 2자세만 시험")
    print("a : 검증용 5자세 자동 촬영")
    print("q : HOME 복귀 후 종료")
    print("==================================================")

    try:
        while True:
            ok, _, display_frame, display = read_and_draw()
            if not ok:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("t"):
                print("2자세 안전 시험 시작")
                if not run_sequence(AUTO_OFFSETS_DEG[:2]):
                    aborted = True
                    break

            elif key == ord("a"):
                print("전체 자동 촬영 시작")
                if not run_sequence(AUTO_OFFSETS_DEG):
                    aborted = True
                    break

            elif key == ord("s"):
                path = DATASET_DIR / f"screen_{int(time.time())}.jpg"
                cv2.imwrite(str(path), display)
                print("화면 저장:", path)

            elif key == ord("q"):
                aborted = True
                break

    except KeyboardInterrupt:
        print("\nCtrl+C 입력")
        aborted = True

    except Exception as exc:
        traceback.print_exc()
        print("프로그램 오류:", exc)
        aborted = True

    finally:
        try:
            robot.request("STOP")
            print("HOME 복귀:", robot.request("HOME"))
        except Exception as exc:
            print("HOME 복귀 실패:", exc)

        cap.release()
        cv2.destroyAllWindows()
        robot.close()

        if aborted:
            print("검증 촬영 중단 및 종료")
        else:
            print("검증 촬영 프로그램 종료")


if __name__ == "__main__":
    main()