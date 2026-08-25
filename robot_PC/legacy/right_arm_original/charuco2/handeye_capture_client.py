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
# 사용자 환경 설정
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

DATASET_DIR = Path(
    "/home/choiminjoon/handeye_dataset_v2"
)

# 사용자가 만든 ChArUco 보드
SQUARES_X = 6
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022
DICTIONARY_ID = aruco.DICT_4X4_50

# 이 보드는 내부 ChArUco 코너가 6개이므로 가능하면 전부 검출한 프레임만 저장
MIN_CHARUCO_CORNERS = 16
MAX_REPROJECTION_ERROR_PX = 1.20

POSE_BUFFER_SIZE = 15
MIN_STABLE_POSES = 10
MAX_TRANSLATION_STD_MM = 1.5
MAX_ROTATION_STD_DEG = 0.8

LOCK_SETTLE_SEC = 2.0
RELEASE_CONFIRM_SEC = 3.0


# ============================================================
# 기본 수학 함수
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
    """
    myCobot get_coords() = [x,y,z,rx,ry,rz]
    위치는 mm, 자세는 degree.
    제조사 문서의 roll/pitch/yaw, Euler ZYX 순서를 사용:
        R = Rz(rz) @ Ry(ry) @ Rx(rx)
    반환값은 ^base T_gripper.
    """
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


# ============================================================
# 로봇 TCP 클라이언트
# ============================================================
class RobotClient:
    def __init__(self, ip: str, port: int) -> None:
        self.sock = socket.create_connection((ip, port), timeout=5.0)
        self.sock.settimeout(8.0)
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
# 카메라 보정값
# ============================================================
if not CALIBRATION_FILE.exists():
    raise FileNotFoundError(f"카메라 보정 파일 없음: {CALIBRATION_FILE}")

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


# ============================================================
# ChArUco 보드 및 검출기
# ============================================================
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

# ChArUco는 체스판 경계와 가까워 marker corner refinement가 오히려 흔들릴 수 있어
# 마커 코너 자체 refinement는 끄고 ChArUco 코너 subpixel 보정을 사용한다.
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
    except Exception as exc:
        print("CharucoDetector 생성 경고, 호환 모드 사용:", exc)
        modern_charuco_detector = None


def detect_charuco(
    gray: np.ndarray,
) -> Tuple[Any, Any, Any, Any]:
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
# 데이터셋 관리
# ============================================================
DATASET_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATASET_DIR / "raw"
ANNOTATED_DIR = DATASET_DIR / "annotated"
RAW_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)


def sample_files() -> List[Path]:
    return sorted(DATASET_DIR.glob("sample_*.json"))


def next_sample_index() -> int:
    files = sample_files()
    if not files:
        return 1
    indices = []
    for file in files:
        try:
            indices.append(int(file.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(indices, default=0) + 1


def delete_last_sample() -> None:
    files = sample_files()
    if not files:
        print("삭제할 샘플이 없습니다.")
        return

    target = files[-1]
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        for key in ("raw_image", "annotated_image"):
            path_text = data.get(key)
            if path_text:
                Path(path_text).unlink(missing_ok=True)
    except Exception:
        pass

    target.unlink(missing_ok=True)
    print("마지막 샘플 삭제:", target.name)


# ============================================================
# 안정성 검사
# ============================================================
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

    trans_std_mm = float(
        np.sqrt(np.mean(np.sum((translations - t_median) ** 2, axis=1)))
        * 1000.0
    )

    rot_errors_deg = [
        rotation_angle_deg(R_mean, R) for R in rotations
    ]
    rot_std_deg = float(np.sqrt(np.mean(np.square(rot_errors_deg))))

    if (
        trans_std_mm > MAX_TRANSLATION_STD_MM
        or rot_std_deg > MAX_ROTATION_STD_DEG
        or float(np.median(errors)) > MAX_REPROJECTION_ERROR_PX
        or min(corner_counts) < MIN_CHARUCO_CORNERS
    ):
        return None

    return {
        "R_target2cam": R_mean,
        "t_target2cam_m": t_median,
        "median_error_px": float(np.median(errors)),
        "translation_spread_mm": trans_std_mm,
        "rotation_spread_deg": rot_std_deg,
        "corner_count": int(min(corner_counts)),
    }


# ============================================================
# 카메라 및 메인 루프
# ============================================================
def main() -> None:
    robot = RobotClient(ROBOT_IP, ROBOT_PORT)
    print("로봇 PING:", robot.request("PING"))
    print("로봇 STATUS:", robot.request("STATUS"))

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 기존 카메라 시험값 유지
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

    robot_state = "LOCKED"
    settle_until = time.monotonic() + LOCK_SETTLE_SEC
    release_confirm_until = 0.0
    last_pose: Optional[Dict[str, Any]] = None
    last_annotated: Optional[np.ndarray] = None

    print()
    print("==================================================")
    print("Eye-in-Hand 데이터 수집")
    print("f 두 번 : 서보 해제")
    print("e       : 서보 잠금")
    print("c       : 샘플 저장")
    print("u       : 마지막 샘플 삭제")
    print("s       : 화면 저장")
    print("q       : 잠금 후 종료")
    print("==================================================")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임 읽기 실패")
                break

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
                    (10, 210),
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

            current_pose = None
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
                    # OpenCV 버전별 draw assertion 회피
                    pts = np.asarray(charuco_corners).reshape(-1, 2)
                    for px, py in pts:
                        cv2.circle(
                            display,
                            (int(round(px)), int(round(py))),
                            4,
                            (0, 255, 0),
                            -1,
                        )

                current_pose = estimate_board_pose(
                    charuco_corners,
                    charuco_ids,
                )

                if current_pose is not None:
                    last_pose = current_pose

                    try:
                        cv2.drawFrameAxes(
                            display,
                            camera_matrix,
                            dist_coeffs,
                            current_pose["rvec"].reshape(3, 1),
                            current_pose["t_target2cam_m"].reshape(3, 1),
                            0.06,
                            2,
                        )
                    except cv2.error:
                        pass

                    if (
                        robot_state == "LOCKED"
                        and time.monotonic() >= settle_until
                        and current_pose["corner_count"] >= MIN_CHARUCO_CORNERS
                        and current_pose["error_px"]
                        <= MAX_REPROJECTION_ERROR_PX
                    ):
                        pose_buffer.append(
                            (
                                current_pose["R_target2cam"],
                                current_pose["t_target2cam_m"],
                                current_pose["error_px"],
                                current_pose["corner_count"],
                            )
                        )

            stable_pose = stable_pose_from_buffer()
            is_stable = stable_pose is not None

            status_color = (0, 255, 0) if robot_state == "LOCKED" else (0, 0, 255)
            cv2.putText(
                display,
                f"ROBOT: {robot_state}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"samples: {len(sample_files())}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"charuco corners: {corner_count}/20",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if current_pose is not None:
                t_cm = current_pose["t_target2cam_m"] * 100.0
                cv2.putText(
                    display,
                    (
                        f"board xyz: {t_cm[0]:.2f}, "
                        f"{t_cm[1]:.2f}, {t_cm[2]:.2f} cm"
                    ),
                    (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"reprojection: {current_pose['error_px']:.3f}px",
                    (10, 146),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            stable_text = "STABLE - C CAPTURE" if is_stable else "WAITING FOR STABLE"
            cv2.putText(
                display,
                stable_text,
                (10, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (0, 255, 0) if is_stable else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            if release_confirm_until > time.monotonic():
                cv2.putText(
                    display,
                    "HOLD ARM! PRESS F AGAIN TO RELEASE",
                    (35, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.67,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                "f,f:FREE  e:LOCK  c:CAPTURE  u:UNDO  q:QUIT",
                (10, FRAME_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            last_annotated = display.copy()
            cv2.imshow("Eye-in-Hand Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("f"):
                now = time.monotonic()

                if robot_state == "FREE":
                    print("이미 FREE 상태입니다.")
                    continue

                if release_confirm_until > now:
                    print("서보 해제 요청:", robot.request("RELEASE"))
                    robot_state = "FREE"
                    pose_buffer.clear()
                    release_confirm_until = 0.0
                    print("팔을 손으로 받치면서 천천히 이동하세요.")
                else:
                    release_confirm_until = now + RELEASE_CONFIRM_SEC
                    print("안전 확인: 집게/카메라를 손으로 받친 후 f를 한 번 더 누르세요.")

            elif key == ord("e"):
                print("서보 잠금 요청:", robot.request("LOCK"))
                robot_state = "LOCKED"
                pose_buffer.clear()
                settle_until = time.monotonic() + LOCK_SETTLE_SEC
                release_confirm_until = 0.0
                print(f"{LOCK_SETTLE_SEC:.1f}초 후 안정값을 모읍니다.")

            elif key == ord("c"):
                if robot_state != "LOCKED":
                    print("저장 거부: 먼저 e를 눌러 서보를 잠그세요.")
                    continue

                stable_pose = stable_pose_from_buffer()
                if stable_pose is None:
                    print(
                        "저장 거부: 6/6 코너와 STABLE 표시를 확인하세요."
                    )
                    continue

                robot_pose = robot.request("POSE")
                if not robot_pose.get("ok"):
                    print("로봇 자세 읽기 실패:", robot_pose)
                    continue

                coords = robot_pose["coords_mm_deg"]
                T_base_gripper = mycobot_coords_to_transform(coords)

                index = next_sample_index()
                stamp = datetime.now().isoformat(timespec="milliseconds")

                raw_path = RAW_DIR / f"sample_{index:03d}.jpg"
                annotated_path = ANNOTATED_DIR / f"sample_{index:03d}.jpg"
                json_path = DATASET_DIR / f"sample_{index:03d}.json"

                cv2.imwrite(str(raw_path), frame)
                cv2.imwrite(str(annotated_path), last_annotated)

                payload = {
                    "index": index,
                    "timestamp": stamp,
                    "board": {
                        "squares_x": SQUARES_X,
                        "squares_y": SQUARES_Y,
                        "square_length_m": SQUARE_LENGTH_M,
                        "marker_length_m": MARKER_LENGTH_M,
                        "dictionary": "DICT_4X4_50",
                    },
                    "robot": {
                        "coords_mm_deg": coords,
                        "angles_deg": robot_pose["angles_deg"],
                        "T_base_gripper": T_base_gripper.tolist(),
                    },
                    "camera_target": {
                        "R_target2cam": stable_pose[
                            "R_target2cam"
                        ].tolist(),
                        "t_target2cam_m": stable_pose[
                            "t_target2cam_m"
                        ].tolist(),
                    },
                    "quality": {
                        "charuco_corner_count": stable_pose["corner_count"],
                        "median_reprojection_error_px": stable_pose[
                            "median_error_px"
                        ],
                        "translation_spread_mm": stable_pose[
                            "translation_spread_mm"
                        ],
                        "rotation_spread_deg": stable_pose[
                            "rotation_spread_deg"
                        ],
                    },
                    "raw_image": str(raw_path),
                    "annotated_image": str(annotated_path),
                }

                json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                print()
                print("==================================================")
                print(f"샘플 {index:03d} 저장 완료")
                print("robot coords:", coords)
                print(
                    "target->camera t(cm):",
                    np.asarray(
                        stable_pose["t_target2cam_m"]
                    )
                    * 100.0,
                )
                print("파일:", json_path)
                print("==================================================")
                pose_buffer.clear()
                settle_until = time.monotonic() + 0.5

            elif key == ord("u"):
                delete_last_sample()

            elif key == ord("s"):
                path = DATASET_DIR / f"screen_{int(time.time())}.jpg"
                cv2.imwrite(str(path), display)
                print("화면 저장:", path)

            elif key == ord("q"):
                print("종료 전 서보 잠금:", robot.request("LOCK"))
                break

    except KeyboardInterrupt:
        print("\nCtrl+C 입력")

    except Exception as exc:
        traceback.print_exc()
        print("프로그램 오류:", exc)

    finally:
        try:
            robot.request("LOCK")
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        robot.close()
        print("수집 프로그램 종료")


if __name__ == "__main__":
    main()