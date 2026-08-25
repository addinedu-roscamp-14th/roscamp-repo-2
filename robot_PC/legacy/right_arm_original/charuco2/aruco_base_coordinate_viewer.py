import json
import math
import socket
import time
from collections import deque
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

# 트럭 ArUco의 검은 외곽 한 변이 정확히 30 mm일 때
MARKER_LENGTH_M = 0.030
DICTIONARY_ID = aruco.DICT_4X4_50

# 특정 ID만 사용할 때 숫자로 변경. 화면에 마커 하나만 있으면 None.
TARGET_ID: Optional[int] = None

STABLE_BUFFER_SIZE = 12
MIN_STABLE_SAMPLES = 8
MAX_TRANSLATION_SPREAD_MM = 2.0
MAX_ROTATION_SPREAD_DEG = 1.0
MAX_REPROJECTION_ERROR_PX = 1.5


# 서버 시작 순간의 HOME을 기준으로 하는 작은 시험 자세 3개.
# 각 값은 [J1, J2, J3, J4, J5, J6] 오프셋(deg)이다.
# 누적 이동이 아니라 항상 HOME + 오프셋으로 이동한다.
TEST_POSE_OFFSETS_DEG = {
    "1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "2": [2.0, -1.0, 0.5, 2.0, -1.0, 2.0],
    "3": [-2.0, 1.0, -0.5, -2.0, 1.0, -2.0],
}

MOVE_SETTLE_SEC = 2.0


# ============================================================
# 좌표 변환
# ============================================================
def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def mycobot_coords_to_transform(coords_mm_deg: List[float]) -> np.ndarray:
    """캘리브레이션 때 사용한 동일한 회전 순서: Rz @ Ry @ Rx."""
    if len(coords_mm_deg) != 6:
        raise ValueError("로봇 coords는 6개여야 합니다.")

    x, y, z, rx_d, ry_d, rz_d = [float(v) for v in coords_mm_deg]
    rx, ry, rz = np.deg2rad([rx_d, ry_d, rz_d])
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


def rotation_difference_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    value = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(value))


def rotation_matrix_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), 반환 [roll,pitch,yaw]."""
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
    raise FileNotFoundError(
        "Eye-in-Hand 파일이 없습니다:\n"
        + "\n".join(str(p) for p in HAND_EYE_FILE_CANDIDATES)
    )


def load_hand_eye(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)

    if "T_gripper_camera" in data:
        T_gc = np.asarray(data["T_gripper_camera"], dtype=np.float64).reshape(4, 4)
    elif "R_cam2gripper" in data and "t_cam2gripper_m" in data:
        R = np.asarray(data["R_cam2gripper"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(data["t_cam2gripper_m"], dtype=np.float64).reshape(3)
        T_gc = make_transform(R, t)
    else:
        raise KeyError(
            "npz에 T_gripper_camera 또는 "
            "R_cam2gripper/t_cam2gripper_m가 없습니다."
        )

    metadata: Dict[str, Any] = {}
    for key in ("method", "translation_rms_mm", "rotation_rms_deg", "sample_count"):
        if key in data:
            value = data[key]
            try:
                metadata[key] = value.item()
            except Exception:
                metadata[key] = np.asarray(value).tolist()
    return T_gc, metadata


def load_camera_calibration(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"카메라 보정 파일이 없습니다: {path}")

    data = np.load(path, allow_pickle=False)
    K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(data["dist_coeffs"], dtype=np.float64)

    if "image_width" in data and "image_height" in data:
        width = int(data["image_width"])
        height = int(data["image_height"])
        if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
            raise RuntimeError(
                f"보정 해상도 {width}x{height}와 실행 해상도 "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT}가 다릅니다."
            )
    return K, D


# ============================================================
# 로봇 서버
# ============================================================
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
                raise ConnectionError("JetCobot 서버 연결이 종료되었습니다.")
            self.buffer += chunk.decode("utf-8", errors="ignore")
        line, self.buffer = self.buffer.split("\n", 1)
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
    dictionary = aruco.getPredefinedDictionary(DICTIONARY_ID)
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
    else:
        detector = None
    return dictionary, params, detector


def detect_markers(gray, dictionary, params, detector):
    if detector is not None:
        return detector.detectMarkers(gray)
    return aruco.detectMarkers(gray, dictionary, parameters=params)


def marker_object_points() -> np.ndarray:
    h = MARKER_LENGTH_M / 2.0
    return np.array(
        [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
        dtype=np.float64,
    )


def estimate_marker_pose(corners, K, D) -> Optional[Dict[str, Any]]:
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    object_points = marker_object_points()
    flag = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )

    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, D, flags=flag
    )
    if not ok:
        return None

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_points, K, D, rvec, tvec
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    projected = np.asarray(projected, dtype=np.float64).reshape(4, 2)
    error_px = float(
        np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
    )

    R_cm, _ = cv2.Rodrigues(rvec)
    t_cm = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(t_cm)) or t_cm[2] <= 0:
        return None

    return {
        "R_camera_marker": R_cm,
        "t_camera_marker_m": t_cm,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
        "error_px": error_px,
    }


def select_marker_index(corners, ids) -> Optional[int]:
    if ids is None or len(ids) == 0:
        return None

    flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if TARGET_ID is not None:
        matches = np.where(flat_ids == TARGET_ID)[0]
        return int(matches[0]) if len(matches) else None

    areas = []
    for marker_corners in corners:
        points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        areas.append(abs(cv2.contourArea(points)))
    return int(np.argmax(areas))


# ============================================================
# 안정화
# ============================================================
PoseRecord = Tuple[np.ndarray, np.ndarray, float]
pose_buffer: Deque[PoseRecord] = deque(maxlen=STABLE_BUFFER_SIZE)


def stable_marker_pose() -> Optional[Dict[str, Any]]:
    if len(pose_buffer) < MIN_STABLE_SAMPLES:
        return None

    records = list(pose_buffer)[-MIN_STABLE_SAMPLES:]
    rotations = [r[0] for r in records]
    translations = np.stack([r[1] for r in records], axis=0)
    errors = np.asarray([r[2] for r in records], dtype=np.float64)

    R_mean = average_rotation(rotations)
    t_median = np.median(translations, axis=0)

    trans_spread = float(
        np.sqrt(np.mean(np.sum((translations - t_median) ** 2, axis=1))) * 1000.0
    )
    rot_spread = float(
        np.sqrt(
            np.mean([rotation_difference_deg(R_mean, R) ** 2 for R in rotations])
        )
    )
    reproj = float(np.median(errors))

    if (
        trans_spread > MAX_TRANSLATION_SPREAD_MM
        or rot_spread > MAX_ROTATION_SPREAD_DEG
        or reproj > MAX_REPROJECTION_ERROR_PX
    ):
        return None

    return {
        "R_camera_marker": R_mean,
        "t_camera_marker_m": t_median,
        "translation_spread_mm": trans_spread,
        "rotation_spread_deg": rot_spread,
        "reprojection_error_px": reproj,
    }


def print_result(robot_pose, stable_pose, T_gripper_camera) -> None:
    coords = robot_pose.get("coords_mm_deg")
    if coords is None:
        raise RuntimeError(f"로봇 coords가 없습니다: {robot_pose}")

    T_base_gripper = mycobot_coords_to_transform(coords)
    T_camera_marker = make_transform(
        stable_pose["R_camera_marker"],
        stable_pose["t_camera_marker_m"],
    )
    T_base_marker = T_base_gripper @ T_gripper_camera @ T_camera_marker

    camera_xyz_mm = T_camera_marker[:3, 3] * 1000.0
    base_xyz_mm = T_base_marker[:3, 3] * 1000.0
    base_rpy_deg = rotation_matrix_to_rpy_deg(T_base_marker[:3, :3])

    print("\n" + "=" * 60)
    print("트럭 ArUco 좌표 변환 결과")
    print("-" * 60)
    print("현재 로봇 coords [mm, deg]:")
    print(np.round(np.asarray(coords, dtype=np.float64), 3))
    print("카메라 기준 ArUco XYZ [mm]:")
    print(np.round(camera_xyz_mm, 3))
    print("로봇 base 기준 ArUco XYZ [mm]:")
    print(np.round(base_xyz_mm, 3))
    print("로봇 base 기준 ArUco RPY [deg]:")
    print(np.round(base_rpy_deg, 3))
    print(
        "품질: "
        f"reproj={stable_pose['reprojection_error_px']:.3f}px, "
        f"translation_spread={stable_pose['translation_spread_mm']:.3f}mm, "
        f"rotation_spread={stable_pose['rotation_spread_deg']:.3f}deg"
    )
    print("주의: 이 좌표는 타이어 중심이 아니라 ArUco 중심입니다.")
    print("=" * 60)


# ============================================================
# 메인
# ============================================================
def main() -> None:
    hand_eye_file = find_hand_eye_file()
    T_gripper_camera, metadata = load_hand_eye(hand_eye_file)
    K, D = load_camera_calibration(CAMERA_CALIBRATION_FILE)
    dictionary, params, detector = create_detector()

    print("Eye-in-Hand 파일:", hand_eye_file)
    print("Eye-in-Hand 정보:", metadata)
    print("카메라 보정 파일:", CAMERA_CALIBRATION_FILE)
    print("ArUco: DICT_4X4_50, marker length=30mm, TARGET_ID=", TARGET_ID)
    print("JetCobot 서버 연결 중...")

    robot = RobotClient(ROBOT_IP, ROBOT_PORT)
    print("서버 응답:", robot.request("PING"))

    home_response = robot.request("GET_HOME")
    if not home_response.get("ok"):
        robot.close()
        raise RuntimeError(f"HOME 관절각을 읽지 못했습니다: {home_response}")

    home_angles_deg = [
        float(value)
        for value in home_response["home_angles_deg"]
    ]
    print("서버 시작 HOME 관절각:", home_angles_deg)

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        robot.close()
        raise RuntimeError(f"카메라를 열 수 없습니다: {CAMERA_DEVICE}")

    time.sleep(1.0)
    for _ in range(10):
        cap.read()

    print("\n이 버전은 카메라 창에서 1, 2, 3 키로 로봇 자세를 바꿉니다.")
    print("각 자세 이동 후 READY가 뜨면 c를 누르세요.")
    print("h: HOME 복귀, q: 종료")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임 읽기 실패")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray, dictionary, params, detector)
            selected_index = select_marker_index(corners, ids)

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(display, corners, ids)

            selected_id_text = "NONE"
            current_pose = None

            if selected_index is not None:
                selected_id = int(np.asarray(ids).reshape(-1)[selected_index])
                selected_id_text = str(selected_id)
                current_pose = estimate_marker_pose(
                    corners[selected_index], K, D
                )

                if current_pose is not None:
                    if current_pose["error_px"] <= MAX_REPROJECTION_ERROR_PX:
                        pose_buffer.append(
                            (
                                current_pose["R_camera_marker"],
                                current_pose["t_camera_marker_m"],
                                current_pose["error_px"],
                            )
                        )

                    try:
                        cv2.drawFrameAxes(
                            display,
                            K,
                            D,
                            current_pose["rvec"].reshape(3, 1),
                            current_pose["t_camera_marker_m"].reshape(3, 1),
                            MARKER_LENGTH_M * 0.7,
                            2,
                        )
                    except cv2.error:
                        pass

                    xyz = current_pose["t_camera_marker_m"] * 1000.0
                    cv2.putText(
                        display,
                        f"camera XYZ mm: {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}",
                        (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        f"reproj: {current_pose['error_px']:.3f}px",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                pose_buffer.clear()

            stable = stable_marker_pose()
            ready = stable is not None
            status = "READY - press c" if ready else f"stabilizing {len(pose_buffer)}/{MIN_STABLE_SAMPLES}"
            color = (0, 255, 0) if ready else (0, 165, 255)

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
                status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "1/2/3: move  h: home  c: calculate  q: quit",
                (10, FRAME_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ArUco to Robot Base Coordinate", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in (ord("1"), ord("2"), ord("3")):
                pose_key = chr(key)
                offset = TEST_POSE_OFFSETS_DEG[pose_key]
                target = [
                    home_angles_deg[index] + offset[index]
                    for index in range(6)
                ]
                command = "MOVE," + ",".join(
                    f"{value:.3f}" for value in target
                )

                print()
                print(f"측정 자세 {pose_key}로 이동")
                print("HOME 오프셋(deg):", offset)
                print("목표 관절각(deg):", [round(v, 3) for v in target])

                move_result = robot.request(command)
                print("이동 결과:", move_result)

                pose_buffer.clear()
                time.sleep(MOVE_SETTLE_SEC)
                print("로봇 정지 후 READY를 기다리세요.")

            if key == ord("h"):
                print("HOME 복귀 중...")
                home_result = robot.request("HOME")
                print("HOME 결과:", home_result)
                pose_buffer.clear()
                time.sleep(MOVE_SETTLE_SEC)

            if key == ord("r"):
                pose_buffer.clear()
                print("측정 버퍼 초기화")

            if key == ord("p"):
                print("현재 로봇 POSE:", robot.request("POSE"))

            if key == ord("c"):
                stable = stable_marker_pose()
                if stable is None:
                    print("아직 안정되지 않았습니다. READY 이후 c를 누르세요.")
                    continue

                robot_pose = robot.request("POSE")
                if not robot_pose.get("ok"):
                    print("로봇 POSE 읽기 실패:", robot_pose)
                    continue

                print_result(robot_pose, stable, T_gripper_camera)
                pose_buffer.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.close()
        print("좌표 출력 프로그램 종료")


if __name__ == "__main__":
    main()