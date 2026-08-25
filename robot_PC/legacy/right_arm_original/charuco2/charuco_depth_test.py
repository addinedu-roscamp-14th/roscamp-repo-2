import time
from collections import deque
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


# =========================================================
# 파일 경로
# =========================================================
WORK_DIR = Path(__file__).resolve().parent
CALIBRATION_FILE = WORK_DIR / "camera_calibration.npz"


# =========================================================
# 카메라 설정
# =========================================================
CAMERA_DEVICE = "/dev/video2"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30


# =========================================================
# 실제 ChArUco 보드 규격
# =========================================================
SQUARES_X = 4
SQUARES_Y = 3

SQUARE_LENGTH = 0.04   # 체스판 한 칸 4cm
MARKER_LENGTH = 0.03   # ArUco 마커 한 변 3cm

ARUCO_DICTIONARY = aruco.DICT_4X4_50

# 자세 계산에 필요한 최소 ChArUco 코너 수
MIN_CHARUCO_CORNERS = 4


# =========================================================
# 측정값 필터
# =========================================================
# 최근 7개 측정값의 중앙값 사용
FILTER_SIZE = 7

x_history = deque(maxlen=FILTER_SIZE)
y_history = deque(maxlen=FILTER_SIZE)
z_history = deque(maxlen=FILTER_SIZE)
distance_history = deque(maxlen=FILTER_SIZE)
error_history = deque(maxlen=FILTER_SIZE)


def reset_filter():
    """거리 필터 초기화."""
    x_history.clear()
    y_history.clear()
    z_history.clear()
    distance_history.clear()
    error_history.clear()


def data_exists(data):
    """OpenCV 결과 데이터가 존재하는지 확인."""
    if data is None:
        return False

    return np.asarray(data).size > 0


def normalize_charuco_data(corners, ids, maximum_id):
    """
    OpenCV 버전에 따라 달라지는 ChArUco 배열 모양을 정리한다.

    반환값:
        corner_points: N x 2
        corner_ids: N
    """
    if corners is None or ids is None:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int32)
        )

    try:
        corner_points = np.asarray(
            corners,
            dtype=np.float64
        ).reshape(-1, 2)

        corner_ids = np.asarray(
            ids,
            dtype=np.int32
        ).reshape(-1)

    except (ValueError, TypeError):
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int32)
        )

    # 코너와 ID 개수가 다르면 공통 개수까지만 사용
    count = min(
        len(corner_points),
        len(corner_ids)
    )

    if count <= 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int32)
        )

    corner_points = corner_points[:count]
    corner_ids = corner_ids[:count]

    # 유효한 ID와 정상 좌표만 사용
    valid_mask = (
        (corner_ids >= 0)
        & (corner_ids < maximum_id)
        & np.isfinite(corner_points[:, 0])
        & np.isfinite(corner_points[:, 1])
    )

    corner_points = np.ascontiguousarray(
        corner_points[valid_mask],
        dtype=np.float64
    )

    corner_ids = np.ascontiguousarray(
        corner_ids[valid_mask],
        dtype=np.int32
    )

    return corner_points, corner_ids


def draw_charuco_points(
    image,
    corner_points,
    corner_ids
):
    """
    drawDetectedCornersCharuco를 사용하지 않고
    ChArUco 코너와 ID를 직접 그린다.
    """
    for point, corner_id in zip(
        corner_points,
        corner_ids
    ):
        pixel_x = int(
            round(float(point[0]))
        )

        pixel_y = int(
            round(float(point[1]))
        )

        # 분홍색 원
        cv2.circle(
            image,
            (pixel_x, pixel_y),
            6,
            (255, 0, 255),
            -1
        )

        # 흰색 테두리
        cv2.circle(
            image,
            (pixel_x, pixel_y),
            7,
            (255, 255, 255),
            1
        )

        # ChArUco 코너 ID
        cv2.putText(
            image,
            str(int(corner_id)),
            (pixel_x + 7, pixel_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )


def draw_markers_manual(
    image,
    marker_corners,
    marker_ids
):
    """
    일반 ArUco 마커 테두리와 ID도 직접 그린다.
    """
    if marker_corners is None or marker_ids is None:
        return 0

    try:
        ids_array = np.asarray(
            marker_ids,
            dtype=np.int32
        ).reshape(-1)

    except (ValueError, TypeError):
        return 0

    marker_count = min(
        len(marker_corners),
        len(ids_array)
    )

    for index in range(marker_count):
        try:
            points = np.asarray(
                marker_corners[index],
                dtype=np.float64
            ).reshape(-1, 2)

        except (ValueError, TypeError):
            continue

        if len(points) != 4:
            continue

        polygon = np.round(
            points
        ).astype(np.int32).reshape(-1, 1, 2)

        # 초록색 마커 테두리
        cv2.polylines(
            image,
            [polygon],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        center = np.mean(
            points,
            axis=0
        )

        center_x = int(round(center[0]))
        center_y = int(round(center[1]))

        cv2.putText(
            image,
            f"id={int(ids_array[index])}",
            (center_x - 18, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return marker_count


def calculate_reprojection_error(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs
):
    """현재 자세 계산의 픽셀 재투영 오차를 계산한다."""
    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    projected_points = np.asarray(
        projected_points,
        dtype=np.float64
    ).reshape(-1, 2)

    detected_points = np.asarray(
        image_points,
        dtype=np.float64
    ).reshape(-1, 2)

    if len(projected_points) == 0:
        return float("nan")

    pixel_errors = np.linalg.norm(
        projected_points - detected_points,
        axis=1
    )

    return float(
        np.sqrt(
            np.mean(pixel_errors ** 2)
        )
    )


# =========================================================
# 보정 파일 불러오기
# =========================================================
if not CALIBRATION_FILE.exists():
    print("카메라 보정 파일을 찾을 수 없습니다.")
    print(CALIBRATION_FILE)
    raise SystemExit


try:
    calibration_data = np.load(
        CALIBRATION_FILE,
        allow_pickle=False
    )

    camera_matrix = np.asarray(
        calibration_data["camera_matrix"],
        dtype=np.float64
    ).reshape(3, 3)

    dist_coeffs = np.asarray(
        calibration_data["dist_coeffs"],
        dtype=np.float64
    )

    calibration_width = int(
        calibration_data["image_width"]
    )

    calibration_height = int(
        calibration_data["image_height"]
    )

except (KeyError, ValueError, OSError) as error:
    print("카메라 보정 파일을 읽지 못했습니다.")
    print(error)
    raise SystemExit


print("========================================")
print("카메라 보정값 불러오기 완료")
print(
    f"보정 해상도: "
    f"{calibration_width} x "
    f"{calibration_height}"
)
print()
print("camera_matrix:")
print(camera_matrix)
print()
print("dist_coeffs:")
print(dist_coeffs)
print("========================================")


if (
    calibration_width != FRAME_WIDTH
    or calibration_height != FRAME_HEIGHT
):
    print("보정 해상도와 실행 해상도가 다릅니다.")
    print("보정할 때와 동일한 640x480을 사용해야 합니다.")
    raise SystemExit


# =========================================================
# ChArUco 보드 생성
# =========================================================
aruco_dictionary = aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY
)

charuco_board = aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dictionary
)


# 보드 내부 ChArUco 코너의 실제 3차원 좌표
board_corner_points = np.asarray(
    charuco_board.getChessboardCorners(),
    dtype=np.float64
).reshape(-1, 3)

MAX_CHARUCO_CORNERS = len(
    board_corner_points
)

# 내부 코너들의 평균 위치는 보드 중심과 일치
board_center_object = np.mean(
    board_corner_points,
    axis=0
).reshape(3, 1)


print()
print(
    f"보드 내부 코너 수: "
    f"{MAX_CHARUCO_CORNERS}"
)


# =========================================================
# 검출 설정
# =========================================================
detector_parameters = aruco.DetectorParameters()

detector_parameters.cornerRefinementMethod = (
    aruco.CORNER_REFINE_NONE
)

detector_parameters.adaptiveThreshWinSizeMin = 3
detector_parameters.adaptiveThreshWinSizeMax = 71
detector_parameters.adaptiveThreshWinSizeStep = 4
detector_parameters.adaptiveThreshConstant = 7

detector_parameters.minMarkerPerimeterRate = 0.02
detector_parameters.maxMarkerPerimeterRate = 4.0

detector_parameters.minDistanceToBorder = 3
detector_parameters.polygonalApproxAccuracyRate = 0.03
detector_parameters.minCornerDistanceRate = 0.03

detector_parameters.detectInvertedMarker = True
detector_parameters.errorCorrectionRate = 0.6

if hasattr(
    detector_parameters,
    "useAruco3Detection"
):
    detector_parameters.useAruco3Detection = True


# =========================================================
# ChArUco 검출기 생성
# =========================================================
try:
    charuco_parameters = aruco.CharucoParameters()

    # 보정값을 ChArUco 검출에도 사용
    charuco_parameters.cameraMatrix = (
        camera_matrix
    )

    charuco_parameters.distCoeffs = (
        dist_coeffs
    )

    charuco_detector = aruco.CharucoDetector(
        charuco_board,
        charuco_parameters,
        detector_parameters
    )

except (AttributeError, TypeError):
    print(
        "보정값 입력 방식이 지원되지 않아 "
        "기본 검출기를 사용합니다."
    )

    try:
        charuco_detector = aruco.CharucoDetector(
            charuco_board
        )

    except AttributeError:
        print("현재 OpenCV에 CharucoDetector가 없습니다.")
        raise SystemExit


# =========================================================
# 카메라 열기
# =========================================================
cap = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    FRAME_FPS
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():
    print(
        f"카메라를 열 수 없습니다: "
        f"{CAMERA_DEVICE}"
    )
    raise SystemExit


actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print()
print("========================================")
print("ChArUco 깊이 측정 테스트")
print(
    f"실제 카메라 해상도: "
    f"{actual_width} x {actual_height}"
)
print("q : 종료")
print("r : 거리 필터 초기화")
print("s : 현재 화면 저장")
print("========================================")


# 카메라 안정화
time.sleep(1.0)

for _ in range(10):
    cap.read()


last_print_time = 0.0


# =========================================================
# 실시간 측정 시작
# =========================================================
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            break

        display = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # -------------------------------------------------
        # ChArUco 검출
        # -------------------------------------------------
        try:
            (
                raw_charuco_corners,
                raw_charuco_ids,
                marker_corners,
                marker_ids
            ) = charuco_detector.detectBoard(
                gray
            )

        except cv2.error as error:
            print("ChArUco 검출 오류:")
            print(error)
            break


        # -------------------------------------------------
        # 마커 직접 그리기
        # -------------------------------------------------
        marker_count = draw_markers_manual(
            display,
            marker_corners,
            marker_ids
        )


        # -------------------------------------------------
        # ChArUco 데이터 정리
        # -------------------------------------------------
        (
            charuco_points,
            charuco_ids
        ) = normalize_charuco_data(
            raw_charuco_corners,
            raw_charuco_ids,
            MAX_CHARUCO_CORNERS
        )

        charuco_count = len(
            charuco_ids
        )


        # 오류가 발생했던 OpenCV 그리기 함수 대신 직접 표시
        draw_charuco_points(
            display,
            charuco_points,
            charuco_ids
        )


        pose_success = False
        filtered_x = None
        filtered_y = None
        filtered_z = None
        filtered_distance = None
        filtered_error = None


        # -------------------------------------------------
        # 보드 자세 및 깊이 계산
        # -------------------------------------------------
        if charuco_count >= MIN_CHARUCO_CORNERS:
            try:
                # ID에 대응하는 실제 보드 3차원 좌표
                object_points = board_corner_points[
                    charuco_ids
                ].reshape(-1, 1, 3)

                # 카메라 영상에서 검출된 2차원 좌표
                image_points = charuco_points.reshape(
                    -1,
                    1,
                    2
                )


                pose_success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )


                if pose_success:
                    # 자세 정밀화
                    if hasattr(
                        cv2,
                        "solvePnPRefineLM"
                    ):
                        try:
                            rvec, tvec = (
                                cv2.solvePnPRefineLM(
                                    object_points,
                                    image_points,
                                    camera_matrix,
                                    dist_coeffs,
                                    rvec,
                                    tvec
                                )
                            )

                        except cv2.error:
                            pass


                    rvec = np.asarray(
                        rvec,
                        dtype=np.float64
                    ).reshape(3, 1)

                    tvec = np.asarray(
                        tvec,
                        dtype=np.float64
                    ).reshape(3, 1)


                    rotation_matrix, _ = cv2.Rodrigues(
                        rvec
                    )


                    # 보드 중심을 카메라 좌표계로 변환
                    center_camera = (
                        rotation_matrix
                        @ board_center_object
                        + tvec
                    )


                    current_x = float(
                        center_camera[0, 0]
                    )

                    current_y = float(
                        center_camera[1, 0]
                    )

                    current_z = float(
                        center_camera[2, 0]
                    )

                    current_distance = float(
                        np.linalg.norm(
                            center_camera
                        )
                    )


                    current_error = (
                        calculate_reprojection_error(
                            object_points,
                            image_points,
                            rvec,
                            tvec,
                            camera_matrix,
                            dist_coeffs
                        )
                    )


                    # 잘못된 자세값 제외
                    if (
                        not np.isfinite(current_x)
                        or not np.isfinite(current_y)
                        or not np.isfinite(current_z)
                        or current_z <= 0
                    ):
                        pose_success = False

                    else:
                        x_history.append(
                            current_x
                        )

                        y_history.append(
                            current_y
                        )

                        z_history.append(
                            current_z
                        )

                        distance_history.append(
                            current_distance
                        )

                        if np.isfinite(current_error):
                            error_history.append(
                                current_error
                            )


                        filtered_x = float(
                            np.median(x_history)
                        )

                        filtered_y = float(
                            np.median(y_history)
                        )

                        filtered_z = float(
                            np.median(z_history)
                        )

                        filtered_distance = float(
                            np.median(
                                distance_history
                            )
                        )

                        if len(error_history) > 0:
                            filtered_error = float(
                                np.median(
                                    error_history
                                )
                            )


                        # 보드 좌표축 표시
                        try:
                            cv2.drawFrameAxes(
                                display,
                                camera_matrix,
                                dist_coeffs,
                                rvec,
                                tvec,
                                0.04,
                                2
                            )

                        except cv2.error:
                            pass


                        # 보드 중심의 영상 좌표 계산
                        center_image, _ = (
                            cv2.projectPoints(
                                board_center_object.reshape(
                                    1,
                                    1,
                                    3
                                ),
                                rvec,
                                tvec,
                                camera_matrix,
                                dist_coeffs
                            )
                        )

                        center_pixel = np.asarray(
                            center_image,
                            dtype=np.float64
                        ).reshape(-1, 2)[0]

                        center_u = int(
                            round(center_pixel[0])
                        )

                        center_v = int(
                            round(center_pixel[1])
                        )


                        cv2.circle(
                            display,
                            (center_u, center_v),
                            8,
                            (0, 0, 255),
                            -1
                        )

                        cv2.circle(
                            display,
                            (center_u, center_v),
                            10,
                            (255, 255, 255),
                            2
                        )


            except (
                cv2.error,
                ValueError,
                IndexError
            ) as error:
                pose_success = False

                print(
                    "자세 계산 오류:",
                    error
                )


        # -------------------------------------------------
        # 화면 정보 표시
        # -------------------------------------------------
        cv2.putText(
            display,
            f"markers: {marker_count}/6",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            (
                f"charuco corners: "
                f"{charuco_count}/"
                f"{MAX_CHARUCO_CORNERS}"
            ),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )


        if pose_success:
            status_text = "POSE + DEPTH OK"
            status_color = (0, 255, 0)

        elif charuco_count >= MIN_CHARUCO_CORNERS:
            status_text = "POSE FAILED"
            status_color = (0, 0, 255)

        else:
            status_text = "NEED 4+ CORNERS"
            status_color = (0, 165, 255)


        cv2.putText(
            display,
            status_text,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA
        )


        if pose_success:
            cv2.putText(
                display,
                f"X: {filtered_x * 100:.2f} cm",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Y: {filtered_y * 100:.2f} cm",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                (
                    f"Z depth: "
                    f"{filtered_z * 100:.2f} cm"
                ),
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                (
                    f"3D distance: "
                    f"{filtered_distance * 100:.2f} cm"
                ),
                (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

            if filtered_error is not None:
                cv2.putText(
                    display,
                    (
                        f"reprojection: "
                        f"{filtered_error:.3f} px"
                    ),
                    (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


            current_time = time.monotonic()

            if (
                current_time
                - last_print_time
                >= 1.0
            ):
                terminal_text = (
                    f"X={filtered_x * 100:.2f}cm, "
                    f"Y={filtered_y * 100:.2f}cm, "
                    f"Z={filtered_z * 100:.2f}cm, "
                    f"distance="
                    f"{filtered_distance * 100:.2f}cm"
                )

                if filtered_error is not None:
                    terminal_text += (
                        f", error="
                        f"{filtered_error:.3f}px"
                    )

                print(terminal_text)

                last_print_time = current_time


        # -------------------------------------------------
        # 화면 출력 및 키 입력
        # -------------------------------------------------
        cv2.imshow(
            "ChArUco Depth Test",
            display
        )

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break


        if key == ord("r"):
            reset_filter()
            print("거리 필터 초기화")


        if key == ord("s"):
            save_path = (
                WORK_DIR
                / (
                    f"charuco_depth_"
                    f"{int(time.time())}.jpg"
                )
            )

            cv2.imwrite(
                str(save_path),
                display
            )

            print(
                f"화면 저장: {save_path}"
            )


except KeyboardInterrupt:
    print("\nCtrl+C 입력 → 종료")


finally:
    cap.release()
    cv2.destroyAllWindows()

    print("ChArUco 깊이 측정 종료")