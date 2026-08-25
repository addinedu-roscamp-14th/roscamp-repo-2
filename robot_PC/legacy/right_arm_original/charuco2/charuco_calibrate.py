import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


# =========================================================
# ChArUco 보드 실제 규격
# =========================================================
SQUARES_X = 4
SQUARES_Y = 3

# 단위: 미터
SQUARE_LENGTH = 0.04   # 체스판 한 칸 4cm
MARKER_LENGTH = 0.03   # ArUco 마커 한 변 3cm

ARUCO_DICTIONARY = aruco.DICT_4X4_50


# =========================================================
# 보정 조건
# =========================================================
# 한 사진에서 최소 몇 개의 ChArUco 코너가 필요할지
MIN_CHARUCO_CORNERS = 4

# 최소 유효 사진 수
MIN_VALID_IMAGES = 10

# 사용할 사진 확장자
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# =========================================================
# 경로 설정
# =========================================================
# 이 파이썬 파일이 들어 있는 폴더
WORK_DIR = Path(__file__).resolve().parent

# 결과 저장 파일
OUTPUT_NPZ = WORK_DIR / "camera_calibration.npz"
OUTPUT_REPORT = WORK_DIR / "camera_calibration_report.txt"


# =========================================================
# 보조 함수
# =========================================================
def data_exists(data) -> bool:
    """OpenCV 반환 데이터가 실제로 존재하는지 확인."""
    if data is None:
        return False

    return np.asarray(data).size > 0


def find_images(folder: Path) -> list[Path]:
    """현재 코드 폴더에서 보정용 사진을 찾는다."""
    image_paths = []

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_paths.append(path)

    return sorted(
        image_paths,
        key=lambda path: path.name
    )


def calculate_reprojection_errors(
    object_points_list,
    image_points_list,
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs
):
    """각 사진의 재투영 오차를 계산한다."""
    errors = []

    for index in range(len(object_points_list)):
        projected_points, _ = cv2.projectPoints(
            object_points_list[index],
            rvecs[index],
            tvecs[index],
            camera_matrix,
            dist_coeffs
        )

        detected_points = np.asarray(
            image_points_list[index],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        projected_points = np.asarray(
            projected_points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        point_count = len(projected_points)

        if point_count == 0:
            errors.append(float("nan"))
            continue

        l2_error = cv2.norm(
            detected_points,
            projected_points,
            cv2.NORM_L2
        )

        rms_error = float(
            l2_error / np.sqrt(point_count)
        )

        errors.append(rms_error)

    return errors


# =========================================================
# 시작 정보
# =========================================================
print("============================================")
print("ChArUco 카메라 보정 시작")
print("============================================")
print(f"작업 폴더: {WORK_DIR}")
print(f"보드: {SQUARES_X} x {SQUARES_Y}")
print(f"체스판 한 칸: {SQUARE_LENGTH * 100:.1f} cm")
print(f"마커 한 변: {MARKER_LENGTH * 100:.1f} cm")
print()


# =========================================================
# 사진 검색
# =========================================================
image_files = find_images(WORK_DIR)

if not image_files:
    print("같은 폴더에서 보정용 사진을 찾지 못했습니다.")
    print("지원 확장자: jpg, jpeg, png, bmp")
    sys.exit(1)


print(f"찾은 사진 수: {len(image_files)}장")
print()


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


# =========================================================
# 검출 설정
# =========================================================
detector_parameters = aruco.DetectorParameters()

# 카메라 보정 전이므로 마커 코너 서브픽셀 보정은 끈다.
# ChArUco 내부 코너는 detectBoard에서 계산된다.
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

    charuco_detector = aruco.CharucoDetector(
        charuco_board,
        charuco_parameters,
        detector_parameters
    )

except (TypeError, AttributeError):
    try:
        charuco_detector = aruco.CharucoDetector(
            charuco_board
        )

    except AttributeError:
        print("현재 OpenCV에 CharucoDetector가 없습니다.")
        print("opencv-contrib-python 설치 여부를 확인하세요.")
        sys.exit(1)


# =========================================================
# 각 사진에서 코너 검출
# =========================================================
object_points_list = []
image_points_list = []

used_image_names = []
used_corner_counts = []

skipped_image_names = []
skipped_reasons = []

image_size = None


for index, image_path in enumerate(
    image_files,
    start=1
):
    image = cv2.imread(str(image_path))

    if image is None:
        reason = "사진 읽기 실패"

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"제외: {image_path.name} - {reason}"
        )

        skipped_image_names.append(image_path.name)
        skipped_reasons.append(reason)
        continue


    height, width = image.shape[:2]
    current_image_size = (width, height)


    # 첫 사진의 해상도를 전체 기준으로 사용
    if image_size is None:
        image_size = current_image_size

        print(
            f"기준 사진 해상도: "
            f"{image_size[0]} x {image_size[1]}"
        )
        print()


    # 해상도가 다른 사진 제외
    if current_image_size != image_size:
        reason = (
            f"해상도 다름 "
            f"{current_image_size[0]}x{current_image_size[1]}"
        )

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"제외: {image_path.name} - {reason}"
        )

        skipped_image_names.append(image_path.name)
        skipped_reasons.append(reason)
        continue


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # ChArUco 검출
    # 반환 순서:
    # charucoCorners, charucoIds,
    # markerCorners, markerIds
    # -----------------------------------------------------
    try:
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids
        ) = charuco_detector.detectBoard(gray)

    except cv2.error as error:
        reason = "OpenCV 검출 오류"

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"제외: {image_path.name} - {reason}"
        )
        print(error)

        skipped_image_names.append(image_path.name)
        skipped_reasons.append(reason)
        continue


    marker_count = 0
    charuco_corner_count = 0

    if data_exists(marker_ids):
        marker_count = int(
            np.asarray(marker_ids).reshape(-1).size
        )

    if data_exists(charuco_ids):
        charuco_corner_count = int(
            np.asarray(charuco_ids).reshape(-1).size
        )


    if (
        not data_exists(charuco_corners)
        or not data_exists(charuco_ids)
    ):
        reason = (
            f"ChArUco 검출 실패 "
            f"(마커 {marker_count}/6)"
        )

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"제외: {image_path.name} - {reason}"
        )

        skipped_image_names.append(image_path.name)
        skipped_reasons.append(reason)
        continue


    if charuco_corner_count < MIN_CHARUCO_CORNERS:
        reason = (
            f"코너 부족 "
            f"{charuco_corner_count}/6"
        )

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"제외: {image_path.name} - {reason}"
        )

        skipped_image_names.append(image_path.name)
        skipped_reasons.append(reason)
        continue


    # -----------------------------------------------------
    # 검출된 ChArUco 코너와 실제 보드 좌표 연결
    # -----------------------------------------------------
    try:
        (
            object_points,
            image_points
        ) = charuco_board.matchImagePoints(
            charuco_corners,
            charuco_ids
        )

    except (cv2.error, AttributeError):
        # matchImagePoints를 지원하지 않는 버전용 대체 방법
        board_points = np.asarray(
            charuco_board.getChessboardCorners(),
            dtype=np.float32
        ).reshape(-1, 3)

        ids_array = np.asarray(
            charuco_ids,
            dtype=np.int32
        ).reshape(-1)

        if np.any(ids_array < 0) or np.any(
            ids_array >= len(board_points)
        ):
            reason = "잘못된 ChArUco 코너 ID"

            print(
                f"[{index:02d}/{len(image_files):02d}] "
                f"제외: {image_path.name} - {reason}"
            )

            skipped_image_names.append(image_path.name)
            skipped_reasons.append(reason)
            continue

        object_points = board_points[
            ids_array
        ]

        image_points = np.asarray(
            charuco_corners,
            dtype=np.float32
        ).reshape(-1, 2)


    object_points = np.asarray(
        object_points,
        dtype=np.float32
    ).reshape(-1, 1, 3)

    image_points = np.asarray(
        image_points,
        dtype=np.float32
    ).reshape(-1, 1, 2)


    if len(object_points) != len(image_points):
        reason = "실제 좌표와 영상 좌표 개수 불일치"

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"제외: {image_path.name} - {reason}"
        )

        skipped_image_names.append(image_path.name)
        skipped_reasons.append(reason)
        continue


    object_points_list.append(object_points)
    image_points_list.append(image_points)

    used_image_names.append(image_path.name)
    used_corner_counts.append(charuco_corner_count)


    print(
        f"[{index:02d}/{len(image_files):02d}] "
        f"사용: {image_path.name} | "
        f"마커={marker_count}/6 | "
        f"코너={charuco_corner_count}/6"
    )


# =========================================================
# 유효 사진 확인
# =========================================================
valid_image_count = len(object_points_list)
skipped_image_count = len(skipped_image_names)

print()
print("============================================")
print("사진 검출 결과")
print("============================================")
print(f"전체 사진: {len(image_files)}장")
print(f"사용 사진: {valid_image_count}장")
print(f"제외 사진: {skipped_image_count}장")
print()


if valid_image_count < MIN_VALID_IMAGES:
    print(
        f"유효한 사진이 부족합니다. "
        f"최소 {MIN_VALID_IMAGES}장 이상 필요합니다."
    )

    sys.exit(1)


if image_size is None:
    print("사진 해상도를 확인하지 못했습니다.")
    sys.exit(1)


# =========================================================
# 카메라 보정
# =========================================================
print("카메라 보정 계산 중...")
print()


try:
    (
        rms_error,
        camera_matrix,
        dist_coeffs,
        rotation_vectors,
        translation_vectors
    ) = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None
    )

except cv2.error as error:
    print("카메라 보정 계산에 실패했습니다.")
    print(error)
    sys.exit(1)


# =========================================================
# 사진별 재투영 오차 계산
# =========================================================
view_errors = calculate_reprojection_errors(
    object_points_list=object_points_list,
    image_points_list=image_points_list,
    rvecs=rotation_vectors,
    tvecs=translation_vectors,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

finite_errors = np.asarray(
    [
        error
        for error in view_errors
        if np.isfinite(error)
    ],
    dtype=np.float64
)


if finite_errors.size > 0:
    mean_view_error = float(
        np.mean(finite_errors)
    )

    median_view_error = float(
        np.median(finite_errors)
    )

    max_view_error = float(
        np.max(finite_errors)
    )

else:
    mean_view_error = float("nan")
    median_view_error = float("nan")
    max_view_error = float("nan")


# 가장 오차가 큰 사진 확인
error_ranking = sorted(
    zip(
        used_image_names,
        view_errors
    ),
    key=lambda item: (
        -1.0
        if not np.isfinite(item[1])
        else item[1]
    ),
    reverse=True
)


# =========================================================
# NPZ 결과 저장
# =========================================================
np.savez(
    OUTPUT_NPZ,

    camera_matrix=np.asarray(
        camera_matrix,
        dtype=np.float64
    ),

    dist_coeffs=np.asarray(
        dist_coeffs,
        dtype=np.float64
    ),

    image_width=np.int32(
        image_size[0]
    ),

    image_height=np.int32(
        image_size[1]
    ),

    rms_error=np.float64(
        rms_error
    ),

    mean_view_error=np.float64(
        mean_view_error
    ),

    median_view_error=np.float64(
        median_view_error
    ),

    max_view_error=np.float64(
        max_view_error
    ),

    squares_x=np.int32(
        SQUARES_X
    ),

    squares_y=np.int32(
        SQUARES_Y
    ),

    square_length=np.float64(
        SQUARE_LENGTH
    ),

    marker_length=np.float64(
        MARKER_LENGTH
    ),

    used_images=np.asarray(
        used_image_names
    ),

    used_corner_counts=np.asarray(
        used_corner_counts,
        dtype=np.int32
    ),

    view_errors=np.asarray(
        view_errors,
        dtype=np.float64
    ),

    skipped_images=np.asarray(
        skipped_image_names
    ),

    skipped_reasons=np.asarray(
        skipped_reasons
    )
)


# =========================================================
# 텍스트 보고서 저장
# =========================================================
report_lines = []

report_lines.append(
    "ChArUco Camera Calibration Report"
)
report_lines.append(
    "=" * 50
)

report_lines.append(
    f"Work directory: {WORK_DIR}"
)

report_lines.append(
    f"Image size: {image_size[0]} x {image_size[1]}"
)

report_lines.append(
    f"Total images: {len(image_files)}"
)

report_lines.append(
    f"Used images: {valid_image_count}"
)

report_lines.append(
    f"Skipped images: {skipped_image_count}"
)

report_lines.append("")
report_lines.append(
    f"RMS error: {rms_error}"
)

report_lines.append(
    f"Mean view error: {mean_view_error}"
)

report_lines.append(
    f"Median view error: {median_view_error}"
)

report_lines.append(
    f"Max view error: {max_view_error}"
)

report_lines.append("")
report_lines.append("Camera matrix:")
report_lines.append(
    np.array2string(
        camera_matrix,
        precision=10,
        suppress_small=False
    )
)

report_lines.append("")
report_lines.append("Distortion coefficients:")
report_lines.append(
    np.array2string(
        dist_coeffs,
        precision=10,
        suppress_small=False
    )
)

report_lines.append("")
report_lines.append("Per-image errors:")

for image_name, error in error_ranking:
    report_lines.append(
        f"{image_name}: {error}"
    )


if skipped_image_names:
    report_lines.append("")
    report_lines.append("Skipped images:")

    for image_name, reason in zip(
        skipped_image_names,
        skipped_reasons
    ):
        report_lines.append(
            f"{image_name}: {reason}"
        )


OUTPUT_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# =========================================================
# 결과 출력
# =========================================================
print("============================================")
print("카메라 보정 완료")
print("============================================")
print(f"사용 사진 수: {valid_image_count}장")
print(
    f"사진 해상도: "
    f"{image_size[0]} x {image_size[1]}"
)
print()

print(f"RMS 오차: {rms_error:.6f}")
print(
    f"평균 사진별 오차: "
    f"{mean_view_error:.6f}"
)
print(
    f"중앙값 사진별 오차: "
    f"{median_view_error:.6f}"
)
print(
    f"최대 사진별 오차: "
    f"{max_view_error:.6f}"
)
print()

print("camera_matrix:")
print(camera_matrix)
print()

print("dist_coeffs:")
print(dist_coeffs)
print()

print("오차가 큰 사진 상위 5개:")

for image_name, error in error_ranking[:5]:
    print(
        f"  {image_name}: {error:.6f}"
    )

print()
print(f"보정 파일 저장: {OUTPUT_NPZ}")
print(f"보고서 저장: {OUTPUT_REPORT}")
print("============================================")