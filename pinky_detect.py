import math

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "/home/parkjh/pinky_yolo_run/best.pt"
HOMOGRAPHY_PATH = "/home/parkjh/pinky_yolo_run/floor_homography.npz"
CAMERA_INDEX = 2

# 저장된 최종 주차 목표
TARGET_X = 0.836
TARGET_Y = 1.017
TARGET_YAW = 90.0

CONFIDENCE = 0.5
IMAGE_SIZE = 960


model = YOLO(MODEL_PATH)

homography_data = np.load(HOMOGRAPHY_PATH)

if "homography" not in homography_data.files:
    raise KeyError("floor_homography.npz에 homography 키가 없습니다.")

H = homography_data["homography"]


def pixel_to_world(u, v):
    point = np.array([[[u, v]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)

    world_x = float(transformed[0, 0, 0])
    world_y = float(transformed[0, 0, 1])

    return world_x, world_y


def angle_error_180(target, current):
    """
    현재 마스크 방식은 Pinky 앞뒤를 구분하지 못하므로
    180도 주기 기준으로 가장 작은 각도 오차를 계산한다.
    """
    return (target - current + 90.0) % 180.0 - 90.0


cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(
        f"/dev/video{CAMERA_INDEX} 카메라를 열 수 없습니다."
    )

print("목표 주차 자세")
print(
    f"x={TARGET_X:.3f}, "
    f"y={TARGET_Y:.3f}, "
    f"yaw={TARGET_YAW:.1f}도"
)
print("q: 종료")

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]
    display_frame = frame.copy()

    if result.masks is not None and len(result.masks.xy) > 0:
        # 검출된 객체 중 가장 큰 마스크 사용
        polygons = [
            polygon.astype(np.float32)
            for polygon in result.masks.xy
        ]

        polygon = max(
            polygons,
            key=cv2.contourArea
        )

        polygon_int = polygon.astype(np.int32)

        cv2.polylines(
            display_frame,
            [polygon_int],
            True,
            (0, 255, 0),
            2
        )

        moments = cv2.moments(polygon)

        if moments["m00"] != 0:
            center_u = int(
                moments["m10"] / moments["m00"]
            )
            center_v = int(
                moments["m01"] / moments["m00"]
            )

            current_x, current_y = pixel_to_world(
                center_u,
                center_v
            )

            rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rect).astype(np.int32)

            width, height = rect[1]
            angle = rect[2]

            if width < height:
                current_yaw = angle
            else:
                current_yaw = angle + 90.0

            current_yaw %= 180.0

            error_x = TARGET_X - current_x
            error_y = TARGET_Y - current_y
            distance_error = math.hypot(
                error_x,
                error_y
            )
            yaw_error = angle_error_180(
                TARGET_YAW,
                current_yaw
            )

            cv2.drawContours(
                display_frame,
                [box],
                0,
                (255, 0, 0),
                2
            )

            cv2.circle(
                display_frame,
                (center_u, center_v),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display_frame,
                f"current: ({current_x:.3f}, {current_y:.3f}) m",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"current yaw: {current_yaw:.1f} deg",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"target: ({TARGET_X:.3f}, {TARGET_Y:.3f}) m",
                (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"target yaw: {TARGET_YAW:.1f} deg",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"error x: {error_x:+.3f} m",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"error y: {error_y:+.3f} m",
                (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"distance: {distance_error:.3f} m",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"yaw error: {yaw_error:+.1f} deg",
                (30, 295),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if distance_error <= 0.01:
                position_status = "POSITION OK"
                position_color = (0, 255, 0)
            else:
                position_status = "POSITION MOVE"
                position_color = (0, 0, 255)

            if abs(yaw_error) <= 3.0:
                yaw_status = "YAW OK"
                yaw_color = (0, 255, 0)
            else:
                yaw_status = "YAW ROTATE"
                yaw_color = (0, 0, 255)

            cv2.putText(
                display_frame,
                position_status,
                (30, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                position_color,
                2
            )

            cv2.putText(
                display_frame,
                yaw_status,
                (30, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                yaw_color,
                2
            )

            print(
                f"\rcurrent=({current_x:.3f}, "
                f"{current_y:.3f}) | "
                f"error=({error_x:+.3f}, "
                f"{error_y:+.3f}) | "
                f"distance={distance_error:.3f} m | "
                f"yaw={current_yaw:.1f} | "
                f"yaw_error={yaw_error:+.1f}",
                end=""
            )

    else:
        cv2.putText(
            display_frame,
            "Pinky not detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Pinky Parking Error",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n종료")
