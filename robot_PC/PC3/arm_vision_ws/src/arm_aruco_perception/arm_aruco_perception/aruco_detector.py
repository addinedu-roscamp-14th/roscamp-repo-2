from __future__ import annotations

import cv2
import numpy as np


class ArucoDetector:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        marker_length_m=0.030,
        dictionary_id=None,
        target_id=None,
        max_reprojection_error_px=2.5,
    ):
        self.camera_matrix = np.asarray(
            camera_matrix,
            dtype=np.float64,
        ).reshape(3, 3)

        self.dist_coeffs = np.asarray(
            dist_coeffs,
            dtype=np.float64,
        )

        self.marker_length_m = float(marker_length_m)

        self.target_id = (
            None
            if target_id is None
            else int(target_id)
        )

        self.max_reprojection_error_px = float(
            max_reprojection_error_px
        )

        # ArUco dictionary 기본값
        if dictionary_id is None:
            dictionary_id = cv2.aruco.DICT_4X4_50

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            int(dictionary_id)
        )

        # OpenCV 버전에 따라 DetectorParameters 생성 방식 대응
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()

        # 현재 설치된 OpenCV 환경에서 아래 설정 시
        # segmentation fault가 발생했으므로 비활성화한다.
        #
        # self.parameters.cornerRefinementMethod = (
        #     cv2.aruco.CORNER_REFINE_SUBPIX
        # )

        # OpenCV 신형 API 지원 여부 확인
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary,
                self.parameters,
            )
        else:
            # 구형 OpenCV에서는 detectMarkers() 함수 사용
            self.detector = None

    def detect(self, frame):
        if frame is None:
            return []

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # OpenCV 신형 API
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(
                gray
            )

        # OpenCV 구형 API
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.parameters,
            )

        if ids is None:
            return []

        output = []

        half = self.marker_length_m / 2.0

        # ArUco 실제 3D 코너 좌표
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        for marker_corners, marker_id in zip(
            corners,
            ids.reshape(-1),
        ):
            marker_id = int(marker_id)

            # 특정 marker ID만 사용할 경우 필터링
            if (
                self.target_id is not None
                and marker_id != self.target_id
            ):
                continue

            image_points = np.asarray(
                marker_corners,
                dtype=np.float64,
            ).reshape(4, 2)

            # 카메라 기준 ArUco pose 계산
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=getattr(
                    cv2,
                    "SOLVEPNP_IPPE_SQUARE",
                    cv2.SOLVEPNP_ITERATIVE,
                ),
            )

            if not ok:
                continue

            # 재투영 오차 계산
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )

            projected = projected.reshape(4, 2)

            error = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected - image_points) ** 2,
                            axis=1,
                        )
                    )
                )
            )

            # 재투영 오차가 너무 크면 사용하지 않음
            if error > self.max_reprojection_error_px:
                continue

            # Rodrigues vector → rotation matrix
            rotation, _ = cv2.Rodrigues(rvec)

            # 4x4 homogeneous transform
            transform = np.eye(
                4,
                dtype=np.float64,
            )

            transform[:3, :3] = rotation

            transform[:3, 3] = np.asarray(
                tvec,
                dtype=np.float64,
            ).reshape(3)

            output.append(
                {
                    "marker_id": marker_id,
                    "transform_camera_marker": transform,
                    "reprojection_error_px": error,
                }
            )

        return output