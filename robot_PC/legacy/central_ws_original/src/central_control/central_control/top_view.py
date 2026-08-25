from __future__ import annotations

import os

import cv2
import numpy as np


class TopViewTransformer:
    """Undistort and warp top-camera frames into the parking BEV plane."""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("bev_enabled", False))
        self.width = int(config.get("bev_width", 600))
        self.height = int(config.get("bev_height", 400))
        self.resize_before_bev = bool(
            config.get("top_camera_resize_before_bev", False)
        )
        self.process_width = int(
            config.get("top_camera_process_width", self.width)
        )
        self.process_height = int(
            config.get("top_camera_process_height", self.height)
        )

        self.camera_matrix = None
        self.dist_coeffs = None
        self.bev_matrix = None
        if not self.enabled:
            return

        camera_matrix_path = self._required_path(
            config, "bev_camera_matrix_path"
        )
        dist_coeffs_path = self._required_path(
            config, "bev_dist_coeffs_path"
        )
        self.camera_matrix = np.load(camera_matrix_path)
        self.dist_coeffs = np.load(dist_coeffs_path)

        src_points = np.asarray(config.get("bev_src_points"), dtype=np.float32)
        if src_points.shape != (4, 2):
            raise ValueError(
                "bev_src_points must contain exactly four [x, y] points"
            )
        dst_points = np.asarray(
            [
                [0, 0],
                [self.width, 0],
                [self.width, self.height],
                [0, self.height],
            ],
            dtype=np.float32,
        )
        self.bev_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    @staticmethod
    def _required_path(config: dict, key: str) -> str:
        value = config.get(key)
        if not value:
            raise ValueError(f"{key} is required when bev_enabled is true")
        path = os.path.expanduser(os.fspath(value))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{key} not found: {path}")
        return path

    def transform(self, frame):
        if not self.enabled:
            return frame

        source = frame
        process_size = (self.process_width, self.process_height)
        if self.resize_before_bev and (
            source.shape[1] != process_size[0]
            or source.shape[0] != process_size[1]
        ):
            source = cv2.resize(
                source, process_size, interpolation=cv2.INTER_LINEAR
            )

        undistorted = cv2.undistort(
            source, self.camera_matrix, self.dist_coeffs
        )
        return cv2.warpPerspective(
            undistorted,
            self.bev_matrix,
            (self.width, self.height),
        )
