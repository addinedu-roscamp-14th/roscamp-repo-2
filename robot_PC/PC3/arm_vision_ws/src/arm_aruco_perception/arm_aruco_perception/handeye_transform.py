from __future__ import annotations
from pathlib import Path
import math
import numpy as np


class CalibrationUnavailable(RuntimeError):
    pass


def make_transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def robot_coords_to_transform(coords):
    if len(coords) != 6:
        raise ValueError("robot coords must contain six values")
    x, y, z, rx, ry, rz = [float(value) for value in coords]
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return make_transform(rot_z @ rot_y @ rot_x, np.array([x, y, z]) / 1000.0)


class HandEyeTransform:
    def __init__(self, calibration_file):
        self.path = Path(calibration_file)
        if not self.path.is_file():
            raise CalibrationUnavailable(
                f"Hand-Eye calibration file is missing: {self.path}. "
                "Movement is blocked; do not reuse the opposite arm calibration."
            )
        data = np.load(self.path, allow_pickle=True)
        if "T_gripper_camera" in data:
            self.transform_gripper_camera = np.asarray(
                data["T_gripper_camera"], dtype=float
            ).reshape(4, 4)
        elif "R_cam2gripper" in data and "t_cam2gripper_m" in data:
            self.transform_gripper_camera = make_transform(
                data["R_cam2gripper"], data["t_cam2gripper_m"]
            )
        else:
            raise CalibrationUnavailable(
                f"invalid Hand-Eye calibration keys in {self.path}"
            )

    def base_marker(self, robot_coords, transform_camera_marker):
        return (
            robot_coords_to_transform(robot_coords)
            @ self.transform_gripper_camera
            @ np.asarray(transform_camera_marker, dtype=float).reshape(4, 4)
        )
