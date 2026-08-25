"""Hand-eye projection and robot-coordinate safety validation."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from arm_aruco_perception.handeye_transform import (
    HandEyeTransform,
)


def finite_transform(value):
    result = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(result)):
        raise ValueError("transform contains NaN or Inf")
    return result


class CoordinateTransformer:
    def __init__(self, handeye_file):
        self.handeye_file = str(Path(handeye_file))
        self.handeye = HandEyeTransform(handeye_file)

    def base_marker(self, robot_coords, camera_marker):
        coords = np.asarray(robot_coords, dtype=float).reshape(-1)
        if coords.size != 6 or not np.all(np.isfinite(coords)):
            raise ValueError("current robot coords must be six finite values")
        return finite_transform(self.handeye.base_marker(coords, finite_transform(camera_marker)))

    def target_from_marker(self, robot_coords, camera_marker, relative_marker_gripper):
        base_marker = self.base_marker(robot_coords, camera_marker)
        target = base_marker @ finite_transform(relative_marker_gripper)
        if not np.all(np.isfinite(target)):
            raise ValueError("target transform contains NaN or Inf")
        return target


def validate_robot_coords(coords, position_limits_mm, rotation_limits_deg):
    values = np.asarray(coords, dtype=float).reshape(-1)
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise ValueError("target coords must contain six finite values")
    xmin, xmax, zmin, zmax = map(float, position_limits_mm)
    rmin, rmax = map(float, rotation_limits_deg)
    if not (xmin <= values[0] <= xmax and xmin <= values[1] <= xmax and zmin <= values[2] <= zmax):
        raise ValueError(f"target position is outside configured limits: {values[:3].tolist()}")
    if not np.all((values[3:] >= rmin) & (values[3:] <= rmax)):
        raise ValueError(f"target rotation is outside configured limits: {values[3:].tolist()}")
    return [float(value) for value in values]
