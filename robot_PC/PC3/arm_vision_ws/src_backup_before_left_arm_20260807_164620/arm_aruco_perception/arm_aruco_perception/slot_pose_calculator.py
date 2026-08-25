from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


class SlotCalibrationError(RuntimeError):
    pass


def transform_to_coords(transform):
    t = np.asarray(transform, dtype=float).reshape(4, 4)
    r = t[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy >= 1e-9:
        roll, pitch, yaw = (
            math.atan2(r[2, 1], r[2, 2]),
            math.atan2(-r[2, 0], sy),
            math.atan2(r[1, 0], r[0, 0]),
        )
    else:
        roll, pitch, yaw = math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0
    xyz = t[:3, 3] * 1000.0
    return [float(v) for v in (*xyz, *np.rad2deg([roll, pitch, yaw]))]


class SlotPoseCalculator:
    """Reprojects taught marker-relative poses at the current marker pose."""

    def __init__(self, registration_file):
        path = Path(registration_file)
        if not path.is_file():
            raise SlotCalibrationError(f"slot registration file is missing: {path}")
        self.data = json.loads(path.read_text(encoding="utf-8"))

    def calculate(self, transform_base_marker, slot, stage):
        entry = self.data.get("slots", {}).get(str(slot), {}).get(stage)
        if not entry:
            raise SlotCalibrationError(
                f"slot {slot} stage {stage} is not calibrated"
            )
        relative = entry.get("T_marker_gripper")
        if relative is None:
            raise SlotCalibrationError(
                f"slot {slot} stage {stage} has no T_marker_gripper"
            )
        target = np.asarray(transform_base_marker) @ np.asarray(relative)
        return transform_to_coords(target)

    def calculate_approach_grasp_place(self, transform_base_marker, slot):
        grasp = self.calculate(transform_base_marker, slot, "grasp")
        approach = self.calculate(transform_base_marker, slot, "approach")
        place_entry = self.data.get("slots", {}).get(str(slot), {}).get("place")
        place = (
            self.calculate(transform_base_marker, slot, "place")
            if place_entry else grasp
        )
        return {"approach": approach, "grasp": grasp, "place": place}
