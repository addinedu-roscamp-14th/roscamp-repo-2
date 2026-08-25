"""Registered marker-relative poses for each transport slot."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from arm_aruco_perception.slot_pose_calculator import SlotPoseCalculator


class SlotRegistration:
    def __init__(self, registration_file):
        self.path = Path(registration_file)
        if not self.path.is_file():
            raise FileNotFoundError(f"slot registration file is missing: {self.path}")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.calculator = SlotPoseCalculator(self.path)

    def marker_reference_coords(self):
        values = self.data.get("marker_reference", {}).get("robot_coords_mm_deg")
        if values is None:
            raise ValueError("slot registration has no marker_reference robot coords")
        return [float(value) for value in values]

    def relative_pose(self, slot, stage):
        entry = self.data.get("slots", {}).get(str(int(slot)), {}).get(stage)
        if not entry or "T_marker_gripper" not in entry:
            raise KeyError(f"slot {slot} stage {stage} has no T_marker_gripper")
        transform = np.asarray(entry["T_marker_gripper"], dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(transform)):
            raise ValueError(f"slot {slot} stage {stage} contains NaN or Inf")
        return transform

    def targets_from_marker(self, base_marker, slot):
        result = {}
        for stage in ("approach", "grasp", "place"):
            try:
                relative = self.relative_pose(slot, stage)
            except KeyError:
                if stage == "place":
                    relative = self.relative_pose(slot, "grasp")
                else:
                    raise
            result[stage] = np.asarray(base_marker, dtype=float) @ relative
        return result
