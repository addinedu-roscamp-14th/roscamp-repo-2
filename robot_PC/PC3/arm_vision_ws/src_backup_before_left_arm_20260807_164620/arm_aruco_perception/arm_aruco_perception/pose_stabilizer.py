from __future__ import annotations
from collections import deque
import math
import numpy as np


def rotation_distance_deg(a, b):
    value = np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def average_rotation(rotations):
    u, _, vt = np.linalg.svd(np.mean(np.stack(rotations), axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


class PoseStabilizer:
    def __init__(self, buffer_size=12, min_samples=8,
                 max_translation_spread_mm=3.0, max_rotation_spread_deg=2.0):
        self.samples = deque(maxlen=int(buffer_size))
        self.min_samples = int(min_samples)
        self.max_translation_spread_mm = float(max_translation_spread_mm)
        self.max_rotation_spread_deg = float(max_rotation_spread_deg)

    def reset(self):
        self.samples.clear()

    def update(self, transform):
        value = np.asarray(transform, dtype=float).reshape(4, 4)
        self.samples.append(value)
        return self.stable_pose()

    def stable_pose(self):
        if len(self.samples) < self.min_samples:
            return None
        translations = np.stack([item[:3, 3] for item in self.samples])
        center = np.median(translations, axis=0)
        spread_mm = float(np.max(np.linalg.norm(translations - center, axis=1)) * 1000.0)
        rotation = average_rotation([item[:3, :3] for item in self.samples])
        rotation_spread = max(
            rotation_distance_deg(rotation, item[:3, :3]) for item in self.samples
        )
        if spread_mm > self.max_translation_spread_mm:
            return None
        if rotation_spread > self.max_rotation_spread_deg:
            return None
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = center
        return {
            "transform": result,
            "translation_spread_mm": spread_mm,
            "rotation_spread_deg": float(rotation_spread),
            "samples": len(self.samples),
        }
