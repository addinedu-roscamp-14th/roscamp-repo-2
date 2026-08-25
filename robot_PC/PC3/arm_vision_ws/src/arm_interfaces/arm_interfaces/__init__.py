from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StableMarkerPose:
    side: str
    marker_id: int
    transform_camera_marker: Any
    translation_spread_mm: float
    rotation_spread_deg: float


@dataclass(frozen=True)
class MotionTarget:
    side: str
    slot: str
    stage: str
    coords: tuple[float, float, float, float, float, float]
