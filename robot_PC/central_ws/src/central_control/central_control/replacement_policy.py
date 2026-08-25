"""Hardware-independent policy for detection retries and left-first execution."""
from __future__ import annotations

from dataclasses import dataclass

FINAL = {"GOOD", "BAD"}
TRANSPORT_SLOTS = {
    "left": {
        "removed_tire": 3,
        "new_tire": 4,
    },
    "right": {
        "removed_tire": 1,
        "new_tire": 2,
    },
}

def transport_slot(side: str, tire_kind: str) -> int:
    try:
        return int(TRANSPORT_SLOTS[str(side)][str(tire_kind)])
    except KeyError as error:
        raise ValueError(
            f"invalid transport slot policy request: side={side} kind={tire_kind}"
        ) from error


def validate_transport_slot(side: str, tire_kind: str, slot: int) -> int:
    expected = transport_slot(side, tire_kind)
    actual = int(slot)
    if actual != expected:
        raise ValueError(
            f"invalid {side} {tire_kind} slot: {actual}; expected {expected}"
        )
    return actual


@dataclass(frozen=True)
class ReplacementPlan:
    sides: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.sides)


class DetectionPolicy:
    """Accept a pair only when both sides are final in the same response."""

    @staticmethod
    def requires_full_recheck(response: dict) -> bool:
        return any(
            response.get(side, {}).get("status") not in FINAL
            for side in ("left", "right")
        )

    @staticmethod
    def build_plan(response: dict) -> ReplacementPlan:
        if DetectionPolicy.requires_full_recheck(response):
            raise ValueError("left/right detection pair is not final; recheck both sides")
        sides = tuple(
            side for side in ("left", "right")
            if response[side]["status"] == "BAD"
        )
        return ReplacementPlan(sides)


class LeftFirstGate:
    """Refuse right-side start until the complete left cycle has finished."""

    def __init__(self, plan: ReplacementPlan):
        self.plan = plan
        self.completed: set[str] = set()
        self.active: str | None = None

    def start(self, side: str) -> None:
        if side not in self.plan.sides:
            raise ValueError(f"side is not in replacement plan: {side}")
        if self.active is not None:
            raise RuntimeError(f"replacement already active: {self.active}")
        if side == "right" and "left" in self.plan.sides and "left" not in self.completed:
            raise RuntimeError("right replacement blocked until left cycle completes")
        self.active = side

    def complete(self, side: str) -> None:
        if self.active != side:
            raise RuntimeError(f"cannot complete inactive side: {side}")
        self.completed.add(side)
        self.active = None
