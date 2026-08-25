from __future__ import annotations
from dataclasses import dataclass

from arm_task_coordinator.emergency_manager import EmergencyManager


@dataclass(frozen=True)
class StageCommand:
    stage: str
    command: str
    coords: tuple[float, ...] | None = None


class ReplacementCoordinator:
    """Runs complete left sequence before allowing the right sequence."""

    def __init__(self, left_client, right_client, pose_providers):
        self.clients = {"left": left_client, "right": right_client}
        self.pose_providers = dict(pose_providers)
        self.emergency = EmergencyManager(self.clients)
        self.left_completed = False
        self.left_required = False
        self.active_side = None

    def run(self, sides, slot_by_side):
        ordered = [side for side in ("left", "right") if side in set(sides)]
        self.left_required = "left" in ordered
        results = {}
        for side in ordered:
            results[side] = self.run_side(side, slot_by_side[side])
        return results

    def run_side(self, side, slot):
        self.emergency.require_clear()
        if side not in self.clients:
            raise ValueError(f"unknown side: {side}")
        if self.active_side is not None:
            raise RuntimeError(f"arm cycle already active: {self.active_side}")
        if side == "right" and self.left_required and not self.left_completed:
            raise RuntimeError("right cycle blocked until left cycle completes")

        provider = self.pose_providers[side]
        # Provider must reject unstable ArUco and unavailable Hand-Eye before any send.
        targets = provider.targets_for_slot(slot)
        required = {"approach", "grasp", "place"}
        if not required.issubset(targets):
            raise RuntimeError(f"incomplete calibrated target set for {side}: {sorted(targets)}")

        self.active_side = side
        client = self.clients[side]
        try:
            client.move_coords(targets["approach"])
            self.emergency.require_clear()
            client.move_coords(targets["grasp"], speed=15)
            client.command("GRIPPER_CLOSE")
            client.move_coords(targets["approach"])
            client.move_coords(targets["place"])
            client.command("GRIPPER_OPEN")
            client.move_coords(targets["approach"])
            client.legacy_command("HOME")
        except Exception:
            try:
                client.stop()
            finally:
                raise
        finally:
            self.active_side = None
        if side == "left":
            self.left_completed = True
        return {"status": "completed", "side": side, "slot": slot}
