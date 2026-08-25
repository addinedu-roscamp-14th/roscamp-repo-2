from __future__ import annotations


class PerceptionNotStable(RuntimeError):
    pass


class SafePoseProvider:
    """Binds a stable camera pose, robot TCP coords, Hand-Eye and slot model."""

    def __init__(self, stabilizer, handeye, slot_calculator, robot_client):
        self.stabilizer = stabilizer
        self.handeye = handeye
        self.slot_calculator = slot_calculator
        self.robot_client = robot_client

    def targets_for_slot(self, slot):
        stable = self.stabilizer.stable_pose()
        if stable is None:
            raise PerceptionNotStable("ArUco pose is not stable; coordinate transmission blocked")
        robot_coords = self.robot_client.get_coords()
        base_marker = self.handeye.base_marker(robot_coords, stable["transform"])
        return self.slot_calculator.calculate_approach_grasp_place(base_marker, slot)
