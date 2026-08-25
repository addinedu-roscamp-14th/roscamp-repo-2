from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from central_control.arm_client import ArmClient, ArmCommandResult


class DualArmManager:
    def __init__(self, left: ArmClient, right: ArmClient):
        self.left = left
        self.right = right

    @classmethod
    def from_config(cls, network_config: dict, operation_config: dict):
        mock_arms = bool(operation_config.get("operation", {}).get("mock_arms", True))
        left_cfg = network_config["left_arm"]
        right_cfg = network_config["right_arm"]
        return cls(
            ArmClient("left", mock_mode=mock_arms, **left_cfg),
            ArmClient("right", mock_mode=mock_arms, **right_cfg),
        )

    def send_left(self, command: str) -> dict[str, ArmCommandResult]:
        return {"left": self.left.send_command(command)}
    def send_side(self, side: str, command: str) -> dict[str, ArmCommandResult]:
        if side == "left":
            return self.send_left(command)
        if side == "right":
            return self.send_right(command)
        raise ValueError(f"unknown arm side: {side}")


    def send_right(self, command: str) -> dict[str, ArmCommandResult]:
        return {"right": self.right.send_command(command)}

    def send_both(self, command: str) -> dict[str, ArmCommandResult]:
        return self.send_selected(left_command=command, right_command=command)

    def send_selected(
        self,
        left_command: str | None = None,
        right_command: str | None = None,
    ) -> dict[str, ArmCommandResult]:
        tasks = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if left_command:
                future = executor.submit(self.left.send_command, left_command)
                tasks[future] = ("left", left_command)
            if right_command:
                future = executor.submit(self.right.send_command, right_command)
                tasks[future] = ("right", right_command)

            results = {}
            for future in as_completed(tasks):
                side, command = tasks[future]
                try:
                    results[side] = future.result()
                except Exception as error:
                    results[side] = ArmCommandResult(
                        arm=side,
                        command=command,
                        success=False,
                        error=f"arm command raised: {error}",
                    )
        return results

    @staticmethod
    def all_success(results: dict[str, ArmCommandResult]) -> bool:
        return bool(results) and all(result.success for result in results.values())
