from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class WorkState(str, Enum):
    WAITING_FOR_VEHICLE = "WAITING_FOR_VEHICLE"
    WAIT_SET_CAMERA_CONFIRM = "WAIT_SET_CAMERA_CONFIRM"
    SETTING_CAMERA = "SETTING_CAMERA"
    WAIT_DETECTION_CONFIRM = "WAIT_DETECTION_CONFIRM"
    DETECTING_TIRES = "DETECTING_TIRES"
    WAIT_ACTION_CONFIRM = "WAIT_ACTION_CONFIRM"
    EXECUTING_ACTION = "EXECUTING_ACTION"
    WAIT_ARUCO_CONFIRM = "WAIT_ARUCO_CONFIRM"
    SETTING_ARUCO = "SETTING_ARUCO"
    WAIT_HOME_CONFIRM = "WAIT_HOME_CONFIRM"
    RETURNING_HOME = "RETURNING_HOME"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


ALLOWED_TIRE_CLASSES = {"bad_tire", "good_tire", "none"}


@dataclass
class WorkStateMachine:
    state: WorkState = WorkState.WAITING_FOR_VEHICLE
    last_status: str = "READY"
    last_error: str = ""
    left_detection: dict | None = None
    right_detection: dict | None = None
    target_arms: set[str] = field(default_factory=set)
    busy: bool = False
    current_worker_operation: str = ""

    def reset(self):
        self.state = WorkState.WAITING_FOR_VEHICLE
        self.last_status = "RESET"
        self.last_error = ""
        self.left_detection = None
        self.right_detection = None
        self.target_arms.clear()
        self.busy = False
        self.current_worker_operation = ""

    def set_busy(self, state: WorkState, operation: str):
        if self.busy:
            return False
        self.state = state
        self.busy = True
        self.current_worker_operation = operation
        self.last_status = operation
        self.last_error = ""
        return True

    def clear_busy(self):
        self.busy = False
        self.current_worker_operation = ""

    def parking_completed(self):
        if self.state == WorkState.WAITING_FOR_VEHICLE:
            self.state = WorkState.WAIT_DETECTION_CONFIRM
            self.last_status = "PARKED: wait SPACE for tire detection"

    def start_set_camera(self) -> bool:
        return self.state == WorkState.WAIT_SET_CAMERA_CONFIRM and self.set_busy(
            WorkState.SETTING_CAMERA, "SETTING_CAMERA"
        )

    def apply_set_camera_result(self, success: bool, message: str = ""):
        self.clear_busy()
        if success:
            self.state = WorkState.WAIT_DETECTION_CONFIRM
            self.last_status = message or "SET_CAMERA complete; wait SPACE for detection"
        else:
            self.to_error(message or "SET_CAMERA failed")

    def start_detection(self) -> bool:
        return self.state == WorkState.WAIT_DETECTION_CONFIRM and self.set_busy(
            WorkState.DETECTING_TIRES, "DETECTING_TIRES"
        )

    def apply_detection_result(self, response: dict | None, error: str | None = None):
        self.clear_busy()
        if error:
            self.to_error(error)
            return
        try:
            self._validate_detection_response(response or {})
        except ValueError as exc:
            self.to_error(str(exc))
            return
        self.left_detection = response["left"]
        self.right_detection = response["right"]
        self.target_arms = self._determine_targets(self.left_detection, self.right_detection)
        if self.target_arms:
            self.state = WorkState.WAIT_ACTION_CONFIRM
            self.last_status = "Detection complete; wait SPACE for tire action"
        else:
            self.state = WorkState.WAIT_HOME_CONFIRM
            self.last_status = "Both tires good; wait SPACE for HOME"

    def start_action(self) -> bool:
        return self.state == WorkState.WAIT_ACTION_CONFIRM and self.set_busy(
            WorkState.EXECUTING_ACTION, "EXECUTING_ACTION"
        )

    def apply_action_result(self, success: bool, message: str = ""):
        self.clear_busy()
        if success:
            self.state = WorkState.WAIT_ARUCO_CONFIRM
            self.last_status = message or "Action complete; wait SPACE for SET_ARUCO"
        else:
            self.to_error(message or "tire action failed")

    def start_aruco(self) -> bool:
        return self.state == WorkState.WAIT_ARUCO_CONFIRM and self.set_busy(
            WorkState.SETTING_ARUCO, "SETTING_ARUCO"
        )

    def apply_aruco_result(self, success: bool, message: str = ""):
        self.clear_busy()
        if success:
            self.state = WorkState.WAIT_HOME_CONFIRM
            self.last_status = message or "SET_ARUCO complete; wait SPACE for HOME"
        else:
            self.to_error(message or "SET_ARUCO failed")

    def start_home(self) -> bool:
        return self.state == WorkState.WAIT_HOME_CONFIRM and self.set_busy(
            WorkState.RETURNING_HOME, "RETURNING_HOME"
        )

    def apply_home_result(self, success: bool, message: str = ""):
        self.clear_busy()
        if success:
            self.state = WorkState.COMPLETED
            self.last_status = message or "COMPLETED"
        else:
            self.to_error(message or "HOME failed")

    def stop(self, message: str = "STOP requested"):
        self.state = WorkState.STOPPED
        self.busy = False
        self.current_worker_operation = ""
        self.last_status = message

    def to_error(self, message: str):
        self.state = WorkState.ERROR
        self.busy = False
        self.current_worker_operation = ""
        self.last_error = message
        self.last_status = "ERROR"

    @staticmethod
    def _validate_detection_response(response: dict):
        left = response.get("left")
        right = response.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError("missing left/right detection result")
        for side, result in (("left", left), ("right", right)):
            class_name = result.get("class_name")
            if class_name not in ALLOWED_TIRE_CLASSES:
                raise ValueError(f"invalid {side} class_name: {class_name}")
            if result.get("stable") is not True:
                raise ValueError(f"{side} detection is not stable")
            if class_name == "none":
                raise ValueError(f"{side} detection is none")

    @staticmethod
    def _determine_targets(left: dict, right: dict) -> set[str]:
        targets = set()
        if left["class_name"] == "bad_tire":
            targets.add("left")
        if right["class_name"] == "bad_tire":
            targets.add("right")
        return targets
