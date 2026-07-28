from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time


class OperationState(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    WAITING_FOR_VEHICLE = "WAITING_FOR_VEHICLE"
    PARKING_STABLE = "PARKING_STABLE"
    SETTING_CAMERA = "SETTING_CAMERA"
    DETECTING_TIRES = "DETECTING_TIRES"
    CHECKING_LEFT_TIRE = "CHECKING_LEFT_TIRE"
    EXECUTING_RIGHT_HELP = "EXECUTING_RIGHT_HELP"
    EXECUTING_LEFT_REMOVE = "EXECUTING_LEFT_REMOVE"
    EMERGENCY_STOPPING = "EMERGENCY_STOPPING"
    EMERGENCY_HOMING = "EMERGENCY_HOMING"
    EMERGENCY = "EMERGENCY"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


ALLOWED_CLASSES = {"bad_tire", "good_tire", "none"}


@dataclass
class OperationController:
    state: OperationState = OperationState.IDLE
    armed: bool = False
    busy: bool = False
    cycle_latched: bool = False
    last_status: str = "IDLE"
    last_error: str = ""
    last_successful_step: str = ""
    left_detection: dict | None = None
    right_detection: dict | None = None
    current_operation: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    emergency_latched: bool = False
    emergency_reason: str = ""

    def arm_system(self):
        if self.emergency_latched or self.state in (
            OperationState.ERROR, OperationState.COMPLETED
        ):
            self.last_status = "RESET required before arm"
            return False
        self.armed = True
        self.state = OperationState.ARMED
        self.last_status = "[AUTO] system armed"
        return True

    def disarm_system(self):
        if self.busy:
            self.last_status = "disarm ignored while busy"
            return False
        self.armed = False
        self.state = OperationState.IDLE
        self.last_status = "[AUTO] system disarmed"
        return True

    def on_vehicle_detected(self):
        if self.armed and self.state == OperationState.ARMED:
            self.state = OperationState.WAITING_FOR_VEHICLE
            self.last_status = "[AUTO] vehicle detected"

    def on_parking_completed(self):
        if not self.armed:
            self.last_status = "parking complete but system is not armed"
            return False
        if self.busy or self.cycle_latched:
            self.last_status = "parking event ignored: cycle already running"
            return False
        if self.state in (
            OperationState.ERROR,
            OperationState.COMPLETED,
            OperationState.STOPPED,
            OperationState.EMERGENCY_STOPPING,
            OperationState.EMERGENCY_HOMING,
            OperationState.EMERGENCY,
        ):
            self.last_status = "parking event ignored in terminal state"
            return False
        self.state = OperationState.PARKING_STABLE
        self.cycle_latched = True
        self.started_at = time.monotonic()
        self.last_status = "[AUTO] parking completed"
        return True

    def start_set_camera(self):
        return self._start_busy(OperationState.PARKING_STABLE, OperationState.SETTING_CAMERA, "SET_CAMERA")

    def apply_set_camera_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return
        self._clear_busy()
        if success:
            self.last_successful_step = "SET_CAMERA"
            self.state = OperationState.DETECTING_TIRES
            self.last_status = message or "[AUTO] SET_CAMERA complete left=OK right=OK"
        else:
            self._error(message or "SET_CAMERA failed")

    def start_detection(self):
        return self._start_busy(OperationState.DETECTING_TIRES, OperationState.DETECTING_TIRES, "DETECT_TIRES")

    def apply_detection_result(self, response: dict | None, error: str | None = None):
        if self.emergency_latched:
            return
        self._clear_busy()
        if error:
            self._error(error)
            return
        try:
            self._validate_detection(response or {})
        except ValueError as exc:
            self._error(str(exc))
            return
        self.left_detection = response["left"]
        self.right_detection = response["right"]
        self.last_successful_step = "DETECT_TIRES"
        self.state = OperationState.CHECKING_LEFT_TIRE
        self.last_status = (
            f"[AUTO] detection left={self.left_detection['class_name']} "
            f"right={self.right_detection['class_name']}"
        )

    def evaluate_left_tire(self):
        if self.state != OperationState.CHECKING_LEFT_TIRE:
            return None
        left = self.left_detection or {}
        class_name = left.get("class_name")
        stable = left.get("stable") is True
        if class_name == "bad_tire" and stable:
            self.state = OperationState.EXECUTING_RIGHT_HELP
            self.busy = True
            self.current_operation = "RIGHT_HELP"
            self.last_status = "[AUTO] sending HELP to right arm"
            return "right_help"
        if class_name == "good_tire" and stable:
            self.state = OperationState.COMPLETED
            self.completed_at = time.monotonic()
            self.last_successful_step = "GOOD_TIRE_NO_ACTION"
            self.last_status = "[AUTO] left tire good; cycle completed"
            return "completed"
        self._error(f"left tire blocks operation: class={class_name} stable={stable}")
        return "error"

    def start_right_help(self):
        if self.state != OperationState.CHECKING_LEFT_TIRE:
            return False
        return self.evaluate_left_tire() == "right_help"

    def apply_right_help_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return
        self._clear_busy()
        if success:
            self.last_successful_step = "RIGHT_HELP"
            self.state = OperationState.EXECUTING_LEFT_REMOVE
            self.busy = True
            self.current_operation = "LEFT_REMOVE"
            self.last_status = message or "[AUTO] right HELP complete"
        else:
            self._error(message or "right HELP failed")

    def start_left_remove(self):
        if self.state != OperationState.EXECUTING_LEFT_REMOVE or self.busy:
            return False
        self.busy = True
        self.current_operation = "LEFT_REMOVE"
        self.last_status = "[AUTO] sending REMOVE_BAD_TIRE to left arm"
        return True

    def apply_left_remove_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return
        self._clear_busy()
        if success:
            self.last_successful_step = "LEFT_REMOVE_BAD_TIRE"
            self.state = OperationState.COMPLETED
            self.completed_at = time.monotonic()
            self.last_status = message or "[AUTO] cycle completed"
        else:
            self._error(message or "left REMOVE_BAD_TIRE failed")

    def trigger_emergency(self, reason: str) -> bool:
        if self.emergency_latched:
            return False
        self.emergency_latched = True
        self.emergency_reason = str(reason)
        self.state = OperationState.EMERGENCY_STOPPING
        self.armed = False
        self.busy = False
        self.current_operation = "EMERGENCY_STOP"
        self.last_error = self.emergency_reason
        self.last_status = "[EMERGENCY] vehicle safety interlock"
        return True

    def mark_emergency_stopping(self):
        if not self.emergency_latched:
            return False
        self.state = OperationState.EMERGENCY_STOPPING
        self.current_operation = "EMERGENCY_STOP"
        return True

    def mark_emergency_homing(self):
        if not self.emergency_latched:
            return False
        self.state = OperationState.EMERGENCY_HOMING
        self.current_operation = "EMERGENCY_HOME"
        self.last_status = "[EMERGENCY] STOP complete; sending HOME"
        return True

    def apply_emergency_home_result(self, success: bool, message: str = ""):
        if not self.emergency_latched:
            return False
        self.state = OperationState.EMERGENCY
        self.busy = False
        self.current_operation = ""
        if success:
            self.last_successful_step = "EMERGENCY_HOME"
            self.last_status = message or "[EMERGENCY] HOME complete; R reset required"
        else:
            detail = message or "emergency HOME failed"
            self.last_error = f"{self.emergency_reason}; {detail}"
            self.last_status = "[EMERGENCY] HOME failed; R reset required"
        return True

    def clear_emergency_on_reset(self):
        self.emergency_latched = False
        self.emergency_reason = ""

    def stop(self, message: str = "[AUTO] STOP requested"):
        if self.emergency_latched:
            self.last_status = "manual STOP ignored: emergency sequence active"
            return False
        self.state = OperationState.STOPPED
        self.armed = False
        self.busy = False
        self.current_operation = ""
        self.last_status = message
        return True

    def reset(self):
        self.state = OperationState.IDLE
        self.armed = False
        self.busy = False
        self.cycle_latched = False
        self.last_status = "RESET"
        self.last_error = ""
        self.last_successful_step = ""
        self.left_detection = None
        self.right_detection = None
        self.current_operation = ""
        self.started_at = None
        self.completed_at = None
        self.clear_emergency_on_reset()

    def _start_busy(self, required_state, next_state, operation):
        if self.busy or self.cycle_latched and required_state == OperationState.PARKING_STABLE and self.state != required_state:
            self.last_status = f"{operation} ignored while busy or latched"
            return False
        if self.state != required_state:
            self.last_status = f"{operation} ignored in {self.state.value}"
            return False
        self.state = next_state
        self.busy = True
        self.current_operation = operation
        self.last_status = f"[AUTO] {operation}"
        return True

    def _clear_busy(self):
        self.busy = False
        self.current_operation = ""

    def _error(self, message: str):
        self.state = OperationState.ERROR
        self.armed = False
        self.busy = False
        self.current_operation = ""
        self.last_error = message
        self.last_status = "ERROR"

    @staticmethod
    def _validate_detection(response: dict):
        left = response.get("left")
        right = response.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError("missing left/right detection result")
        for side, result in (("left", left), ("right", right)):
            class_name = result.get("class_name")
            if class_name not in ALLOWED_CLASSES:
                raise ValueError(f"invalid {side} class_name: {class_name}")
            if result.get("stable") is not True:
                raise ValueError(f"{side} detection is not stable")
            if class_name == "none":
                raise ValueError(f"{side} detection is none")
