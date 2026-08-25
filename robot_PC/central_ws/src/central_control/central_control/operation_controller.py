from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

from central_control.replacement_policy import (
    DetectionPolicy,
    LeftFirstGate,
    transport_slot,
)

class OperationState(str, Enum):
    IDLE = "IDLE"
    BUILDING_REPLACEMENT_PLAN = "BUILDING_REPLACEMENT_PLAN"
    PREPARING_SIDE = "PREPARING_SIDE"
    PREPARE_SIDE = "PREPARING_SIDE"
    SET_ARUCO = "SETTING_ARUCO"
    WAITING_SET_ARUCO_DONE = "WAIT_SET_ARUCO_DONE"
    WAIT_SET_ARUCO_DONE = "WAIT_SET_ARUCO_DONE"
    START_PINKY = "START_PINKY"
    HELPING_OPPOSITE_ARM = "HELPING_OPPOSITE_ARM"
    REMOVING_BAD_TIRE = "REMOVING_BAD_TIRE"
    SETTING_ARUCO = "SETTING_ARUCO"
    WAITING_TRANSPORT = "WAITING_TRANSPORT"
    WAIT_PINKY_ARRIVAL = "WAITING_TRANSPORT"
    PLACING_REMOVED_TIRE = "PLACING_REMOVED_TIRE"
    PICKING_NEW_TIRE = "PICKING_NEW_TIRE"
    INSTALLING_NEW_TIRE = "INSTALLING_NEW_TIRE"
    SIDE_COMPLETED = "SIDE_COMPLETED"
    CHECK_NEXT_SIDE = "CHECK_NEXT_SIDE"
    RETURNING_HOME = "RETURNING_HOME"

    # Legacy states retained for API compatibility; the auto runtime no longer enters them.

    ARMED = "ARMED"
    WAITING_FOR_VEHICLE = "WAITING_FOR_VEHICLE"
    PARKING_STABLE = "PARKING_STABLE"
    SETTING_CAMERA = "SETTING_CAMERA"
    DETECTING_TIRES = "DETECTING_TIRES"
    DETECTION_ERROR = "DETECTION_ERROR"
    CHECKING_LEFT_TIRE = "CHECKING_LEFT_TIRE"
    EXECUTING_RIGHT_HELP = "EXECUTING_RIGHT_HELP"
    EXECUTING_LEFT_REMOVE = "EXECUTING_LEFT_REMOVE"
    REQUESTING_PINKY_MOVE = "REQUESTING_PINKY_MOVE"
    WAITING_PINKY_COMPLETE = "WAITING_PINKY_COMPLETE"
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
    replacement_plan: tuple[str, ...] = ()
    replacement_index: int = 0
    current_side: str | None = None
    completed_sides: set[str] = None
    replacement_gate: LeftFirstGate | None = None
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
    target_arms: set[str] = None

    def __post_init__(self):
        if self.target_arms is None:
            self.target_arms = set()
        if self.completed_sides is None:
            self.completed_sides = set()

    def arm_system(self):
        if self.emergency_latched or self.state in (
            OperationState.ERROR, OperationState.DETECTION_ERROR,
            OperationState.COMPLETED
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
            OperationState.DETECTION_ERROR,
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

    def skip_set_camera(self):
        """Enter tire detection directly; gripper cameras belong to arm vision."""
        if self.state != OperationState.PARKING_STABLE or self.busy:
            return False
        self.state = OperationState.DETECTING_TIRES
        self.last_status = "[AUTO] parking stable; requesting tire detection"
        return True

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
        self.target_arms = {
            side for side, result in (
                ("left", self.left_detection), ("right", self.right_detection)
            ) if result["class_name"] == "bad_tire"
        }
        self.last_successful_step = "DETECT_TIRES"
        self.state = OperationState.BUILDING_REPLACEMENT_PLAN
        self.last_status = (
            f"[AUTO] detection left={self.left_detection['class_name']} "
            f"right={self.right_detection['class_name']}"
        )

    def build_replacement_plan(self):
        if self.state != OperationState.BUILDING_REPLACEMENT_PLAN:
            return False
        response = {
            "left": {"status": "BAD" if self.left_detection["class_name"] == "bad_tire" else "GOOD"},
            "right": {"status": "BAD" if self.right_detection["class_name"] == "bad_tire" else "GOOD"},
        }
        try:
            plan = DetectionPolicy.build_plan(response)
        except ValueError as error:
            self._error(str(error))
            return False
        self.replacement_plan = plan.sides
        self.replacement_index = 0
        self.completed_sides.clear()
        self.replacement_gate = LeftFirstGate(plan)
        if plan.sides:
            self.current_side = plan.sides[0]
            self.state = OperationState.PREPARING_SIDE
            self.last_status = f"[AUTO] replacement plan={list(plan.sides)}"
        else:
            self.current_side = None
            self.state = OperationState.RETURNING_HOME
            self.last_status = "[AUTO] both tires good; returning both arms HOME"
        return True
    @property
    def opposite_side(self):
        if self.current_side not in {"left", "right"}:
            raise RuntimeError("current replacement side is not selected")
        return "right" if self.current_side == "left" else "left"

    def begin_opposite_help(self):
        if self.state != OperationState.HELPING_OPPOSITE_ARM or self.busy:
            return None
        self.busy = True
        self.current_operation = f"{self.opposite_side.upper()}_HELP"
        self.last_status = f"[AUTO] sending HELP to {self.opposite_side} arm"
        return self.opposite_side

    def apply_opposite_help_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        self._clear_busy()
        if not success:
            self._error(message or f"{self.opposite_side} HELP failed")
            return False
        self.last_successful_step = f"{self.opposite_side.upper()}_HELP"
        self.state = OperationState.REMOVING_BAD_TIRE
        self.last_status = message or f"[AUTO] {self.opposite_side} HELP complete"
        return True

    def _begin_current_side_step(self, required_state, operation):
        if self.state != required_state or self.busy or self.current_side is None:
            return None
        self.busy = True
        self.current_operation = f"{self.current_side.upper()}_{operation}"
        self.last_status = f"[AUTO] {operation} side={self.current_side}"
        return self.current_side

    def begin_remove_bad_tire(self):
        return self._begin_current_side_step(OperationState.REMOVING_BAD_TIRE, "REMOVE_BAD_TIRE")

    def apply_remove_bad_tire_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        self._clear_busy()
        if not success:
            self._error(message or f"{self.current_side} REMOVE_BAD_TIRE failed")
            return False
        self.last_successful_step = f"{self.current_side.upper()}_REMOVE_BAD_TIRE"
        self.state = OperationState.WAITING_TRANSPORT
        self.last_status = message or f"[AUTO] {self.current_side} REMOVE_BAD_TIRE complete; waiting Pinky"
        return True

    def prepare_current_side(self):
        if self.state != OperationState.PREPARING_SIDE or self.busy:
            return False
        try:
            self.replacement_gate.start(self.current_side)
        except (ValueError, RuntimeError) as error:
            self._error(str(error))
            return False
        self.state = OperationState.SETTING_ARUCO
        self.last_status = f"[AUTO] preparing SET_ARUCO side={self.current_side}"
        return True

    def begin_set_aruco(self):
        side = self._begin_current_side_step(OperationState.SETTING_ARUCO, "SET_ARUCO")
        if side is not None:
            self.state = OperationState.WAITING_SET_ARUCO_DONE
        return side

    def apply_set_aruco_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        if self.state != OperationState.WAIT_SET_ARUCO_DONE or not self.busy:
            return False
        self._clear_busy()
        if not success:
            self._error(message or f"{self.current_side} SET_ARUCO failed")
            return False
        self.last_successful_step = f"{self.current_side.upper()}_SET_ARUCO"
        self.state = OperationState.START_PINKY
        self.last_status = f"[AUTO] SET_ARUCO complete side={self.current_side}; starting Pinky"
        return True

    def mark_pinky_started(self):
        if self.state != OperationState.START_PINKY or self.busy:
            return False
        self.state = OperationState.HELPING_OPPOSITE_ARM
        self.last_status = f"[AUTO] Pinky started; sending HELP side={self.opposite_side}"
        return True

    def mark_transport_ready(self):
        if self.state != OperationState.WAITING_TRANSPORT or self.busy:
            return False
        self.state = OperationState.PLACING_REMOVED_TIRE
        self.last_status = f"[AUTO] Pinky arrived; placing removed tire side={self.current_side}"
        return True

    def current_transport_slot(self, tire_kind: str):
        if self.current_side is None:
            raise RuntimeError("current replacement side is not selected")
        return transport_slot(self.current_side, tire_kind)

    def begin_place_removed_tire(self):
        return self._begin_current_side_step(OperationState.PLACING_REMOVED_TIRE, "PLACE_REMOVED_TIRE")

    def apply_place_removed_tire_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        self._clear_busy()
        if not success:
            self._error(message or f"{self.current_side} PLACE_REMOVED_TIRE failed")
            return False
        self.last_successful_step = f"{self.current_side.upper()}_PLACE_REMOVED_TIRE"
        self.state = OperationState.PICKING_NEW_TIRE
        self.last_status = message or f"[AUTO] {self.current_side} removed tire placed"
        return True

    def begin_pick_new_tire(self):
        return self._begin_current_side_step(OperationState.PICKING_NEW_TIRE, "PICK_NEW_TIRE")

    def apply_pick_new_tire_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        self._clear_busy()
        if not success:
            self._error(message or f"{self.current_side} PICK_NEW_TIRE failed")
            return False
        self.last_successful_step = f"{self.current_side.upper()}_PICK_NEW_TIRE"
        self.state = OperationState.INSTALLING_NEW_TIRE
        self.last_status = message or f"[AUTO] {self.current_side} new tire picked"
        return True

    def begin_install_new_tire(self):
        return self._begin_current_side_step(OperationState.INSTALLING_NEW_TIRE, "INSTALL_NEW_TIRE")

    def apply_install_new_tire_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        self._clear_busy()
        if not success:
            self._error(message or f"{self.current_side} INSTALL_NEW_TIRE failed")
            return False
        self.last_successful_step = f"{self.current_side.upper()}_INSTALL_NEW_TIRE"
        self.state = OperationState.SIDE_COMPLETED
        self.last_status = message or f"[AUTO] {self.current_side} replacement complete"
        return True

    def complete_current_side(self):
        if self.state != OperationState.SIDE_COMPLETED or self.current_side is None:
            return False
        try:
            self.replacement_gate.complete(self.current_side)
        except RuntimeError as error:
            self._error(str(error))
            return False
        self.completed_sides.add(self.current_side)
        self.state = OperationState.CHECK_NEXT_SIDE
        self.last_status = f"[AUTO] side completed: {self.current_side}"
        return True

    def check_next_side(self):
        if self.state != OperationState.CHECK_NEXT_SIDE:
            return False
        self.replacement_index += 1
        if self.replacement_index < len(self.replacement_plan):
            self.current_side = self.replacement_plan[self.replacement_index]
            self.state = OperationState.PREPARING_SIDE
            self.last_status = f"[AUTO] preparing next side={self.current_side}"
        else:
            self.current_side = None
            self.state = OperationState.RETURNING_HOME
            self.last_status = "[AUTO] all replacements complete; returning HOME"
        return True

    def begin_return_home(self):
        if self.state != OperationState.RETURNING_HOME or self.busy:
            return False
        self.busy = True
        self.current_operation = "HOME_BOTH"
        self.last_status = "[AUTO] sending HOME to both arms"
        return True

    def apply_return_home_result(self, success: bool, message: str = ""):
        if self.emergency_latched:
            return False
        self._clear_busy()
        if not success:
            self._error(message or "one or both HOME commands failed")
            return False
        self.state = OperationState.REQUESTING_PINKY_MOVE
        self.last_successful_step = "HOME_BOTH"
        self.last_status = message or "[AUTO] both HOME complete; requesting Pinky 3->4->5"
        return True

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
            self.state = OperationState.REQUESTING_PINKY_MOVE
            self.last_status = (
                "[AUTO] left REMOVE_BAD_TIRE complete; "
                "waiting for Pinky connection"
            )
            if message:
                self.last_status += f" | {message}"
        else:
            self._error(message or "left REMOVE_BAD_TIRE failed")

    def mark_pinky_move_requested(self):
        if self.state != OperationState.REQUESTING_PINKY_MOVE:
            return False
        self.state = OperationState.WAITING_PINKY_COMPLETE
        self.current_operation = "PINKY_MOVE"
        self.last_status = "[AUTO] Pinky route requested"
        return True

    def apply_pinky_status(self, status: str):
        status = str(status).strip()
        if status == "RUNNING":
            if self.state == OperationState.WAITING_PINKY_COMPLETE:
                self.last_status = "[AUTO] Pinky route running"
                return True
            return False
        if status == "COMPLETED":
            if self.state != OperationState.WAITING_PINKY_COMPLETE:
                return False
            self.current_operation = ""
            self.last_successful_step = "PINKY_AFTER_TIRE_SERVICE"
            self.state = OperationState.COMPLETED
            self.completed_at = time.monotonic()
            self.last_status = "[AUTO] Pinky route completed; cycle completed"
            return True
        if status.startswith("ERROR:"):
            if self.state not in (
                OperationState.REQUESTING_PINKY_MOVE,
                OperationState.WAITING_PINKY_COMPLETE,
            ):
                return False
            detail = status.partition(":")[2].strip() or "unknown error"
            self._error(f"Pinky route error: {detail}")
            return True
        if status == "STOPPED":
            if self.state not in (
                OperationState.REQUESTING_PINKY_MOVE,
                OperationState.WAITING_PINKY_COMPLETE,
            ):
                return False
            self._error("Pinky route stopped before completion")
            return True
        if status == "IDLE":
            return self.state in (
                OperationState.REQUESTING_PINKY_MOVE,
                OperationState.WAITING_PINKY_COMPLETE,
            )
        return False

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

    def prepare_detection_retry(self, message="detection requires retry"):
        if self.emergency_latched:
            return False
        self._clear_busy()
        self.left_detection = None
        self.right_detection = None
        self.target_arms.clear()
        self.state = OperationState.DETECTING_TIRES
        self.last_status = message
        return True

    def to_detection_error(self, message):
        self._clear_busy()
        self.left_detection = None
        self.right_detection = None
        self.target_arms.clear()
        self.armed = False
        self.state = OperationState.DETECTION_ERROR
        self.last_error = str(message)
        self.last_status = "DETECTION_ERROR"

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
        self.target_arms.clear()
        self.current_operation = ""
        self.started_at = None
        self.completed_at = None
        self.replacement_plan = ()
        self.replacement_index = 0
        self.current_side = None
        self.completed_sides.clear()
        self.replacement_gate = None
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
            expected_status = "GOOD" if class_name == "good_tire" else "BAD"
            if result.get("status", expected_status) != expected_status:
                raise ValueError(f"{side} detection status is not final")
