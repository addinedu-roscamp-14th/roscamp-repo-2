from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import math
import queue
import threading
import time

import cv2

from central_control.arm_client import ArmCommandResult
from central_control.dual_arm_manager import DualArmManager
from central_control.manual_test_input import (
    ManualTestKeyboard,
    ManualTestVisionClient,
)
from central_control.manual_central_node import (
    CentralParkingStateMachine,
    apply_overrides,
    load_yaml,
    open_top_camera,
    resolve_source_config,
    str_to_bool,
)
from central_control.operation_controller import OperationController, OperationState
from central_control.replacement_policy import TRANSPORT_SLOTS
from central_control.stop_detector import StopDetector
from central_control.top_view import TopViewTransformer
from central_control.vehicle_detector import create_vehicle_detector
from central_control.vision_client import VisionClient
from central_control.transport_operations import (
    MockTransportOperationHandler,
    RosTransportOperationHandler,
    UnavailableTransportOperationHandler,
)


SAFETY_MONITORED_STATES = {
    OperationState.SETTING_CAMERA,
    OperationState.DETECTING_TIRES,
    OperationState.BUILDING_REPLACEMENT_PLAN,
    OperationState.PREPARING_SIDE,
    OperationState.WAITING_SET_ARUCO_DONE,
    OperationState.START_PINKY,
    OperationState.HELPING_OPPOSITE_ARM,
    OperationState.REMOVING_BAD_TIRE,
    OperationState.SETTING_ARUCO,
    OperationState.WAITING_TRANSPORT,
    OperationState.PLACING_REMOVED_TIRE,
    OperationState.PICKING_NEW_TIRE,
    OperationState.INSTALLING_NEW_TIRE,
    OperationState.CHECKING_LEFT_TIRE,
    OperationState.EXECUTING_RIGHT_HELP,
    OperationState.EXECUTING_LEFT_REMOVE,
    OperationState.RETURNING_HOME,
}


def apply_manual_test_mode_config(
    parking_config, yolo_config, auto_config, test_mode, hardware_enabled
):
    if not test_mode:
        return
    parking_config["mock_camera"] = True
    yolo_config["mock_vehicle_detector"] = True
    yolo_config["show_window"] = False
    operation = auto_config.setdefault("operation", {})
    mock_hardware = not bool(hardware_enabled)
    operation["mock_arms"] = mock_hardware
    operation["mock_transport_operations"] = False
    operation["mock_aruco_transport_operations"] = mock_hardware
    operation["mock_install_new_tire"] = mock_hardware


class AutoOperationRuntime:
    def __init__(
        self,
        parking_config: dict,
        yolo_config: dict,
        network_config: dict,
        auto_config: dict,
        arm_manager=None,
        vision_client=None,
        transport_operations=None,
        pinky_command_sender=None,
        pinky_subscription_count=None,
        test_mode=False,
        hardware_enabled=False,
    ):
        self.parking_config = parking_config
        self.yolo_config = yolo_config
        self.network_config = network_config
        self.auto_config = auto_config
        self.test_mode = bool(test_mode)
        self.hardware_enabled = bool(hardware_enabled)
        self.test_key_queue = queue.SimpleQueue()
        self.manual_detection_submitted = False
        self.test_transport_ready_side = None
        self.operation = auto_config.get("operation", {})
        self.commands = auto_config.get("commands", {})
        self.delays = auto_config.get("delays", {})
        self.safety = auto_config.get("safety", {})
        self.timeouts = auto_config.get("timeouts", {})
        self.parking = CentralParkingStateMachine(parking_config)
        self.controller = OperationController()
        self.detector = create_vehicle_detector(yolo_config)
        self.camera = open_top_camera(parking_config)
        self.top_view = TopViewTransformer(parking_config)
        self.show_window = bool(yolo_config.get("show_window", True))
        if transport_operations is not None:
            self.transport_operations = transport_operations
        elif bool(self.operation.get("mock_aruco_transport_operations", False)):
            self.transport_operations = MockTransportOperationHandler()
        else:
            self.transport_operations = UnavailableTransportOperationHandler()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.emergency_executor = ThreadPoolExecutor(max_workers=1)
        self.worker_future: Future | None = None
        self.worker_operation = ""
        self.worker_generation = 0
        self.active_worker_generation: int | None = None
        self.detection_attempt_count = 0
        self.detection_retry_ready_at = 0.0
        self.emergency_future: Future | None = None
        self.emergency_operation = ""
        self.emergency_stop_results = {}
        self.emergency_home_results = {}
        self.pinky_command_sender = pinky_command_sender or (lambda command: None)
        self.pinky_subscription_count = pinky_subscription_count or (lambda: 0)
        self.pinky_start_sent = False
        self.pinky_move_started_at = None
        self.pinky_last_status = "IDLE"
        self.pinky_side_route = None
        self.pinky_side = None
        self.pinky_side_seen_running = False
        self.pinky_side_arrived = False
        self.safety_reference_center = None
        self.safety_current_center = None
        self.safety_displacement_px = None
        self.safety_jitter_px = None
        self.safety_speed_px_sec = None
        self.motion_violation_started_at = None
        self.motion_violation_elapsed = 0.0
        self.vehicle_lost_started_at = None
        self.vehicle_lost_elapsed = 0.0
        self.safety_tracker = self._new_safety_tracker()
        self.quit_requested = False
        self.connection_status = {
            "vision": "mock" if self.operation.get("mock_vision", False) else "configured",
            "left_arm": "mock" if self.operation.get("mock_arms", True) else "configured",
            "right_arm": "mock" if self.operation.get("mock_arms", True) else "configured",
        }
        self.arm_manager = arm_manager or DualArmManager.from_config(
            network_config, {"operation": {"mock_arms": self.operation.get("mock_arms", True)}}
        )
        self.vision_client = vision_client or VisionClient(
            mock_mode=bool(self.operation.get("mock_vision", False)),
            **network_config["vision_pc"],
        )
        if bool(self.operation.get("auto_start", False)):
            self.controller.arm_system()
            self.controller.on_vehicle_detected()

    def step(self):
        self._apply_emergency_result_if_ready()
        self._apply_worker_result_if_ready()
        if self.test_mode:
            self._drain_test_keys()
            self._advance_auto_workers()
            self._advance_pinky_move(time.monotonic())
            return None

        now = time.monotonic()
        ok, frame = self.camera.read()
        if ok:
            frame = self.top_view.transform(frame)
            detections, _ = self.detector.detect(frame)
            result = self.parking.process_detections(detections, now=now)
            if result.vehicle_detected:
                self.controller.on_vehicle_detected()
            if result.parked_event:
                self._on_parking_completed(result=result, now=now)
            self._update_vehicle_safety(result, now=now)
            if self.show_window:
                cv2.imshow("AUTO OPERATION", self._draw(frame, result))
                key = cv2.waitKey(1) & 0xFF
                self.handle_key(key)
        else:
            result = self.parking.process_detections([], now=now)
            self._update_vehicle_safety(result, now=now)
        self._advance_auto_workers()
        self._advance_pinky_move(now)
        return result

    def handle_key(self, key):
        if key in (-1, 255):
            return None
        if key in (ord("s"), ord("S")):
            self.start_stop()
            return "STOP"
        if key in (ord("q"), ord("Q"), 27):
            self.quit_requested = True
            return "QUIT"
        if key in (ord("r"), ord("R")):
            if self.controller.state in (
            OperationState.ERROR,
            OperationState.DETECTION_ERROR,
                OperationState.COMPLETED,
                OperationState.STOPPED,
                OperationState.EMERGENCY,
            ):
                self.reset_cycle()
            else:
                self.controller.last_status = "RESET ignored outside terminal state"
            return "RESET"
        if key in (ord("h"), ord("H")):
            self.start_home()
            return "HOME"
        if key in (ord("n"), ord("N")):
            if self.controller.emergency_latched:
                self.controller.last_status = "BYPASS ignored: R reset required"
                return "BYPASS"
            self.parking.bypass_parking()
            self._on_parking_completed(force=True)
            return "BYPASS"
        if key in (ord("a"), ord("A")):
            if self.controller.armed:
                self.controller.disarm_system()
            else:
                self.controller.arm_system()
                self.controller.on_vehicle_detected()
                if self._parking_currently_stable():
                    self._on_parking_completed(force=True)
            return "ARM_TOGGLE"
        return "IGNORED"

    def enqueue_test_key(self, key):
        if self.test_mode and key:
            self.test_key_queue.put(str(key)[0].lower())

    def _drain_test_keys(self):
        try:
            key = self.test_key_queue.get_nowait()
        except queue.Empty:
            return
        self.handle_test_key(key)

    def handle_test_key(self, key):
        if not self.test_mode:
            return "IGNORED"
        key = str(key).lower()[:1]
        if key == "p":
            print("[TEST] p -> vehicle parked/stable")
            if self.controller.state == OperationState.IDLE:
                self.handle_key(ord("a"))
            before = self.controller.cycle_latched
            self.handle_key(ord("n"))
            accepted = self.controller.cycle_latched and not before
            if accepted:
                self.manual_detection_submitted = False
                self.test_transport_ready_side = None
                clear = getattr(self.vision_client, "clear", None)
                if clear is not None:
                    clear()
            else:
                print(f"[TEST] p ignored state={self.controller.state.value}")
            return "PARKED" if accepted else "IGNORED"

        detection_keys = {
            "1": ("GOOD", "GOOD"),
            "2": ("BAD", "GOOD"),
            "3": ("GOOD", "BAD"),
            "4": ("BAD", "BAD"),
        }
        if key in detection_keys:
            left_status, right_status = detection_keys[key]
            print(
                f"[TEST] {key} -> LEFT={left_status} RIGHT={right_status}"
            )
            if (
                self.controller.state != OperationState.DETECTING_TIRES
                or self.manual_detection_submitted
            ):
                print(
                    f"[TEST] tire result ignored state={self.controller.state.value} "
                    f"submitted={self.manual_detection_submitted}"
                )
                return "IGNORED"
            submit = getattr(self.vision_client, "submit_detection", None)
            if submit is None or not submit(left_status, right_status):
                print("[TEST] tire result rejected by manual vision adapter")
                return "IGNORED"
            self.manual_detection_submitted = True
            return "TIRE_RESULT"

        if key == "t":
            side = self.controller.current_side
            accepted = self.handle_transport_ready()
            if accepted:
                self.test_transport_ready_side = side
                print(f"[TEST] t -> TRANSPORT_READY side={side}")
                return "TRANSPORT_READY"
            print(
                f"[TEST] t ignored state={self.controller.state.value}"
            )
            return "IGNORED"
        if key == "h":
            print(self.format_test_status(), flush=True)
            return "STATUS"
        if key == "s":
            print("[TEST] s -> normal STOP")
            self.handle_key(ord("s"))
            return "STOP"
        if key == "e":
            print("[TEST] e -> EMERGENCY STOP")
            self._trigger_vehicle_emergency("manual test emergency")
            return "EMERGENCY"
        if key == "r":
            print("[TEST] r -> reset")
            self.handle_key(ord("r"))
            return "RESET"
        if key == "q":
            print("[TEST] q -> keyboard input stopped; ROS node continues")
            return "QUIT_KEYBOARD"
        print(f"[TEST] ignored key={key!r}")
        return "IGNORED"

    def test_status(self):
        side = self.controller.current_side
        return {
            "STATE": self.controller.state.value,
            "CURRENT_SIDE": side,
            "REPLACEMENT_PLAN": list(self.controller.replacement_plan),
            "REPLACEMENT_INDEX": self.controller.replacement_index,
            "LEFT_TIRE": self.controller.left_detection,
            "RIGHT_TIRE": self.controller.right_detection,
            "TRANSPORT_READY": self.test_transport_ready_side == side and side is not None,
            "BUSY": self.controller.busy,
            "IN_FLIGHT": self.worker_operation or None,
            "EMERGENCY": self.controller.emergency_latched,
            "TEST_MODE": self.test_mode,
            "HARDWARE_ENABLED": self.hardware_enabled,
            "TRANSPORT_SLOTS": TRANSPORT_SLOTS,
        }

    def format_test_status(self):
        status = self.test_status()
        return "\n".join(
            f"{key}={value}" for key, value in status.items()
        )

    def reset_cycle(self):
        self._invalidate_worker()
        self.parking.reset()
        self.controller.reset()
        self._reset_safety_tracking(clear_reference=True)
        self.emergency_stop_results = {}
        self.emergency_home_results = {}
        self._reset_pinky_tracking()
        self.detection_attempt_count = 0
        self.detection_retry_ready_at = 0.0
        self.manual_detection_submitted = False
        self.test_transport_ready_side = None
        clear = getattr(self.vision_client, "clear", None)
        if clear is not None:
            clear()

    def _new_safety_tracker(self):
        return StopDetector(
            median_samples=self.parking_config.get("parking_median_samples", 5),
            motion_window_sec=self.parking_config.get(
                "parking_motion_window_sec", 1.0
            ),
            stop_jitter_threshold=self.parking_config.get(
                "parking_stop_jitter_px", 6.0
            ),
            move_jitter_threshold=self.safety.get("emergency_jitter_px", 12.0),
            stop_speed_threshold=self.parking_config.get(
                "parking_stop_speed_px_sec", 2.0
            ),
            move_speed_threshold=self.safety.get(
                "emergency_speed_px_sec", 6.0
            ),
            motion_grace_sec=self.safety.get("emergency_hold_sec", 0.35),
        )

    def _safety_monitor_active(self):
        return (
            bool(self.safety.get("vehicle_motion_monitor_enabled", True))
            and self.controller.state in SAFETY_MONITORED_STATES
            and not self.controller.emergency_latched
        )

    def _reset_safety_tracking(self, clear_reference=False):
        self.safety_tracker = self._new_safety_tracker()
        self.safety_current_center = None
        self.safety_displacement_px = None
        self.safety_jitter_px = None
        self.safety_speed_px_sec = None
        self.motion_violation_started_at = None
        self.motion_violation_elapsed = 0.0
        self.vehicle_lost_started_at = None
        self.vehicle_lost_elapsed = 0.0
        if clear_reference:
            self.safety_reference_center = None

    def _capture_safety_reference(self, result=None, now=None):
        now = time.monotonic() if now is None else float(now)
        reference = self.parking.parking_monitor.filtered_center
        if reference is None and result is not None:
            reference = result.center
        self._reset_safety_tracking(clear_reference=True)
        if reference is None:
            return
        self.safety_reference_center = (
            float(reference[0]), float(reference[1])
        )
        self.safety_tracker.update(self.safety_reference_center, now)
        self.safety_current_center = self.safety_reference_center
        self.safety_displacement_px = 0.0

    def _update_vehicle_safety(self, result, now=None):
        now = time.monotonic() if now is None else float(now)
        if not self._safety_monitor_active():
            self.motion_violation_started_at = None
            self.motion_violation_elapsed = 0.0
            self.vehicle_lost_started_at = None
            self.vehicle_lost_elapsed = 0.0
            return False

        detected = bool(result.vehicle_detected and result.center is not None)
        if not detected:
            return self._evaluate_safety_metrics(
                current_center=None,
                jitter=None,
                speed=None,
                vehicle_detected=False,
                now=now,
            )

        self.safety_tracker.update(result.center, now)
        self.safety_current_center = self.safety_tracker.filtered_center
        self.safety_jitter_px = self.safety_tracker.jitter_radius
        self.safety_speed_px_sec = self.safety_tracker.speed
        return self._evaluate_safety_metrics(
            current_center=self.safety_current_center,
            jitter=self.safety_jitter_px,
            speed=self.safety_speed_px_sec,
            vehicle_detected=True,
            now=now,
        )

    def _evaluate_safety_metrics(
        self,
        current_center,
        jitter,
        speed,
        vehicle_detected,
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        if not self._safety_monitor_active():
            return False

        if not vehicle_detected:
            self.motion_violation_started_at = None
            self.motion_violation_elapsed = 0.0
            if self.vehicle_lost_started_at is None:
                self.vehicle_lost_started_at = now
            self.vehicle_lost_elapsed = max(
                0.0, now - self.vehicle_lost_started_at
            )
            lost_limit = float(
                self.safety.get("detection_lost_emergency_sec", 1.0)
            )
            if self.vehicle_lost_elapsed >= lost_limit:
                return self._trigger_vehicle_emergency(
                    f"vehicle detection lost for "
                    f"{self.vehicle_lost_elapsed:.2f}s"
                )
            return False

        self.vehicle_lost_started_at = None
        self.vehicle_lost_elapsed = 0.0
        self.safety_current_center = current_center
        self.safety_jitter_px = jitter
        self.safety_speed_px_sec = speed
        if current_center is not None and self.safety_reference_center is not None:
            self.safety_displacement_px = math.hypot(
                float(current_center[0]) - self.safety_reference_center[0],
                float(current_center[1]) - self.safety_reference_center[1],
            )
        else:
            self.safety_displacement_px = None

        reasons = []
        jitter_limit = float(self.safety.get("emergency_jitter_px", 12.0))
        speed_limit = float(
            self.safety.get("emergency_speed_px_sec", 6.0)
        )
        displacement_limit = float(
            self.safety.get("emergency_reference_displacement_px", 12.0)
        )
        if jitter is not None and float(jitter) >= jitter_limit:
            reasons.append(f"jitter={float(jitter):.2f}px")
        if speed is not None and float(speed) >= speed_limit:
            reasons.append(f"speed={float(speed):.2f}px/s")
        if (
            self.safety_displacement_px is not None
            and self.safety_displacement_px >= displacement_limit
        ):
            reasons.append(
                f"displacement={self.safety_displacement_px:.2f}px"
            )

        if not reasons:
            self.motion_violation_started_at = None
            self.motion_violation_elapsed = 0.0
            return False
        if self.motion_violation_started_at is None:
            self.motion_violation_started_at = now
        self.motion_violation_elapsed = max(
            0.0, now - self.motion_violation_started_at
        )
        hold = float(self.safety.get("emergency_hold_sec", 0.35))
        if self.motion_violation_elapsed < hold:
            return False
        return self._trigger_vehicle_emergency(
            "vehicle motion: " + ", ".join(reasons)
        )

    def _trigger_vehicle_emergency(self, reason):
        if not self.controller.trigger_emergency(reason):
            return False
        self._invalidate_worker()
        if self.worker_future is not None:
            self.worker_future.cancel()
        print(f"[EMERGENCY] {reason}")
        if bool(self.safety.get("send_stop_before_home", True)):
            self.controller.mark_emergency_stopping()
            self.emergency_operation = "STOP"
            self._log_mock_arm_stop()
            self.emergency_future = self.emergency_executor.submit(
                self._send_both_stop,
            )
        elif bool(self.safety.get("home_on_vehicle_motion", True)):
            self._start_emergency_home()
        else:
            self.controller.apply_emergency_home_result(
                True, "[EMERGENCY] latched; STOP/HOME transmission disabled"
            )
        self._publish_pinky_command("STOP")
        return True

    def _start_emergency_home(self):
        self.controller.mark_emergency_homing()
        self.emergency_operation = "HOME"
        self.emergency_future = self.emergency_executor.submit(
            self._worker_emergency_home
        )

    def _worker_emergency_home(self):
        time.sleep(float(self.safety.get("stop_to_home_delay_sec", 0.5)))
        return self.arm_manager.send_both(self.commands.get("home", "HOME"))

    def _apply_emergency_result_if_ready(self):
        if self.emergency_future is None or not self.emergency_future.done():
            return
        operation = self.emergency_operation
        command = self.commands.get(
            "stop" if operation == "STOP" else "home", operation
        )
        try:
            results = self.emergency_future.result()
        except Exception as error:
            results = self._failed_both_results(command, error)
        self.emergency_future = None
        self.emergency_operation = ""
        self._update_arm_status(results)

        if operation == "STOP":
            self.emergency_stop_results = results
            if bool(self.safety.get("home_on_vehicle_motion", True)):
                self._start_emergency_home()
            else:
                self.controller.apply_emergency_home_result(
                    True, "[EMERGENCY] STOP complete; automatic HOME disabled"
                )
            return

        self.emergency_home_results = results
        success = self.arm_manager.all_success(results)
        self.controller.apply_emergency_home_result(
            success, self._result_message(results)
        )

    @staticmethod
    def _failed_both_results(command, error):
        return {
            side: ArmCommandResult(
                arm=side,
                command=command,
                success=False,
                error=f"emergency command raised: {error}",
            )
            for side in ("left", "right")
        }

    def _parking_currently_stable(self):
        monitor = self.parking.parking_monitor
        return self.parking.parked_latched and monitor.is_stationary and monitor.fully_inside

    def _on_parking_completed(self, result=None, now=None, force=False):
        del force
        if self.controller.armed:
            if self.controller.on_parking_completed():
                self._capture_safety_reference(result, now=now)
                print("[AUTO] parking completed")
        else:
            self.controller.last_status = "parking complete but system is not armed"

    def _advance_auto_workers(self):
        if self.controller.emergency_latched:
            return
        if self.worker_future is not None or self.controller.busy:
            return
        if self.controller.state == OperationState.PARKING_STABLE:
            self.controller.skip_set_camera()
        elif self.controller.state == OperationState.DETECTING_TIRES:
            if time.monotonic() < self.detection_retry_ready_at:
                return
            if self.controller.start_detection():
                self.detection_attempt_count += 1
                print("[AUTO] requesting tire detection")
                self._submit("DETECT_TIRES", self._worker_detect_tires)
        elif self.controller.state == OperationState.BUILDING_REPLACEMENT_PLAN:
            self.controller.build_replacement_plan()
        elif self.controller.state == OperationState.PREPARING_SIDE:
            self.controller.prepare_current_side()
        elif self.controller.state == OperationState.SETTING_ARUCO:
            side = self.controller.begin_set_aruco()
            if side:
                print(f"[{side.upper()}] SET_ARUCO requested", flush=True)
                self._submit(
                    "SET_ARUCO",
                    lambda side=side: self._worker_arm_command(
                        side, "set_aruco", "SET_ARUCO"
                    ),
                )
        elif self.controller.state == OperationState.START_PINKY:
            side = self.controller.current_side
            if side not in ("left", "right"):
                self.controller._error("cannot start Pinky without a replacement side")
                return
            if bool(self.operation.get("mock_transport_operations", False)):
                self.controller.mark_pinky_started()
            elif self.pinky_side != side:
                route = "tire_stop_1" if side == "left" else "tire_stop_2"
                if self._publish_pinky_command(f"START:{route}"):
                    self.pinky_side = side
                    self.pinky_side_route = route
                    self.pinky_side_seen_running = False
                    self.pinky_side_arrived = False
                    print(
                        f"[{side.upper()}] Starting Pinky route {route}",
                        flush=True,
                    )
                    self.controller.mark_pinky_started()
                else:
                    self.controller._error(f"failed to request Pinky {route}")
                    return
            else:
                self.controller.mark_pinky_started()
        elif self.controller.state == OperationState.HELPING_OPPOSITE_ARM:
            opposite = self.controller.begin_opposite_help()
            if opposite:
                self._submit(
                    "HELP_OPPOSITE",
                    lambda side=opposite: self._worker_arm_command(side, "help", "HELP"),
                )
        elif self.controller.state == OperationState.REMOVING_BAD_TIRE:
            side = self.controller.begin_remove_bad_tire()
            if side:
                self._submit("REMOVE_BAD_TIRE", lambda side=side: self._worker_arm_command(side, "remove_bad_tire", "REMOVE_BAD_TIRE"))
        elif self.controller.state == OperationState.WAITING_TRANSPORT:
            if self.pinky_side_arrived and self.pinky_side == self.controller.current_side:
                if self.handle_transport_ready():
                    print(f"[AUTO] Pinky arrived at {self.pinky_side_route}")
            elif bool(self.operation.get("mock_transport_operations", False)):
                print("[AUTO] Pinky transport bypassed", flush=True)
                if self.handle_transport_ready():
                    print(
                        "[AUTO] transport ready "
                        "(mock_transport_operations=true)",
                        flush=True,
                    )
        elif self.controller.state == OperationState.PLACING_REMOVED_TIRE:
            side = self.controller.begin_place_removed_tire()
            if side:
                slot = self.controller.current_transport_slot("removed_tire")
                self._submit(
                    "PLACE_REMOVED_TIRE",
                    lambda side=side, slot=slot: self._worker_transport("PLACE_REMOVED_TIRE", side, slot),
                )
        elif self.controller.state == OperationState.PICKING_NEW_TIRE:
            side = self.controller.begin_pick_new_tire()
            if side:
                slot = self.controller.current_transport_slot("new_tire")
                self._submit(
                    "PICK_NEW_TIRE",
                    lambda side=side, slot=slot: self._worker_transport("PICK_NEW_TIRE", side, slot),
                )
        elif self.controller.state == OperationState.INSTALLING_NEW_TIRE:
            side = self.controller.begin_install_new_tire()
            if side:
                self._submit(
                    "INSTALL_NEW_TIRE",
                    lambda side=side: self._worker_install_new_tire(side),
                )
        elif self.controller.state == OperationState.SIDE_COMPLETED:
            self.controller.complete_current_side()
        elif self.controller.state == OperationState.CHECK_NEXT_SIDE:
            previous_side = self.controller.current_side
            self.controller.check_next_side()
            if self.controller.current_side != previous_side:
                self.pinky_side = None
                self.pinky_side_route = None
                self.pinky_side_seen_running = False
                self.pinky_side_arrived = False
        elif self.controller.state == OperationState.RETURNING_HOME:
            if self.controller.begin_return_home():
                self._submit("HOME_BOTH", self._worker_home_both)

    def handle_transport_ready(self):
        if self.controller.emergency_latched:
            return False
        return self.controller.mark_transport_ready()


    def _advance_pinky_move(self, now=None):
        if self.controller.state == OperationState.REQUESTING_PINKY_MOVE:
            if self.pinky_start_sent or self.pinky_subscription_count() < 1:
                return
            if not self.controller.mark_pinky_move_requested():
                return
            self.pinky_start_sent = True
            self.pinky_move_started_at = time.monotonic() if now is None else float(now)
            if not self._publish_pinky_command("START:after_tire_service"):
                self.controller._error("failed to publish Pinky route command")
                return
            print("[AUTO] Pinky START:after_tire_service published")
            return
        if (
            self.controller.state == OperationState.WAITING_PINKY_COMPLETE
            and self.pinky_move_started_at is not None
        ):
            now = time.monotonic() if now is None else float(now)
            timeout = float(self.timeouts.get("pinky_move_sec", 180.0))
            if now - self.pinky_move_started_at >= timeout:
                self.controller._error(f"Pinky move timeout after {timeout:.1f}s")

    def handle_pinky_route_status(self, status):
        status = str(status).strip()
        self.pinky_last_status = status

        # During tire service, Pinky station arrival is latched independently
        # from the arm state so arriving before REMOVE_BAD_TIRE finishes is safe.
        if self.pinky_side_route is not None and self.controller.current_side == self.pinky_side:
            if status == "RUNNING":
                self.pinky_side_seen_running = True
            elif status == "COMPLETED" and self.pinky_side_seen_running:
                self.pinky_side_arrived = True
            elif status.startswith("ERROR:") or status == "STOPPED":
                self.controller._error(f"Pinky {self.pinky_side_route} failed: {status}")
                return True

        accepted = self.controller.apply_pinky_status(status)
        if accepted and (status == "COMPLETED" or status.startswith("ERROR:")):
            self.pinky_move_started_at = None
        return accepted or self.pinky_side_arrived

    def _publish_pinky_command(self, command):
        try:
            self.pinky_command_sender(command)
            return True
        except Exception as error:
            self.controller.last_error = f"Pinky publish failed: {error}"
            return False

    def _reset_pinky_tracking(self):
        self.pinky_start_sent = False
        self.pinky_move_started_at = None
        self.pinky_last_status = "IDLE"
        self.pinky_side_route = None
        self.pinky_side = None
        self.pinky_side_seen_running = False
        self.pinky_side_arrived = False

    def _invalidate_worker(self):
        self.worker_generation += 1

    def _submit(self, operation, func):
        if self.worker_future is not None:
            self.controller.last_status = f"{operation} ignored: worker running"
            return
        self.worker_operation = operation
        self.active_worker_generation = self.worker_generation
        self.worker_future = self.executor.submit(func)

    def _apply_worker_result_if_ready(self):
        if self.worker_future is None or not self.worker_future.done():
            return
        operation = self.worker_operation
        stale = self.active_worker_generation != self.worker_generation
        if self.controller.emergency_latched or stale:
            try:
                self.worker_future.result()
            except Exception:
                pass
            self.worker_future = None
            self.worker_operation = ""
            self.active_worker_generation = None
            return
        try:
            result = self.worker_future.result()
        except Exception as error:
            self.worker_future = None
            self.worker_operation = ""
            self.active_worker_generation = None
            if operation == "DETECT_TIRES":
                self._schedule_detection_retry(f"{operation} worker failed: {error}")
            else:
                self.controller._error(f"{operation} worker failed: {error}")
            return
        self.worker_future = None
        self.worker_operation = ""
        self.active_worker_generation = None
        if operation == "DETECT_TIRES":
            if result.success:
                self.connection_status["vision"] = "ok"
                left = result.response.get("left", {})
                right = result.response.get("right", {})
                print(
                    f"[AUTO] detection left={left.get('status')} "
                    f"right={right.get('status')}"
                )
                if self._both_detections_final(result.response):
                    self.controller.apply_detection_result(result.response)
                else:
                    self._schedule_detection_retry(
                        f"detection incomplete: left={self._side_status(left)} "
                        f"right={self._side_status(right)}"
                    )
            else:
                self.connection_status["vision"] = "error"
                self._schedule_detection_retry(result.error or "vision detection failed")
        elif operation == "HELP_OPPOSITE":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.controller.apply_opposite_help_result(success, self._result_message(result))
        elif operation == "REMOVE_BAD_TIRE":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.controller.apply_remove_bad_tire_result(success, self._result_message(result))
        elif operation == "SET_ARUCO":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            side = self.controller.current_side
            if self.controller.apply_set_aruco_result(
                success, self._result_message(result)
            ) and side in ("left", "right"):
                print(f"[{side.upper()}] SET_ARUCO completed", flush=True)
        elif operation == "PLACE_REMOVED_TIRE":
            self.controller.apply_place_removed_tire_result(result.success, result.message)
        elif operation == "PICK_NEW_TIRE":
            self.controller.apply_pick_new_tire_result(result.success, result.message)
        elif operation == "INSTALL_NEW_TIRE":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.controller.apply_install_new_tire_result(success, self._result_message(result))
        elif operation == "HOME_BOTH":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.controller.apply_return_home_result(success, self._result_message(result))
        elif operation == "STOP":
            self._update_arm_status(result)
            self.controller.stop(self._result_message(result) or "STOP sent")
            self.parking.stop()
        elif operation == "HOME":
            self._update_arm_status(result)
            self.controller.last_status = self._result_message(result) or "HOME sent"

    def _worker_detect_tires(self):
        time.sleep(float(self.delays.get("after_parking_sec", 0.0)))
        try:
            return self.vision_client.request_detection(reset_cycle=True)
        except TypeError:
            return self.vision_client.request_detection()

    @staticmethod
    def _side_status(result):
        status = result.get("status")
        if status in {"GOOD", "BAD", "RECHECK", "ERROR"}:
            return status
        if result.get("stable") is True:
            if result.get("class_name") == "good_tire":
                return "GOOD"
            if result.get("class_name") == "bad_tire":
                return "BAD"
        return "RECHECK"

    def _both_detections_final(self, response):
        left = response.get("left", {})
        right = response.get("right", {})
        return all(
            self._side_status(result) in {"GOOD", "BAD"}
            for result in (left, right)
        )

    def _schedule_detection_retry(self, reason):
        maximum = max(1, int(self.operation.get("detection_max_retries", 3)))
        if self.detection_attempt_count >= maximum:
            self.controller.to_detection_error(
                f"{reason}; attempts={self.detection_attempt_count}/{maximum}"
            )
            return
        self.controller.prepare_detection_retry(
            f"{reason}; retry {self.detection_attempt_count + 1}/{maximum}"
        )
        delay = float(self.operation.get("detection_retry_delay_sec", 0.5))
        self.detection_retry_ready_at = time.monotonic() + max(0.0, delay)

    def _mock_log(self, message):
        if self.test_mode and not self.hardware_enabled:
            print(f"[MOCK] {message}", flush=True)

    def _log_mock_arm_stop(self):
        command = self.commands.get("stop", "STOP")
        self._mock_log(f"LEFT ARM {command}")
        self._mock_log(f"RIGHT ARM {command}")

    def _send_both_stop(self):
        return self.arm_manager.send_both(self.commands.get("stop", "STOP"))

    def _worker_transport(self, operation, side, slot):
        self._mock_log(f"{side.upper()} {operation} slot={slot}")
        if operation == "PLACE_REMOVED_TIRE":
            return self.transport_operations.place_removed_tire(side, slot)
        if operation == "PICK_NEW_TIRE":
            return self.transport_operations.pick_new_tire(side, slot)
        raise ValueError(f"unknown transport operation: {operation}")

    def _worker_home_both(self):
        self._mock_log("LEFT HOME")
        self._mock_log("RIGHT HOME")
        return self.arm_manager.send_both(self.commands.get("home", "HOME"))


    def _worker_arm_command(self, side, command_key, default_command):
        delay = 0.0
        if command_key == "help":
            delay = float(self.delays.get("after_detection_sec", 0.0))
        elif command_key == "remove_bad_tire":
            delay = float(self.delays.get("after_opposite_help_sec", 0.0))
        time.sleep(delay)
        command = self.commands.get(command_key, default_command)
        self._mock_log(f"{side.upper()} {command}")
        return self.arm_manager.send_side(side, command)


    def _worker_install_new_tire(self, side):
        command = self.commands.get("install_new_tire", "INSTALL_NEW_TIRE")
        if (
            bool(self.operation.get("mock_arms", True))
            and not bool(self.operation.get("mock_install_new_tire", False))
        ):
            return {
                side: ArmCommandResult(
                    arm=side,
                    command=command,
                    success=False,
                    error="NOT_IMPLEMENTED: INSTALL_NEW_TIRE hardware motion is disabled",
                )
            }
        self._mock_log(f"{side.upper()} {command}")
        return self.arm_manager.send_side(side, command)


    def start_stop(self):
        if self.controller.emergency_latched:
            self.controller.last_status = (
                "manual STOP ignored: emergency sequence active"
            )
            return False
        self._invalidate_worker()
        if self.worker_future is not None:
            self.worker_future.cancel()
        if not self.controller.stop("[AUTO] STOP requested"):
            return False
        self.parking.stop()
        stop_generation = self.worker_generation
        self._log_mock_arm_stop()

        def send_stop():
            results = self._send_both_stop()
            self._update_arm_status(results)
            if (
                self.controller.state == OperationState.STOPPED
                and self.worker_generation == stop_generation
            ):
                self.controller.last_status = (
                    self._result_message(results) or "STOP sent"
                )

        threading.Thread(target=send_stop, daemon=True).start()
        self._publish_pinky_command("STOP")
        print("[STATE] -> STOPPED", flush=True)
        return True

    def start_home(self):
        if self.controller.emergency_latched:
            self.controller.last_status = "manual HOME ignored: emergency active"
            return
        if self.worker_future is None:
            self._submit(
                "HOME",
                lambda: self.arm_manager.send_both(
                    self.commands.get("home", "HOME")
                ),
            )

    @staticmethod
    def _result_message(results):
        parts = []
        for side, result in sorted(results.items()):
            status = "OK" if result.success else "ERR"
            detail = result.response or result.error or ""
            parts.append(f"{side}:{status}:{detail}")
        return " | ".join(parts)

    def _update_arm_status(self, results):
        for side, result in results.items():
            self.connection_status[f"{side}_arm"] = "ok" if result.success else "error"

    def _draw(self, frame, parking_result):
        annotated = frame.copy()
        cv2.polylines(
            annotated,
            [self.parking.parking_detection_roi],
            True,
            (255, 0, 0),
            2,
        )
        for parking_roi in self.parking.parking_monitor.parking_rois:
            cv2.polylines(annotated, [parking_roi], True, (0, 255, 255), 2)

        selected = parking_result.selected_detection
        if selected is not None:
            x1, y1, x2, y2 = map(int, selected["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{selected['class_name']} {selected['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            if parking_result.center is not None:
                cv2.circle(annotated, parking_result.center, 5, (0, 0, 255), -1)
            for point, color in zip(
                self.parking.parking_monitor.entry_points or (),
                ((255, 0, 255), (0, 165, 255)),
            ):
                cv2.circle(annotated, point, 6, color, -1)

        lines = [
            "AUTO OPERATION",
            f"Armed: {self.controller.armed}",
            f"Parking state: {parking_result.parking_state.value}",
            f"Operation state: {self.controller.state.value}",
            f"Current operation: {self.controller.current_operation or '-'}",
            f"Replacement plan: {list(self.controller.replacement_plan)}",
            f"Current side/index: {self.controller.current_side or '-'} / {self.controller.replacement_index}",
            *self._connection_status_lines(self.connection_status),
            f"Pinky: {self.pinky_last_status}",
            f"Left tire: {self._fmt_det(self.controller.left_detection)}",
            f"Right tire: {self._fmt_det(self.controller.right_detection)}",
            f"Last successful step: {self.controller.last_successful_step or '-'}",
            f"Last status: {self.controller.last_status}",
            f"Last error: {self.controller.last_error or '-'}",
            "Controls: A arm/disarm | S stop | R reset | H home | Q quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (20, 28 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        safety_lines = [
            f"Safety monitoring: {'ON' if self._safety_monitor_active() else 'OFF'}",
            f"Safety reference: {self._fmt_center(self.safety_reference_center)}",
            f"Displacement: {self._fmt_metric(self.safety_displacement_px)} px",
            f"Jitter: {self._fmt_metric(self.safety_jitter_px)} px",
            f"Speed: {self._fmt_metric(self.safety_speed_px_sec)} px/s",
            f"Motion violation: {self.motion_violation_elapsed:.2f}s",
            f"Vehicle lost: {self.vehicle_lost_elapsed:.2f}s",
            f"Emergency latched: {self.controller.emergency_latched}",
            f"Emergency reason: {self.controller.emergency_reason or '-'}",
            f"Emergency STOP: {self._fmt_arm_results(self.emergency_stop_results)}",
            f"Emergency HOME: {self._fmt_arm_results(self.emergency_home_results)}",
        ]
        for i, line in enumerate(safety_lines):
            cv2.putText(
                annotated,
                line,
                (330, 22 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 255) if self.controller.emergency_latched else (255, 255, 0),
                1,
            )
        return annotated

    @staticmethod
    def _connection_status_lines(connection_status):
        devices = (
            ("Vision PC", "vision"),
            ("Left arm", "left_arm"),
            ("Right arm", "right_arm"),
        )
        return [
            f"{label}: {connection_status.get(key, '-')}"
            for label, key in devices
            if connection_status.get(key, "-") != "mock"
        ]

    @staticmethod
    def _fmt_center(center):
        if center is None:
            return "-"
        return f"({float(center[0]):.1f},{float(center[1]):.1f})"

    @staticmethod
    def _fmt_metric(value):
        return "-" if value is None else f"{float(value):.2f}"

    @staticmethod
    def _fmt_arm_results(results):
        if not results:
            return "-"
        return " ".join(
            f"{side[0].upper()}:{'OK' if result.success else 'ERR'}"
            for side, result in sorted(results.items())
        )

    @staticmethod
    def _fmt_det(detection):
        if not detection:
            return "-"
        return f"{detection.get('class_name')} {float(detection.get('confidence', 0.0)):.2f} stable={detection.get('stable')}"

    def close(self):
        close_vision = getattr(self.vision_client, "close", None)
        if close_vision is not None:
            close_vision()
        self.camera.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.emergency_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from robot_interfaces.srv import TransportTask
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("central_control"))
        default_parking = share / "config" / "parking.yaml"
        default_yolo = share / "config" / "yolo_vehicle.yaml"
        default_network = share / "config" / "network.yaml"
        default_operation = share / "config" / "operation.yaml"
        default_auto = share / "config" / "auto_operation.yaml"
    except Exception:
        default_parking = resolve_source_config("parking.yaml")
        default_yolo = resolve_source_config("yolo_vehicle.yaml")
        default_network = resolve_source_config("network.yaml")
        default_operation = resolve_source_config("operation.yaml")
        default_auto = resolve_source_config("auto_operation.yaml")

    rclpy.init(args=args)
    node = Node("auto_operation_node")
    for name, default in (("parking_config", str(default_parking)), ("yolo_config", str(default_yolo)), ("network_config", str(default_network)), ("operation_config", str(default_operation)), ("auto_config", str(default_auto))):
        node.declare_parameter(name, default)
    node.declare_parameter("mock_camera", True)
    node.declare_parameter("mock_vehicle_detector", True)
    node.declare_parameter("mock_arms", True)
    node.declare_parameter("mock_vision", False)
    node.declare_parameter("mock_transport_operations", True)
    node.declare_parameter("auto_start", False)
    node.declare_parameter("test_mode", False)
    node.declare_parameter("hardware_enabled", False)
    node.declare_parameter("vision_host", "192.168.0.81")
    node.declare_parameter("vision_port", 6000)
    node.declare_parameter("left_arm_host", "192.168.0.63")
    node.declare_parameter("left_arm_port", 5000)
    node.declare_parameter("right_arm_host", "192.168.0.112")
    node.declare_parameter("right_arm_port", 5001)
    node.declare_parameter("vehicle_motion_monitor_enabled", True)
    node.declare_parameter("emergency_jitter_px", 12.0)
    node.declare_parameter("emergency_speed_px_sec", 6.0)
    node.declare_parameter("emergency_reference_displacement_px", 12.0)
    node.declare_parameter("emergency_hold_sec", 0.35)
    node.declare_parameter("detection_lost_emergency_sec", 1.0)
    node.declare_parameter("send_stop_before_home", True)
    node.declare_parameter("stop_to_home_delay_sec", 0.5)
    node.declare_parameter("home_on_vehicle_motion", True)
    node.declare_parameter("require_manual_reset", True)
    node.declare_parameter("pinky_move_timeout_sec", -1.0)

    parking = load_yaml(node.get_parameter("parking_config").value)
    yolo = load_yaml(node.get_parameter("yolo_config").value)
    network = load_yaml(node.get_parameter("network_config").value)
    auto = load_yaml(node.get_parameter("auto_config").value)
    apply_overrides(parking, yolo, network, auto, {
        "mock_camera": str_to_bool(node.get_parameter("mock_camera").value),
        "mock_vehicle_detector": str_to_bool(node.get_parameter("mock_vehicle_detector").value),
        "mock_arms": str_to_bool(node.get_parameter("mock_arms").value),
        "mock_vision": str_to_bool(node.get_parameter("mock_vision").value),
        "vision_host": node.get_parameter("vision_host").value,
        "vision_port": node.get_parameter("vision_port").value,
        "left_arm_host": node.get_parameter("left_arm_host").value,
        "left_arm_port": node.get_parameter("left_arm_port").value,
        "right_arm_host": node.get_parameter("right_arm_host").value,
        "right_arm_port": node.get_parameter("right_arm_port").value,
    })
    operation = auto.setdefault("operation", {})
    operation["auto_start"] = str_to_bool(
        node.get_parameter("auto_start").value
    )
    operation["mock_transport_operations"] = str_to_bool(
        node.get_parameter("mock_transport_operations").value
    )
    test_mode = str_to_bool(node.get_parameter("test_mode").value)
    hardware_enabled = str_to_bool(
        node.get_parameter("hardware_enabled").value
    )
    apply_manual_test_mode_config(
        parking, yolo, auto, test_mode, hardware_enabled
    )
    safety = auto.setdefault("safety", {})
    for name in (
        "emergency_jitter_px",
        "emergency_speed_px_sec",
        "emergency_reference_displacement_px",
        "emergency_hold_sec",
        "detection_lost_emergency_sec",
        "stop_to_home_delay_sec",
    ):
        safety[name] = float(node.get_parameter(name).value)
    for name in (
        "vehicle_motion_monitor_enabled",
        "send_stop_before_home",
        "home_on_vehicle_motion",
        "require_manual_reset",
    ):
        safety[name] = str_to_bool(node.get_parameter(name).value)
    timeout_override = float(
        node.get_parameter("pinky_move_timeout_sec").value
    )
    timeouts = auto.setdefault("timeouts", {})
    if timeout_override > 0.0:
        timeouts["pinky_move_sec"] = timeout_override
    else:
        timeouts.setdefault("pinky_move_sec", 180.0)

    transport_config = auto.setdefault("transport_services", {})
    transport_operations = None
    if not bool(auto.get("operation", {}).get("mock_aruco_transport_operations", False)):
        transport_operations = RosTransportOperationHandler(
            node=node,
            service_type=TransportTask,
            service_names={
                "left": transport_config.get(
                    "left", "/left_arm/transport_task"
                ),
                "right": transport_config.get(
                    "right", "/right_arm/transport_task"
                ),
            },
            timeout_sec=float(timeouts.get("transport_service_sec", 120.0)),
            service_wait_timeout_sec=float(
                timeouts.get("transport_service_wait_sec", 2.0)
            ),
        )


    manual_vision = ManualTestVisionClient() if test_mode else None
    pinky_command_publisher = node.create_publisher(
        String, "/pinky/route_command", 10
    )

    def publish_pinky_command(command):
        if test_mode and not hardware_enabled:
            print(f"[MOCK] PINKY {command}", flush=True)
            return
        pinky_command_publisher.publish(String(data=command))

    full_auto_without_pinky = (
        not test_mode
        and hardware_enabled
        and not parking.get("mock_camera", False)
        and not yolo.get("mock_vehicle_detector", False)
        and not operation.get("mock_vision", False)
        and not operation.get("mock_arms", True)
        and not operation.get("mock_aruco_transport_operations", False)
        and operation.get("mock_transport_operations", False)
    )
    mode_name = (
        "FULL AUTO WITHOUT PINKY"
        if full_auto_without_pinky
        else ("MANUAL TEST" if test_mode else "AUTO OPERATION")
    )
    mode_lines = [
        f"MODE: {mode_name}",
        "",
        f"TOP CAMERA: {'MOCK' if parking.get('mock_camera', False) else 'REAL'}",
        "BEV: REAL",
        f"VEHICLE DETECTOR: {'MOCK' if yolo.get('mock_vehicle_detector', False) else 'REAL'}",
        f"TIRE VISION: {'MOCK' if operation.get('mock_vision', False) else 'REAL'}",
        f"LEFT ARM: {'MOCK' if operation.get('mock_arms', True) else 'REAL'}",
        f"RIGHT ARM: {'MOCK' if operation.get('mock_arms', True) else 'REAL'}",
        f"ARUCO TRANSPORT: {'MOCK' if operation.get('mock_aruco_transport_operations', False) else 'REAL'}",
        (
            "PINKY TRANSPORT: MOCK/AUTO-BYPASS"
            if operation.get("mock_transport_operations", False)
            else "PINKY TRANSPORT: REAL"
        ),
    ]
    print("\n".join(mode_lines), flush=True)

    runtime = AutoOperationRuntime(
        parking,
        yolo,
        network,
        auto,
        vision_client=manual_vision,
        transport_operations=transport_operations,
        pinky_command_sender=publish_pinky_command,
        pinky_subscription_count=pinky_command_publisher.get_subscription_count,
        test_mode=test_mode,
        hardware_enabled=hardware_enabled,
    )

    def on_transport_ready(_request, response):
        accepted = runtime.handle_transport_ready()
        response.success = accepted
        response.message = (
            f"TRANSPORT_READY accepted side={runtime.controller.current_side}"
            if accepted
            else f"TRANSPORT_READY rejected state={runtime.controller.state.value}"
        )
        return response

    _transport_ready_service = node.create_service(
        Trigger,
        transport_config.get("ready", "/central/transport_ready"),
        on_transport_ready,
    )


    def on_pinky_status(message):
        runtime.handle_pinky_route_status(message.data)

    _pinky_status_subscription = node.create_subscription(
        String, "/pinky/route_status", on_pinky_status, 10
    )

    keyboard = None
    if test_mode:
        keyboard = ManualTestKeyboard(runtime.enqueue_test_key)
        keyboard.start()

    def on_timer():
        runtime.step()
        if runtime.quit_requested:
            rclpy.shutdown()

    timer = node.create_timer(0.03, on_timer)
    del timer
    node.get_logger().info("auto operation node started")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if keyboard is not None:
            keyboard.stop()
        runtime.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
