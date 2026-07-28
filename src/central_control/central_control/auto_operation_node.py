from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import math
import threading
import time

import cv2

from central_control.arm_client import ArmCommandResult
from central_control.dual_arm_manager import DualArmManager
from central_control.manual_central_node import (
    CentralParkingStateMachine,
    apply_overrides,
    load_yaml,
    open_top_camera,
    resolve_source_config,
    str_to_bool,
)
from central_control.operation_controller import OperationController, OperationState
from central_control.stop_detector import StopDetector
from central_control.top_view import TopViewTransformer
from central_control.vehicle_detector import create_vehicle_detector
from central_control.vision_client import VisionClient


SAFETY_MONITORED_STATES = {
    OperationState.SETTING_CAMERA,
    OperationState.DETECTING_TIRES,
    OperationState.CHECKING_LEFT_TIRE,
    OperationState.EXECUTING_RIGHT_HELP,
    OperationState.EXECUTING_LEFT_REMOVE,
}


class AutoOperationRuntime:
    def __init__(
        self,
        parking_config: dict,
        yolo_config: dict,
        network_config: dict,
        auto_config: dict,
        arm_manager=None,
        vision_client=None,
    ):
        self.parking_config = parking_config
        self.yolo_config = yolo_config
        self.network_config = network_config
        self.auto_config = auto_config
        self.operation = auto_config.get("operation", {})
        self.commands = auto_config.get("commands", {})
        self.delays = auto_config.get("delays", {})
        self.safety = auto_config.get("safety", {})
        self.parking = CentralParkingStateMachine(parking_config)
        self.controller = OperationController()
        self.detector = create_vehicle_detector(yolo_config)
        self.camera = open_top_camera(parking_config)
        self.top_view = TopViewTransformer(parking_config)
        self.show_window = bool(yolo_config.get("show_window", True))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.emergency_executor = ThreadPoolExecutor(max_workers=1)
        self.worker_future: Future | None = None
        self.worker_operation = ""
        self.emergency_future: Future | None = None
        self.emergency_operation = ""
        self.emergency_stop_results = {}
        self.emergency_home_results = {}
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

    def reset_cycle(self):
        self.parking.reset()
        self.controller.reset()
        self._reset_safety_tracking(clear_reference=True)
        self.emergency_stop_results = {}
        self.emergency_home_results = {}

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
        if self.worker_future is not None:
            self.worker_future.cancel()
        print(f"[EMERGENCY] {reason}")
        if bool(self.safety.get("send_stop_before_home", True)):
            self.controller.mark_emergency_stopping()
            self.emergency_operation = "STOP"
            self.emergency_future = self.emergency_executor.submit(
                self.arm_manager.send_both,
                self.commands.get("stop", "STOP"),
            )
        elif bool(self.safety.get("home_on_vehicle_motion", True)):
            self._start_emergency_home()
        else:
            self.controller.apply_emergency_home_result(
                True, "[EMERGENCY] latched; STOP/HOME transmission disabled"
            )
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
            if self.controller.start_set_camera():
                print("[AUTO] sending SET_CAMERA to both arms")
                self._submit("SET_CAMERA", self._worker_set_camera)
        elif self.controller.state == OperationState.DETECTING_TIRES:
            if self.controller.start_detection():
                print("[AUTO] requesting tire detection")
                self._submit("DETECT_TIRES", self._worker_detect_tires)
        elif self.controller.state == OperationState.CHECKING_LEFT_TIRE:
            decision = self.controller.evaluate_left_tire()
            if decision == "right_help":
                self._submit("RIGHT_HELP", self._worker_right_help)
            elif decision == "completed":
                print("[AUTO] cycle completed")

    def _submit(self, operation, func):
        if self.worker_future is not None:
            self.controller.last_status = f"{operation} ignored: worker running"
            return
        self.worker_operation = operation
        self.worker_future = self.executor.submit(func)

    def _apply_worker_result_if_ready(self):
        if self.worker_future is None or not self.worker_future.done():
            return
        operation = self.worker_operation
        if self.controller.emergency_latched:
            try:
                self.worker_future.result()
            except Exception:
                pass
            self.worker_future = None
            self.worker_operation = ""
            return
        try:
            result = self.worker_future.result()
        except Exception as error:
            self.controller.apply_detection_result(None, f"{operation} worker failed: {error}")
            self.worker_future = None
            return
        self.worker_future = None
        if operation == "SET_CAMERA":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            if success:
                print("[AUTO] SET_CAMERA complete left=OK right=OK")
            self.controller.apply_set_camera_result(success, self._result_message(result))
        elif operation == "DETECT_TIRES":
            if result.success:
                self.connection_status["vision"] = "ok"
                left = result.response.get("left", {})
                right = result.response.get("right", {})
                print(f"[AUTO] detection left={left.get('class_name')} right={right.get('class_name')}")
                self.controller.apply_detection_result(result.response)
            else:
                self.connection_status["vision"] = "error"
                self.controller.apply_detection_result(None, result.error)
        elif operation == "RIGHT_HELP":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            if success:
                print("[AUTO] right HELP complete")
            self.controller.apply_right_help_result(success, self._result_message(result))
            if success:
                self._submit("LEFT_REMOVE", self._worker_left_remove)
        elif operation == "LEFT_REMOVE":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            if success:
                print("[AUTO] cycle completed")
            self.controller.apply_left_remove_result(success, self._result_message(result))
        elif operation == "STOP":
            self._update_arm_status(result)
            self.controller.stop(self._result_message(result) or "STOP sent")
            self.parking.stop()
        elif operation == "HOME":
            self._update_arm_status(result)
            self.controller.last_status = self._result_message(result) or "HOME sent"

    def _worker_set_camera(self):
        time.sleep(float(self.delays.get("after_parking_sec", 0.0)))
        return self.arm_manager.send_both(self.commands.get("set_camera", "SET_CAMERA"))

    def _worker_detect_tires(self):
        time.sleep(float(self.delays.get("after_set_camera_sec", 0.0)))
        return self.vision_client.request_detection()

    def _worker_right_help(self):
        time.sleep(float(self.delays.get("after_detection_sec", 0.0)))
        return self.arm_manager.send_right(self.commands.get("right_help", "HELP"))

    def _worker_left_remove(self):
        time.sleep(float(self.delays.get("after_right_help_sec", 0.0)))
        return self.arm_manager.send_left(self.commands.get("left_remove_bad_tire", "REMOVE_BAD_TIRE"))

    def start_stop(self):
        if self.controller.emergency_latched:
            self.controller.last_status = (
                "manual STOP ignored: emergency sequence active"
            )
            return
        self.controller.stop("[AUTO] STOP requested")
        self.parking.stop()

        def send_stop():
            results = self.arm_manager.send_both(self.commands.get("stop", "STOP"))
            self._update_arm_status(results)
            self.controller.last_status = self._result_message(results) or "STOP sent"

        threading.Thread(target=send_stop, daemon=True).start()

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
            f"Vision PC: {self.connection_status.get('vision', '-')}",
            f"Left arm: {self.connection_status.get('left_arm', '-')}",
            f"Right arm: {self.connection_status.get('right_arm', '-')}",
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
        self.camera.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.emergency_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    import rclpy
    from rclpy.node import Node

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
    node.declare_parameter("auto_start", False)
    node.declare_parameter("vision_host", "192.168.0.81")
    node.declare_parameter("vision_port", 6000)
    node.declare_parameter("left_arm_host", "192.168.0.115")
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
    auto.setdefault("operation", {})["auto_start"] = str_to_bool(
        node.get_parameter("auto_start").value
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
    runtime = AutoOperationRuntime(parking, yolo, network, auto)

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
        runtime.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
