from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
import time

import cv2
import numpy as np
import yaml

from central_control.dual_arm_manager import DualArmManager
from central_control.key_handler import KeyAction, KeyHandler
from central_control.parking_monitor import (
    ParkingMonitor,
    ParkingState,
    vehicle_entry_points,
)
from central_control.top_view import TopViewTransformer
from central_control.vehicle_detector import create_vehicle_detector
from central_control.vision_client import VisionClient
from central_control.work_state_machine import WorkState, WorkStateMachine


class CentralState(str, Enum):
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


@dataclass(frozen=True)
class ParkingFrameResult:
    state: CentralState
    parking_state: ParkingState
    parked_event: bool
    vehicle_detected: bool
    vehicle_in_roi: bool
    vehicle_stopped: bool
    selected_detection: dict | None
    center: tuple[int, int] | None


def load_yaml(path: str | os.PathLike) -> dict:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    model_path = config.get("model_path")
    if model_path and not Path(str(model_path)).is_absolute():
        config["model_path"] = str(resolved.parent.parent / str(model_path))
    return config


def resolve_source_config(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "config" / filename


def build_parking_monitor(config: dict) -> ParkingMonitor:
    return ParkingMonitor(
        parking_roi=config.get("parking_rois", config["parking_roi"]),
        stop_threshold=config.get("parking_move_threshold", 4.0),
        stop_hold_time=config.get("parking_stop_seconds", 10.0),
        center_history=config.get("parking_center_history", 10),
        median_samples=config.get("parking_median_samples", 5),
        motion_window_sec=config.get("parking_motion_window_sec", 1.0),
        stop_jitter_threshold=config.get("parking_stop_jitter_px", 6.0),
        move_jitter_threshold=config.get("parking_move_jitter_px", 10.0),
        stop_speed_threshold=config.get("parking_stop_speed_px_sec", 2.0),
        move_speed_threshold=config.get("parking_move_speed_px_sec", 5.0),
        motion_grace_sec=config.get("parking_motion_grace_sec", 0.4),
        detection_lost_grace_sec=config.get("parking_detection_lost_grace_sec", 0.3),
        roi_exit_grace_frames=config.get("parking_roi_exit_grace_frames", 3),
    )


class CentralParkingStateMachine:
    def __init__(self, parking_config: dict):
        self.parking_config = parking_config
        self.parking_monitor = build_parking_monitor(parking_config)
        self.parking_detection_roi = np.asarray(
            parking_config["parking_detection_roi"], dtype=np.int32
        )
        self.parking_entry_point_inset_ratio = float(
            parking_config.get("parking_entry_point_inset_ratio", 0.1)
        )
        self.state = CentralState.WAITING_FOR_VEHICLE
        self.parked_latched = False
        self.last_parking_state = ParkingState.WAITING
        self.last_status = "READY"

    def reset(self):
        self.parking_monitor.triggered = False
        self.parking_monitor.state = ParkingState.WAITING
        self.parking_monitor.reset_motion()
        self.state = CentralState.WAITING_FOR_VEHICLE
        self.parked_latched = False
        self.last_parking_state = ParkingState.WAITING
        self.last_status = "RESET"

    def stop(self):
        self.state = CentralState.STOPPED
        self.last_status = "STOP"

    def bypass_parking(self):
        self.parked_latched = True
        self.parking_monitor.state = ParkingState.TRIGGER
        self.parking_monitor.triggered = True
        self.state = CentralState.WAIT_DETECTION_CONFIRM
        self.last_status = "PARKING BYPASSED: WAIT DETECTION"

    def select_vehicle(self, detections):
        candidates = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if cv2.pointPolygonTest(
                self.parking_detection_roi, center, False
            ) >= 0:
                candidates.append((detection, center))

        if not candidates:
            return None, None

        selected, center = max(
            candidates,
            key=lambda item: (
                self.parking_monitor.contains(item[1]),
                item[0]["confidence"],
            ),
        )
        return selected, center

    def process_detections(self, detections, now=None) -> ParkingFrameResult:
        selected, center = self.select_vehicle(detections)
        entry_points = None
        if selected is not None:
            entry_points = vehicle_entry_points(
                selected["bbox"], self.parking_entry_point_inset_ratio
            )
        parking_state = self.parking_monitor.update(
            center, now=now, entry_points=entry_points
        )

        if selected is None and not self.parking_monitor.entry_confirmed:
            self.parked_latched = False
            if self.state != CentralState.STOPPED:
                self.state = CentralState.WAITING_FOR_VEHICLE

        parked_event = False
        if parking_state == ParkingState.PARKED and not self.parked_latched:
            self.parked_latched = True
            parked_event = True
            self.state = CentralState.WAIT_DETECTION_CONFIRM
            self.last_status = "PARKED: WAIT SPACE FOR DETECTION"
            self.parking_monitor.consume_parked()
        elif self.state not in (
            CentralState.WAIT_SET_CAMERA_CONFIRM,
            CentralState.SETTING_CAMERA,
            CentralState.WAIT_DETECTION_CONFIRM,
            CentralState.DETECTING_TIRES,
            CentralState.WAIT_ACTION_CONFIRM,
            CentralState.EXECUTING_ACTION,
            CentralState.WAIT_ARUCO_CONFIRM,
            CentralState.SETTING_ARUCO,
            CentralState.WAIT_HOME_CONFIRM,
            CentralState.RETURNING_HOME,
            CentralState.COMPLETED,
            CentralState.STOPPED,
            CentralState.ERROR,
        ):
            self.state = CentralState.WAITING_FOR_VEHICLE

        self.last_parking_state = parking_state
        return ParkingFrameResult(
            state=self.state,
            parking_state=parking_state,
            parked_event=parked_event,
            vehicle_detected=selected is not None,
            vehicle_in_roi=self.parking_monitor.fully_inside,
            vehicle_stopped=self.parking_monitor.stop_start_time is not None,
            selected_detection=selected,
            center=center,
        )

    def sync_work_state(self, work_state: WorkState):
        if work_state.value in CentralState.__members__:
            self.state = CentralState(work_state.value)


def draw_parking_overlay(
    frame,
    state_machine: CentralParkingStateMachine,
    work_state_machine: WorkStateMachine,
    result,
    connection_status: dict,
):
    annotated = frame.copy()
    cv2.polylines(
        annotated, [state_machine.parking_detection_roi], True, (255, 0, 0), 2
    )
    for parking_roi in state_machine.parking_monitor.parking_rois:
        cv2.polylines(annotated, [parking_roi], True, (0, 255, 255), 2)

    selected = result.selected_detection
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
        if result.center is not None:
            cv2.circle(annotated, result.center, 5, (0, 0, 255), -1)
        for point, color in zip(
            state_machine.parking_monitor.entry_points or (),
            ((255, 0, 255), (0, 165, 255)),
        ):
            cv2.circle(annotated, point, 6, color, -1)

    monitor = state_machine.parking_monitor
    jitter = monitor.jitter_radius
    speed = monitor.speed
    jitter_text = "-" if jitter is None else f"{jitter:.2f}px"
    speed_text = "-" if speed is None else f"{speed:.2f}px/s"
    left = _format_detection(work_state_machine.left_detection)
    right = _format_detection(work_state_machine.right_detection)
    targets = ",".join(sorted(work_state_machine.target_arms)) or "-"
    lines = [
        f"Parking: {result.parking_state.value}",
        f"Work: {work_state_machine.state.value}",
        f"Vision PC: {connection_status.get('vision', '-')}",
        f"Left arm: {connection_status.get('left_arm', '-')}",
        f"Right arm: {connection_status.get('right_arm', '-')}",
        f"Vehicle: {result.vehicle_detected} ROI: {result.vehicle_in_roi}",
        f"Stopped: {result.vehicle_stopped} Stop: {monitor.stopped_seconds():.1f}/{monitor.stop_hold_time:.1f}s",
        f"Jitter: {jitter_text} Speed: {speed_text}",
        f"Left tire: {left}",
        f"Right tire: {right}",
        f"Targets: {targets} Busy: {work_state_machine.busy}",
        f"Last: {work_state_machine.last_status}",
        f"Error: {work_state_machine.last_error or '-'}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (20, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    return annotated


def _format_detection(detection):
    if not detection:
        return "-"
    return f"{detection.get('class_name')} {float(detection.get('confidence', 0.0)):.2f} stable={detection.get('stable')}"


class FakeTopCamera:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

    def read(self):
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self):
        pass


def open_top_camera(config):
    if bool(config.get("mock_camera", True)):
        return FakeTopCamera(
            config.get("top_camera_process_width", config.get("width", 600)),
            config.get("top_camera_process_height", config.get("height", 400)),
        )
    cap = cv2.VideoCapture(config.get("camera_device", 0), cv2.CAP_V4L2)
    camera_format = str(config.get("format", "")).strip()
    if len(camera_format) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.get("width", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.get("height", 480)))
    cap.set(cv2.CAP_PROP_FPS, int(config.get("fps", 20)))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"top camera open failed: {config.get('camera_device')}")
    return cap


class ManualCentralRuntime:
    def __init__(
        self,
        parking_config: dict,
        yolo_config: dict,
        network_config: dict,
        operation_config: dict,
    ):
        self.parking_config = parking_config
        self.yolo_config = yolo_config
        self.network_config = network_config
        self.operation_config = operation_config
        self.commands = operation_config.get("commands", {})
        self.state_machine = CentralParkingStateMachine(parking_config)
        self.work_state_machine = WorkStateMachine()
        self.key_handler = KeyHandler()
        self.detector = create_vehicle_detector(yolo_config)
        self.camera = open_top_camera(parking_config)
        self.top_view = TopViewTransformer(parking_config)
        self.show_window = bool(yolo_config.get("show_window", True))
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.worker_future: Future | None = None
        self.worker_operation = ""
        self.quit_requested = False
        self.connection_status = {
            "vision": "mock" if operation_config.get("operation", {}).get("mock_vision", False) else "configured",
            "left_arm": "mock" if operation_config.get("operation", {}).get("mock_arms", True) else "configured",
            "right_arm": "mock" if operation_config.get("operation", {}).get("mock_arms", True) else "configured",
        }
        self.arm_manager = DualArmManager.from_config(network_config, operation_config)
        vision_cfg = dict(network_config["vision_pc"])
        self.vision_client = VisionClient(
            mock_mode=bool(operation_config.get("operation", {}).get("mock_vision", False)),
            **vision_cfg,
        )

    def step(self):
        self._apply_worker_result_if_ready()
        ok, frame = self.camera.read()
        if not ok:
            result = self.state_machine.process_detections([], now=time.monotonic())
        else:
            frame = self.top_view.transform(frame)
            detections, _ = self.detector.detect(frame)
            result = self.state_machine.process_detections(detections)
            if result.parked_event:
                self.work_state_machine.parking_completed()
            self.state_machine.sync_work_state(self.work_state_machine.state)

        annotated = None
        if ok:
            annotated = draw_parking_overlay(
                frame,
                self.state_machine,
                self.work_state_machine,
                result,
                self.connection_status,
            )
            if self.show_window:
                cv2.imshow("Central Manual - Parking", annotated)
                key = cv2.waitKey(1) & 0xFF
                self.handle_key(key)
        return result, annotated

    def handle_key(self, key):
        decision = self.key_handler.decide(key, self.work_state_machine.busy)
        if decision is None:
            return None
        if decision.action == KeyAction.STOP:
            self.start_stop_worker()
        elif decision.action == KeyAction.QUIT:
            self.quit_requested = True
        elif decision.action == KeyAction.RESET:
            self.reset()
        elif decision.action == KeyAction.BYPASS_PARKING:
            self.state_machine.bypass_parking()
            self.work_state_machine.parking_completed()
            self.state_machine.sync_work_state(self.work_state_machine.state)
        elif decision.action == KeyAction.HOME:
            self.start_manual_home_worker()
        elif decision.action == KeyAction.STEP:
            self.start_current_step_worker()
        elif decision.action == KeyAction.IGNORED:
            self.work_state_machine.last_status = decision.reason
        return decision.action

    def reset(self):
        if self.worker_future is not None and not self.worker_future.done():
            self.work_state_machine.last_status = "RESET ignored while worker is running"
            return
        self.state_machine.reset()
        self.work_state_machine.reset()

    def start_current_step_worker(self):
        state = self.work_state_machine.state
        if state == WorkState.WAIT_DETECTION_CONFIRM and self.work_state_machine.start_detection():
            self._submit_worker("DETECT_TIRES", self._worker_detect_tires)
        elif state == WorkState.WAIT_ACTION_CONFIRM and self.work_state_machine.start_action():
            self._submit_worker("REMOVE_BAD_TIRE", self._worker_action)
        elif state == WorkState.WAIT_ARUCO_CONFIRM and self.work_state_machine.start_aruco():
            self._submit_worker("SET_ARUCO", self._worker_aruco)
        elif state == WorkState.WAIT_HOME_CONFIRM and self.work_state_machine.start_home():
            self._submit_worker("HOME", self._worker_home)
        else:
            self.work_state_machine.last_status = f"SPACE ignored in {state.value}"

    def start_manual_home_worker(self):
        if self.work_state_machine.busy:
            self.work_state_machine.last_status = "HOME ignored while busy"
            return
        if self.work_state_machine.set_busy(WorkState.RETURNING_HOME, "MANUAL_HOME"):
            self._submit_worker("HOME", self._worker_home)

    def start_stop_worker(self):
        if self.worker_future is not None and not self.worker_future.done():
            self.work_state_machine.last_status = "STOP requested while worker is running"
        self._submit_worker("STOP", self._worker_stop, force=True)

    def _submit_worker(self, operation, func, force=False):
        if not force and self.worker_future is not None and not self.worker_future.done():
            self.work_state_machine.last_status = f"{operation} ignored: worker running"
            return
        self.worker_operation = operation
        self.worker_future = self.executor.submit(func)

    def _apply_worker_result_if_ready(self):
        if self.worker_future is None or not self.worker_future.done():
            return
        operation = self.worker_operation
        try:
            result = self.worker_future.result()
        except Exception as error:
            self.work_state_machine.to_error(f"{operation} worker failed: {error}")
            self.worker_future = None
            return
        self.worker_future = None
        if operation == "DETECT_TIRES":
            if result.success:
                self.connection_status["vision"] = "ok"
                self.work_state_machine.apply_detection_result(result.response)
            else:
                self.connection_status["vision"] = "error"
                self.work_state_machine.apply_detection_result(None, result.error)
        elif operation == "REMOVE_BAD_TIRE":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.work_state_machine.apply_action_result(success, self._result_message(result))
        elif operation == "SET_ARUCO":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.work_state_machine.apply_aruco_result(success, self._result_message(result))
        elif operation == "HOME":
            success = self.arm_manager.all_success(result)
            self._update_arm_status(result)
            self.work_state_machine.apply_home_result(success, self._result_message(result))
        elif operation == "STOP":
            self._update_arm_status(result)
            self.work_state_machine.stop(self._result_message(result) or "STOP sent")
            self.state_machine.stop()
        self.state_machine.sync_work_state(self.work_state_machine.state)

    def _worker_detect_tires(self):
        return self.vision_client.request_detection()

    def _worker_action(self):
        command = self.commands.get("remove_bad_tire", "REMOVE_BAD_TIRE")
        return self.arm_manager.send_selected(
            left_command=command if "left" in self.work_state_machine.target_arms else None,
            right_command=command if "right" in self.work_state_machine.target_arms else None,
        )

    def _worker_aruco(self):
        command = self.commands.get("set_aruco", "SET_ARUCO")
        return self.arm_manager.send_selected(
            left_command=command if "left" in self.work_state_machine.target_arms else None,
            right_command=command if "right" in self.work_state_machine.target_arms else None,
        )

    def _worker_home(self):
        return self.arm_manager.send_both(self.commands.get("home", "HOME"))

    def _worker_stop(self):
        return self.arm_manager.send_both(self.commands.get("stop", "STOP"))

    def _update_arm_status(self, results):
        for side, result in results.items():
            self.connection_status[f"{side}_arm"] = "ok" if result.success else "error"

    @staticmethod
    def _result_message(results):
        parts = []
        for side, result in sorted(results.items()):
            status = "OK" if result.success else "ERR"
            detail = result.response or result.error or ""
            parts.append(f"{side}:{status}:{detail}")
        return " | ".join(parts)

    def close(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False, cancel_futures=True)


def apply_overrides(parking_config, yolo_config, network_config, operation_config, overrides):
    if overrides.get("mock_camera") is not None:
        parking_config["mock_camera"] = overrides["mock_camera"]
    if overrides.get("mock_vehicle_detector") is not None:
        yolo_config["mock_vehicle_detector"] = overrides["mock_vehicle_detector"]
    operation = operation_config.setdefault("operation", {})
    if overrides.get("mock_arms") is not None:
        operation["mock_arms"] = overrides["mock_arms"]
    if overrides.get("mock_vision") is not None:
        operation["mock_vision"] = overrides["mock_vision"]
    if overrides.get("vision_host"):
        network_config["vision_pc"]["host"] = overrides["vision_host"]
    if overrides.get("vision_port") is not None:
        network_config["vision_pc"]["port"] = int(overrides["vision_port"])
    if overrides.get("left_arm_host"):
        network_config["left_arm"]["host"] = overrides["left_arm_host"]
    if overrides.get("left_arm_port") is not None:
        network_config["left_arm"]["port"] = int(overrides["left_arm_port"])
    if overrides.get("right_arm_host"):
        network_config["right_arm"]["host"] = overrides["right_arm_host"]
    if overrides.get("right_arm_port") is not None:
        network_config["right_arm"]["port"] = int(overrides["right_arm_port"])


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def main(args=None):
    import rclpy
    from rclpy.node import Node

    try:
        from ament_index_python.packages import get_package_share_directory
        package_share = Path(get_package_share_directory("central_control"))
        default_parking = package_share / "config" / "parking.yaml"
        default_yolo = package_share / "config" / "yolo_vehicle.yaml"
        default_network = package_share / "config" / "network.yaml"
        default_operation = package_share / "config" / "operation.yaml"
    except Exception:
        default_parking = resolve_source_config("parking.yaml")
        default_yolo = resolve_source_config("yolo_vehicle.yaml")
        default_network = resolve_source_config("network.yaml")
        default_operation = resolve_source_config("operation.yaml")

    rclpy.init(args=args)
    node = Node("manual_central_node")
    node.declare_parameter("parking_config", str(default_parking))
    node.declare_parameter("yolo_config", str(default_yolo))
    node.declare_parameter("network_config", str(default_network))
    node.declare_parameter("operation_config", str(default_operation))
    node.declare_parameter("mock_camera", True)
    node.declare_parameter("mock_vehicle_detector", True)
    node.declare_parameter("mock_arms", True)
    node.declare_parameter("mock_vision", False)
    node.declare_parameter("vision_host", "192.168.0.81")
    node.declare_parameter("vision_port", 6000)
    node.declare_parameter("left_arm_host", "192.168.0.63")
    node.declare_parameter("left_arm_port", 5000)
    node.declare_parameter("right_arm_host", "192.168.0.112")
    node.declare_parameter("right_arm_port", 5001)

    parking_config = load_yaml(node.get_parameter("parking_config").value)
    yolo_config = load_yaml(node.get_parameter("yolo_config").value)
    network_config = load_yaml(node.get_parameter("network_config").value)
    operation_config = load_yaml(node.get_parameter("operation_config").value)
    apply_overrides(
        parking_config,
        yolo_config,
        network_config,
        operation_config,
        {
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
        },
    )
    runtime = ManualCentralRuntime(parking_config, yolo_config, network_config, operation_config)

    def on_timer():
        runtime.step()
        if runtime.quit_requested:
            rclpy.shutdown()

    timer = node.create_timer(0.03, on_timer)
    del timer
    node.get_logger().info("manual central node started; parked state waits for SPACE")

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
