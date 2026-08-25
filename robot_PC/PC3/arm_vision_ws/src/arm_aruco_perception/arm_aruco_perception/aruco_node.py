from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np

from arm_aruco_perception.aruco_detector import ArucoDetector
from arm_aruco_perception.pose_stabilizer import PoseStabilizer


def load_camera_calibration(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"camera calibration file is missing: {path}")
    data = np.load(path, allow_pickle=False)
    return data["camera_matrix"], data["dist_coeffs"]


class ArucoCameraRunner:
    """Camera-only runner; it never imports or controls pymycobot."""

    def __init__(self, side, camera_device, camera_calibration_file, **config):
        self.side = side
        matrix, distortion = load_camera_calibration(camera_calibration_file)
        self.detector = ArucoDetector(
            matrix, distortion,
            marker_length_m=config.get("marker_length_m", 0.030),
            target_id=config.get("target_id"),
            max_reprojection_error_px=config.get("max_reprojection_error_px", 2.5),
        )
        self.stabilizer = PoseStabilizer(
            buffer_size=config.get("stable_buffer_size", 12),
            min_samples=config.get("min_stable_samples", 8),
            max_translation_spread_mm=config.get("max_translation_spread_mm", 3.0),
            max_rotation_spread_deg=config.get("max_rotation_spread_deg", 2.0),
        )
        self.capture = cv2.VideoCapture(camera_device)

    def read_stable(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"{self.side} gripper camera frame unavailable")
        detections = self.detector.detect(frame)
        if not detections:
            self.stabilizer.reset()
            return None
        detected = detections[0]
        stable = self.stabilizer.update(detected["transform_camera_marker"])
        if stable is None:
            return None
        return {
            "side": self.side,
            "marker_id": detected["marker_id"],
            **stable,
        }


def run_ros_node(side):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class NodeImpl(Node):
        def __init__(self):
            super().__init__(f"{side}_aruco_node")
            self.declare_parameter("camera_device", "")
            self.declare_parameter("camera_calibration_file", "")
            device = self.get_parameter("camera_device").value
            calibration = self.get_parameter("camera_calibration_file").value
            if not device:
                raise RuntimeError(f"{side} camera_device ROS parameter is required")
            if not calibration:
                raise RuntimeError(f"{side} camera_calibration_file ROS parameter is required")
            self.runner = ArucoCameraRunner(side, device, calibration)
            self.publisher = self.create_publisher(String, f"/arm_vision/{side}/aruco_pose", 10)
            self.create_timer(0.05, self.tick)

        def tick(self):
            result = self.runner.read_stable()
            if result:
                payload = dict(result)
                payload["transform"] = result["transform"].tolist()
                message = String()
                message.data = json.dumps(payload)
                self.publisher.publish(message)

    rclpy.init()
    node = NodeImpl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
