import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("central_control")
    default_parking = os.path.join(package_share, "config", "parking.yaml")
    default_yolo = os.path.join(package_share, "config", "yolo_vehicle.yaml")
    default_network = os.path.join(package_share, "config", "network.yaml")
    default_operation = os.path.join(package_share, "config", "operation.yaml")
    default_auto = os.path.join(package_share, "config", "auto_operation.yaml")

    args = [
        DeclareLaunchArgument("parking_config", default_value=default_parking),
        DeclareLaunchArgument("yolo_config", default_value=default_yolo),
        DeclareLaunchArgument("network_config", default_value=default_network),
        DeclareLaunchArgument("operation_config", default_value=default_operation),
        DeclareLaunchArgument("auto_config", default_value=default_auto),
        DeclareLaunchArgument(
            "python_executable",
            default_value="python3",
        ),
        DeclareLaunchArgument("mock_camera", default_value="true"),
        DeclareLaunchArgument("mock_vehicle_detector", default_value="true"),
        DeclareLaunchArgument("mock_arms", default_value="true"),
        DeclareLaunchArgument("mock_vision", default_value="false"),
        DeclareLaunchArgument(
            "mock_transport_operations", default_value="true"
        ),
        DeclareLaunchArgument("auto_start", default_value="false"),
        DeclareLaunchArgument("test_mode", default_value="false"),
        DeclareLaunchArgument("hardware_enabled", default_value="false"),
        DeclareLaunchArgument("vision_host", default_value="192.168.0.81"),
        DeclareLaunchArgument("vision_port", default_value="6000"),
        DeclareLaunchArgument("left_arm_host", default_value="192.168.0.63"),
        DeclareLaunchArgument("left_arm_port", default_value="5000"),
        DeclareLaunchArgument("right_arm_host", default_value="192.168.0.112"),
        DeclareLaunchArgument("right_arm_port", default_value="5001"),
        DeclareLaunchArgument(
            "vehicle_motion_monitor_enabled", default_value="true"
        ),
        DeclareLaunchArgument("emergency_jitter_px", default_value="12.0"),
        DeclareLaunchArgument(
            "emergency_speed_px_sec", default_value="6.0"
        ),
        DeclareLaunchArgument(
            "emergency_reference_displacement_px", default_value="12.0"
        ),
        DeclareLaunchArgument("emergency_hold_sec", default_value="0.35"),
        DeclareLaunchArgument(
            "detection_lost_emergency_sec", default_value="1.0"
        ),
        DeclareLaunchArgument("send_stop_before_home", default_value="true"),
        DeclareLaunchArgument(
            "stop_to_home_delay_sec", default_value="0.5"
        ),
        DeclareLaunchArgument("home_on_vehicle_motion", default_value="true"),
        DeclareLaunchArgument("require_manual_reset", default_value="true"),
        DeclareLaunchArgument("pinky_move_timeout_sec", default_value="-1.0"),
    ]
    node = Node(
        package="central_control",
        executable="auto_operation",
        name="auto_operation_node",
        output="screen",
        prefix=LaunchConfiguration("python_executable"),
        parameters=[
            {"parking_config": LaunchConfiguration("parking_config")},
            {"yolo_config": LaunchConfiguration("yolo_config")},
            {"network_config": LaunchConfiguration("network_config")},
            {"operation_config": LaunchConfiguration("operation_config")},
            {"auto_config": LaunchConfiguration("auto_config")},
            {"mock_camera": LaunchConfiguration("mock_camera")},
            {"mock_vehicle_detector": LaunchConfiguration("mock_vehicle_detector")},
            {"mock_arms": LaunchConfiguration("mock_arms")},
            {"mock_vision": LaunchConfiguration("mock_vision")},
            {
                "mock_transport_operations": LaunchConfiguration(
                    "mock_transport_operations"
                )
            },
            {"auto_start": LaunchConfiguration("auto_start")},
            {"test_mode": LaunchConfiguration("test_mode")},
            {"hardware_enabled": LaunchConfiguration("hardware_enabled")},
            {"vision_host": LaunchConfiguration("vision_host")},
            {"vision_port": LaunchConfiguration("vision_port")},
            {"left_arm_host": LaunchConfiguration("left_arm_host")},
            {"left_arm_port": LaunchConfiguration("left_arm_port")},
            {"right_arm_host": LaunchConfiguration("right_arm_host")},
            {"right_arm_port": LaunchConfiguration("right_arm_port")},
            {
                "vehicle_motion_monitor_enabled": LaunchConfiguration(
                    "vehicle_motion_monitor_enabled"
                )
            },
            {"emergency_jitter_px": LaunchConfiguration("emergency_jitter_px")},
            {
                "emergency_speed_px_sec": LaunchConfiguration(
                    "emergency_speed_px_sec"
                )
            },
            {
                "emergency_reference_displacement_px": LaunchConfiguration(
                    "emergency_reference_displacement_px"
                )
            },
            {"emergency_hold_sec": LaunchConfiguration("emergency_hold_sec")},
            {
                "detection_lost_emergency_sec": LaunchConfiguration(
                    "detection_lost_emergency_sec"
                )
            },
            {
                "send_stop_before_home": LaunchConfiguration(
                    "send_stop_before_home"
                )
            },
            {
                "stop_to_home_delay_sec": LaunchConfiguration(
                    "stop_to_home_delay_sec"
                )
            },
            {
                "home_on_vehicle_motion": LaunchConfiguration(
                    "home_on_vehicle_motion"
                )
            },
            {
                "require_manual_reset": LaunchConfiguration(
                    "require_manual_reset"
                )
            },
            {
                "pinky_move_timeout_sec": LaunchConfiguration(
                    "pinky_move_timeout_sec"
                )
            },
        ],
    )
    pinky_bridge = Node(
        condition=UnlessCondition(
            LaunchConfiguration("mock_transport_operations")
        ),
        package="central_control",
        executable="pinky_tcp_bridge",
        name="pinky_tcp_bridge",
        output="screen",
        parameters=[
            {"network_config": LaunchConfiguration("network_config")},
        ],
    )
    return LaunchDescription(args + [pinky_bridge, node])
