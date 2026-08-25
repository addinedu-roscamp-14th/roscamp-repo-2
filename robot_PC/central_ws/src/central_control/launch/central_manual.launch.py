import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("central_control")
    default_parking = os.path.join(package_share, "config", "parking.yaml")
    default_yolo = os.path.join(package_share, "config", "yolo_vehicle.yaml")
    default_network = os.path.join(package_share, "config", "network.yaml")
    default_operation = os.path.join(package_share, "config", "operation.yaml")

    args = [
        DeclareLaunchArgument("parking_config", default_value=default_parking),
        DeclareLaunchArgument("yolo_config", default_value=default_yolo),
        DeclareLaunchArgument("network_config", default_value=default_network),
        DeclareLaunchArgument("operation_config", default_value=default_operation),
        DeclareLaunchArgument(
            "python_executable",
            default_value="python3",
        ),
        DeclareLaunchArgument("mock_camera", default_value="true"),
        DeclareLaunchArgument("mock_vehicle_detector", default_value="true"),
        DeclareLaunchArgument("mock_arms", default_value="true"),
        DeclareLaunchArgument("mock_vision", default_value="false"),
        DeclareLaunchArgument("vision_host", default_value="192.168.0.81"),
        DeclareLaunchArgument("vision_port", default_value="6000"),
        DeclareLaunchArgument("left_arm_host", default_value="192.168.0.63"),
        DeclareLaunchArgument("left_arm_port", default_value="5000"),
        DeclareLaunchArgument("right_arm_host", default_value="192.168.0.112"),
        DeclareLaunchArgument("right_arm_port", default_value="5001"),
    ]

    node = Node(
        package="central_control",
        executable="manual_central",
        name="manual_central_node",
        output="screen",
        prefix=LaunchConfiguration("python_executable"),
        parameters=[
            {"parking_config": LaunchConfiguration("parking_config")},
            {"yolo_config": LaunchConfiguration("yolo_config")},
            {"network_config": LaunchConfiguration("network_config")},
            {"operation_config": LaunchConfiguration("operation_config")},
            {"mock_camera": LaunchConfiguration("mock_camera")},
            {"mock_vehicle_detector": LaunchConfiguration("mock_vehicle_detector")},
            {"mock_arms": LaunchConfiguration("mock_arms")},
            {"mock_vision": LaunchConfiguration("mock_vision")},
            {"vision_host": LaunchConfiguration("vision_host")},
            {"vision_port": LaunchConfiguration("vision_port")},
            {"left_arm_host": LaunchConfiguration("left_arm_host")},
            {"left_arm_port": LaunchConfiguration("left_arm_port")},
            {"right_arm_host": LaunchConfiguration("right_arm_host")},
            {"right_arm_port": LaunchConfiguration("right_arm_port")},
        ],
    )
    return LaunchDescription(args + [node])
