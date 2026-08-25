import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("tire_vision")
    default_config = os.path.join(package_share, "config", "vision.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("mock_mode", default_value="true"),
            DeclareLaunchArgument("tcp_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("tcp_port", default_value="6000"),
            Node(
                package="tire_vision",
                executable="vision_server",
                name="tire_vision_server",
                output="screen",
                parameters=[{
                    "config_path": LaunchConfiguration("config_file"),
                    "mock_mode": LaunchConfiguration("mock_mode"),
                    "tcp_host": LaunchConfiguration("tcp_host"),
                    "tcp_port": LaunchConfiguration("tcp_port"),
                }],
            ),
        ]
    )
