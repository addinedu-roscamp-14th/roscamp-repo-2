from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="tire_arm_control",
                executable="tcp_server_node",
                name="tcp_arm_server_node",
                output="screen",
            )
        ]
    )
