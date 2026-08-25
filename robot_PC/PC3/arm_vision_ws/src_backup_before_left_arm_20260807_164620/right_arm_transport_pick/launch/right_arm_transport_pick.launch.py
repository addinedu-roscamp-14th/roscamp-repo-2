from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    names = {
        "camera_device": "/dev/video4",
        "robot_host": "192.168.0.112",
        "robot_port": "5001",
        "preview_only": "true",
        "allow_robot_motion": "false",
        "allow_gripper_close": "false",
        "slot_id": "1",
    }
    arguments = [DeclareLaunchArgument(name, default_value=value) for name, value in names.items()]
    parameters = {name: LaunchConfiguration(name) for name in names}
    config_file = PathJoinSubstitution([FindPackageShare("right_arm_transport_pick"), "config", "right_arm_transport_pick.yaml"])
    return LaunchDescription(arguments + [
        Node(
            package="right_arm_transport_pick",
            executable="right_arm_transport_pick",
            name="right_arm_transport_pick",
            output="screen",
            parameters=[config_file, parameters],
        )
    ])
