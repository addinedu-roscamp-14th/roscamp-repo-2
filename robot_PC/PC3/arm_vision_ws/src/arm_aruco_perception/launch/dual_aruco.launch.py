from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("left_camera_device", default_value=""),
        DeclareLaunchArgument("right_camera_device", default_value=""),
        DeclareLaunchArgument("left_camera_calibration_file", default_value=""),
        DeclareLaunchArgument("right_camera_calibration_file", default_value=""),
        Node(
            package="arm_aruco_perception", executable="left_aruco_node",
            parameters=[{
                "camera_device": LaunchConfiguration("left_camera_device"),
                "camera_calibration_file": LaunchConfiguration("left_camera_calibration_file"),
            }],
        ),
        Node(
            package="arm_aruco_perception", executable="right_aruco_node",
            parameters=[{
                "camera_device": LaunchConfiguration("right_camera_device"),
                "camera_calibration_file": LaunchConfiguration("right_camera_calibration_file"),
            }],
        ),
    ])
