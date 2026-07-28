from glob import glob
import os

from setuptools import find_packages, setup

package_name = "tire_arm_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "motions"), glob("motions/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="TCP ROS2 arm control package for JetCobot tire task",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
    "console_scripts": [
        "tcp_server_node = tire_arm_control.tcp_server_node:main",
        "pose_reader = tire_arm_control.pose_reader:main",
        "pose_recorder = tire_arm_control.pose_recorder:main",
    ],
},
)
