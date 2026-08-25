from glob import glob
import os
from setuptools import find_packages, setup
setup(
    name="arm_aruco_perception", version="0.1.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/arm_aruco_perception"]),
        ("share/arm_aruco_perception", ["package.xml"]),
        ("share/arm_aruco_perception/config", glob("config/*.yaml")),
        ("share/arm_aruco_perception/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="robot-team", maintainer_email="robot@example.com",
    description="Dual gripper-camera ArUco perception", license="MIT",
    entry_points={"console_scripts": [
        "left_aruco_node=arm_aruco_perception.left_aruco_node:main",
        "right_aruco_node=arm_aruco_perception.right_aruco_node:main",
    ]},
)
