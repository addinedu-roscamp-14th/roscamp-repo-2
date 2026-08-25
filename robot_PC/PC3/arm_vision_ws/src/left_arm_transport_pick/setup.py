from glob import glob
import os
from setuptools import find_packages, setup

package_name = "left_arm_transport_pick"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "calibration", "left"), glob("config/calibration/left/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Safe, independent left-arm ArUco transport tire picker",
    entry_points={
        "console_scripts": [
            "left_arm_transport_pick = left_arm_transport_pick.node:main",
        ],
    },
)
