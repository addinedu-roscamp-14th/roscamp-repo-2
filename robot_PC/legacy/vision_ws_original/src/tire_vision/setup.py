from glob import glob
import os

from setuptools import find_packages, setup

package_name = "tire_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Vision-side tire detection TCP service.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_server = tire_vision.vision_server_node:main",
            "tire_marker_test = tire_vision.tire_marker_test:main",
        ],
    },
)
