from glob import glob
import os

from setuptools import find_packages, setup

package_name = "central_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        *(
            (
                os.path.join("share", package_name, os.path.dirname(model_file)),
                [model_file],
            )
            for model_file in glob("models/**/*", recursive=True)
            if os.path.isfile(model_file)
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Central control package with parking gate state machine.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "manual_central = central_control.manual_central_node:main",
            "auto_operation = central_control.auto_operation_node:main",
            "pinky_keyboard_trigger = "
            "central_control.pinky_keyboard_trigger:main",
        ],
    },
)
