from glob import glob
from setuptools import find_packages, setup
setup(
    name="arm_task_coordinator", version="0.1.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/arm_task_coordinator"]),
        ("share/arm_task_coordinator", ["package.xml"]),
        ("share/arm_task_coordinator/config", glob("config/*.yaml")),
        ("share/arm_task_coordinator/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="robot-team", maintainer_email="robot@example.com",
    description="Left-first replacement coordinator using TCP arm servers", license="MIT",
)
