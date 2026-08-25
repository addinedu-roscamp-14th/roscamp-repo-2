from setuptools import find_packages, setup
setup(
    name="arm_interfaces", version="0.1.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/arm_interfaces"]),
        ("share/arm_interfaces", ["package.xml"]),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="robot-team", maintainer_email="robot@example.com",
    description="Shared Python interfaces for arm vision coordination", license="MIT",
)
