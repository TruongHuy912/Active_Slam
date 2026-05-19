from setuptools import setup

package_name = "bumperbot_active_slam"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", ["config/active_slam.yaml"]),
        ("share/" + package_name + "/launch", ["launch/active_slam.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Antonio Brandi",
    maintainer_email="antonio.brandi@outlook.it",
    description="Active SLAM frontier and entropy utilities for Bumper-Bot.",
    license="Apache 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "active_slam_node = bumperbot_active_slam.active_slam_node:main",
        ],
    },
)
