import os
import glob
from setuptools import find_packages, setup

package_name = 'shrinkray_heist'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # data_files=[
    #     ('share/ament_index/resource_index/packages',
    #         ['resource/' + package_name]),
    #     ('share/' + package_name, ['package.xml']),
    # ],
    include_package_data=True,  # This is for the package_data files
    package_data={
        package_name: [f"**/*{ext}" for ext in [".png", ".yaml"]],
    },
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/shrinkray_heist/launch/sim", glob.glob(os.path.join("launch", "sim", "*launch.*"))),
        ("share/shrinkray_heist/launch/real", glob.glob(os.path.join("launch", "real", "*launch.*"))),
        ("share/shrinkray_heist/launch/debug", glob.glob(os.path.join("launch", "debug", "*launch.*"))),
        (os.path.join("share", package_name, "config", "sim"), glob.glob("config/sim/*.yaml")),
        (os.path.join("share", package_name, "config", "real"), glob.glob("config/real/*.yaml")),
        (os.path.join("share", package_name, "config", "debug"), glob.glob("config/debug/*.yaml")),
        # ("share/shrinkray_heist/example_trajectories", glob.glob(os.path.join("example_trajectories", "*.traj"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adelene',
    maintainer_email='adelene@mit.edu',
    description='Shrinkray Heist ROS2 package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "states_node = shrinkray_heist.states:main",
            "trajectory_planner = shrinkray_heist.trajectory_planner:main",
            "trajectory_follower = shrinkray_heist.trajectory_follower:main",
            "detector = shrinkray_heist.model.detector_node:main",
        ],
    },
)
