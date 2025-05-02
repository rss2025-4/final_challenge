#!/usr/bin/env python

import os
import subprocess

import rclpy

from libracecar.sandbox import isolate


@isolate
def main():
    import time
    from pathlib import Path

    import better_exceptions
    import jax
    import numpy as np

    from final_challenge.alan.tracker_node import TrackerConfig, TrackerNode
    from libracecar.test_utils import proc_manager

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    # jax.config.update("jax_platform_name", "cpu")

    np.set_printoptions(precision=5, suppress=True)
    # jax.config.update("jax_enable_x64", True)

    bag_dir = Path("/root/repos/rosbags_4_29/")
    bag = "bag3"

    procs = proc_manager.new()
    procs.spin_thread()

    tracker_cfg = TrackerConfig(
        init_y=0.5,
        shifts=[-3.0, -2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
    )

    bag_p = procs.popen(
        ["ros2", "bag", "play", bag],
        cwd=str(bag_dir),
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )

    # procs.ros_node_thread(
    #     lambda context: TrackerNode(context=context, cfg=tracker_cfg),
    # )

    rclpy.init()
    pc = TrackerNode(cfg=tracker_cfg)
    rclpy.spin(pc)
    rclpy.shutdown()

    time.sleep(10000)


if __name__ == "__main__":
    main()
