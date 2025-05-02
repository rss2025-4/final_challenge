#!/usr/bin/env python

import os
import subprocess

import better_exceptions
import jax
import numpy as np
import rclpy

from final_challenge.alan.tracker_node import TrackerConfig, TrackerNode
from libracecar.sandbox import isolate
from libracecar.test_utils import proc_manager

jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")

np.set_printoptions(precision=5, suppress=True)
# jax.config.update("jax_enable_x64", True)


def main():
    tracker_cfg = TrackerConfig(
        init_y=0.0,
        shifts=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    )
    rclpy.init()
    pc = TrackerNode(cfg=tracker_cfg)
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
