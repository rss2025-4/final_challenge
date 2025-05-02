#!/usr/bin/env python


import jax
import numpy as np
import rclpy

from final_challenge.alan.tracker_node import TrackerConfig, TrackerNode

jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")

np.set_printoptions(precision=5, suppress=True)
# jax.config.update("jax_enable_x64", True)


def main():
    tracker_cfg = TrackerConfig(
        init_y=0.5,
        shifts=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    )
    rclpy.init()
    pc = TrackerNode(cfg=tracker_cfg)
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
