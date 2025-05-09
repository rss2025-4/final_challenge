#!/usr/bin/env python


import jax
import numpy as np
import rclpy

from final_challenge.alan.tracker_node import TrackerConfig, TrackerNode
from final_challenge.alan.plot_node import PlotConfig, PlotNode

jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")

np.set_printoptions(precision=5, suppress=True)
# jax.config.update("jax_enable_x64", True)


def main():
    cfg = PlotConfig(
        shifts=[x * TrackerConfig.LANE_WIDTH for x in range(3, -4, -1)],
    )
    rclpy.init()
    pc = PlotNode(cfg=cfg)
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
