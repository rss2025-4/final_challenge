#!/usr/bin/env python


import jax
import numpy as np
import rclpy

from final_challenge.alan.plot_node import PlotConfig, PlotNode

jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_platform_name", "cpu")

np.set_printoptions(precision=5, suppress=True)
# jax.config.update("jax_enable_x64", True)


def main():
    cfg = PlotConfig(
        shifts=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    )
    rclpy.init()
    pc = PlotNode(cfg=cfg)
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
