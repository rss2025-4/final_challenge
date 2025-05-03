from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Vector3
from jax import Array
from jax import numpy as jnp
from jax import vmap
from nav_msgs.msg import Odometry
from rclpy import Context
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import (
    Node,
)

from libracecar.utils import jit, time_function

from ..homography import (
    ImagPlot,
    Line,
    LinePlot,
    LinePlotXY,
    setup_xy_plot,
    shift_line,
    xy_plot_top_to_uv_line,
    xy_to_uv_line,
)
from .colors import load_color_filter
from .ros import ImageMsg
from .tracker_node import process_image


@dataclass
class PlotConfig:
    #: list of parellel lines, in meters
    shifts: list[float]


class PlotNode(Node):
    def __init__(self, *, cfg: PlotConfig, context: Context | None = None):
        super().__init__(type(self).__qualname__, context=context)

        self.cfg = cfg

        self.line_sub = self.create_subscription(
            Vector3,
            "/tracker_line",
            self.line_callback,
            5,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            5,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/vesc/odom",
            self.odom_callback,
            5,
        )

        self.draw_timer = self.create_timer(1 / 15, self.draw_callback)

        self.color_filter = jnp.array(load_color_filter())

        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        cmap = plt.get_cmap("turbo")

        shifts_len = len(cfg.shifts)

        self.image_plot = ImagPlot(self.ax1)
        self.lines_plot = [
            LinePlot(self.ax1, color=cmap(i / shifts_len)) for i, _ in enumerate(cfg.shifts)
        ]

        LinePlot(self.ax1).set_line(xy_plot_top_to_uv_line())

        setup_xy_plot(self.ax2)
        self.image_plot_xy = ImagPlot(self.ax2, cmap="viridis", vmin=0, vmax=2.0)
        self.lines_plot_xy = [
            LinePlotXY(
                self.ax2,
                color=cmap(i / shifts_len),
                # linewidth=6,
                linewidth=2,
            )
            for i, _ in enumerate(cfg.shifts)
        ]

    @time_function
    def draw_callback(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def odom_callback(self, msg: Odometry):
        pass

    def image_callback(self, msg_ros: Image):
        print("plot node: image_callback")
        msg = ImageMsg.parse(msg_ros)

        self.image_plot.set_imag(msg.image)

        # color_mask = color_counter.apply_filter(self.color_filter, msg.image)

        # self.image_plot.set_imag(msg.image)

        image_processed = process_image(np.array(msg.image), self.color_filter)

        self.image_plot_xy.set_imag(image_processed)

    @time_function
    def line_callback(self, msg: Vector3):

        line_xy = (float(msg.x), float(msg.y), float(msg.z))

        xy_lines, uv_lines = _get_xy_uv_lines_for_plt(
            line_xy,
            jnp.array(self.cfg.shifts),
        )

        for p, l in zip(self.lines_plot, np.array(uv_lines)):
            p.set_line(l)

        for p, l in zip(self.lines_plot_xy, np.array(xy_lines)):
            p.set_line(l)


@jit
def _get_xy_uv_lines_for_plt(line_xy: Line, shifts: Array):
    def inner(s: Array):
        assert s.shape == ()
        ans_xy = shift_line(line_xy, s)
        ans_uv = xy_to_uv_line(ans_xy)
        return ans_xy, ans_uv

    return vmap(inner)(shifts)
