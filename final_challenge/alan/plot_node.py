from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Vector3
from jax import Array
from jax import numpy as jnp
from jax import vmap
from jax.typing import ArrayLike
from nav_msgs.msg import Odometry
from PIL import Image
from rclpy import Context
from rclpy.node import Node
from scipy.ndimage import uniform_filter
from sensor_msgs.msg import Image as RosImage
from tf2_ros import (
    Node,
)

from final_challenge.alan.image import draw_lines
from final_challenge.alan.tracker import process_image
from libracecar.batched import batched
from libracecar.utils import jit, time_function, tree_select

from ..homography import (
    ImagPlot,
    Line,
    LinePlot,
    LinePlotXY,
    homography_image_rev,
    homography_line,
    homography_mask,
    matrix_xy_to_xy_img,
    setup_xy_plot,
    shift_line,
    xy_plot_top_to_uv_line,
    xy_to_uv_line,
)
from .colors import load_color_filter
from .image import color_image, xy_line_to_xyplot_image
from .ros import ImageMsg


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
            RosImage,
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

        self.last_image: Image.Image = Image.fromarray(np.zeros((360, 640, 4), dtype=np.uint8))

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

    def image_callback(self, msg_ros: RosImage):
        # print("plot node: image_callback")
        msg = ImageMsg.parse(msg_ros)
        self.last_image = msg.image

        # self.image_plot.set_imag(msg.image)

        # color_mask = color_counter.apply_filter(self.color_filter, msg.image)

        # self.image_plot.set_imag(msg.image)

        # image_processed = process_image(np.array(msg.image), self.color_filter)

        # self.image_plot_xy.set_imag(image_processed)

    @time_function
    def line_callback(self, msg: Vector3):

        line_xy = (float(msg.x), float(msg.y), float(msg.z))

        xy_image = np.array(xy_line_to_xyplot_image(line_xy, jnp.array(self.cfg.shifts)))

        xy_image = (
            uniform_filter(xy_image.astype(np.float32), size=7, mode="constant", cval=0.0) > 1e-6
        )
        # self.image_plot_xy.set_imag(xy_image)

        image_processed = process_image(np.array(self.last_image), self.color_filter)
        self.image_plot_xy.set_imag(image_processed)

        overlay = homography_image_rev(np.array(color_image(xy_image)))
        # composite.paste(Image.fromarray(overlay))

        # color_image(xy_image)

        self.image_plot.set_imag(Image.alpha_composite(self.last_image, Image.fromarray(overlay)))

        # xy_image

        # draw_line

        # for p, l in zip(self.lines_plot, np.array(uv_lines)):
        #     p.set_line(l)

        # for p, l in zip(self.lines_plot_xy, np.array(plot_xy_lines)):
        #     p.set_line(l)
