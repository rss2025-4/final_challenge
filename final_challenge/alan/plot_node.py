from __future__ import annotations

from dataclasses import dataclass

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from PIL import Image
from rclpy import Context
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.ndimage import uniform_filter
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from tf2_ros import (
    Node,
)

from libracecar.ros_utils import time_msg_to_float
from libracecar.utils import time_function

from ..homography import (
    ImagPlot,
    LinePlot,
    LinePlotXY,
    homography_image_rev,
    setup_xy_plot,
    shift_line,
    xy_plot_top_to_uv_line,
)
from .colors import load_color_filter
from .image import color_image, xy_line_to_xyplot_image
from .ros import ImageMsg
from .tracker import process_image
from .utils import cast_unchecked


@dataclass
class PlotConfig:
    #: list of parellel lines, in meters
    shifts: list[float]

    log_topic: str = "/tracker_log"


class PlotNode(Node):
    def __init__(self, *, cfg: PlotConfig, context: Context | None = None):
        super().__init__(type(self).__qualname__, context=context)

        self.cfg = cfg

        self.log_sub = self.create_subscription(
            String,
            self.cfg.log_topic,
            self.log_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        self.image_sub = self.create_subscription(
            RosImage,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        # self.draw_timer = self.create_timer(1 / 15, self.draw_callback)

        self.color_filter = jnp.array(load_color_filter())

        self.pending_logs: list[String] = []
        self.pending_images: list[ImageMsg] = []

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

        self._counter = 0

    # @time_function
    # def draw_callback(self):
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()

    def image_callback(self, msg_ros: RosImage):
        msg = ImageMsg.parse(msg_ros)
        self.pending_images.append(msg)

    def log_callback(self, msg: String):
        print("<<<<<<<<<<<<<<<<<\nlog_callback!!!")
        self.pending_logs.append(msg)

        if len(self.pending_logs) > 20:
            self.log_callback_(self.pending_logs.pop(0))

            print(">>>>>>>>>>>>>>>>>")
            print()
            print()

    def get_image_for_time(self, t: float) -> Image.Image | None:
        while len(self.pending_images) > 0:
            image_time = self.pending_images[0].time
            # print("get_image_for_time", image_time, t)
            if abs(image_time - t) <= 1e-4:
                return self.pending_images.pop(0).image
            elif image_time < t:
                print("throwing out image", image_time, t)
                _ = self.pending_images.pop(0)
                continue
            else:
                return None
        return None

    # @time_function
    def log_callback_controller(self, data):
        print("drive:", data["data"]["control_ang"])

    # @time_function
    def log_callback_(self, msg: String):

        data = cast_unchecked[dict]()(jsonpickle.decode(msg.data))

        if data["tag"] == "image":
            self.log_callback_image(data)
        elif data["tag"] == "controller":
            self.log_callback_controller(data)
        else:
            assert False, data["tag"]

    @time_function
    def log_callback_image(self, data: dict):

        # print("data", data)

        pt = data["python_time"]
        print("ros_clock", data["ros_clock"] * 1e-9 - pt)
        print("controller_time", data["controller_time"] - pt)
        print("_pending_odoms_times", [x - pt for x in data["_pending_odoms_times"]])

        image = self.get_image_for_time(time_msg_to_float(data["controller_time"]))
        if image is None:
            return

        self._counter += 1
        if self.cfg.log_topic == "/tracker_log":
            if self._counter % 2 == 0:
                return

        # print("got image")

        # line_xy = data["line_xy"]
        line_xy = data["forecast_line_xy"][1]
        assert isinstance(line_xy, tuple)

        xy_image = np.array(xy_line_to_xyplot_image(line_xy, jnp.array(self.cfg.shifts)))

        xy_image = (
            uniform_filter(xy_image.astype(np.float32), size=7, mode="constant", cval=0.0) > 1e-6
        )
        # self.image_plot_xy.set_imag(xy_image)

        image_processed = process_image(np.array(image), self.color_filter)
        self.image_plot_xy.set_imag(image_processed)

        overlay = homography_image_rev(np.array(color_image(xy_image)))
        # composite.paste(Image.fromarray(overlay))

        # color_image(xy_image)

        self.image_plot.set_imag(Image.alpha_composite(image, Image.fromarray(overlay)))

        # xy_image

        # draw_line

        # for p, l in zip(self.lines_plot, np.array(uv_lines)):
        #     p.set_line(l)

        for p, s in zip(self.lines_plot_xy, self.cfg.shifts):
            p.set_line(shift_line(line_xy, s))

        plt.tight_layout()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
