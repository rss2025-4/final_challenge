from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import tf2_ros
import tqdm
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovariance
from jax import Array, lax
from jax import numpy as jnp
from jax.typing import ArrayLike
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy import Context
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from scipy.ndimage import uniform_filter
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import (
    Duration,
    Node,
    PoseWithCovarianceStamped,
    TransformBroadcaster,
    TransformStamped,
    rclpy,
)
from visualization_msgs.msg import Marker

from libracecar.utils import timer

from ..homography import (
    ImagPlot,
    Line,
    LinePlot,
    LinePlotXY,
    _ck_line,
    get_foot,
    point_coord,
    setup_xy_plot,
    shift_line,
    xy_to_uv_line,
)
from .colors import color_counter, load_color_filter
from .detect_lines_sweep import update_line
from .ros import ImageMsg
from .utils import check


@dataclass
class TrackerConfig:
    #: list of parellel lines, in meters
    shifts: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])

    #: initial y location; currently assuming heading exactly +x
    init_y: float = 0.5

    target_y: float | None = None

    base_frame: str = "base_link"

    odom_sub_topic: str = "/vesc/odom"

    visualization_topic: str = "/visualization"

    time_overwrite: bool = False
    invert_odom: bool = True

    def get_target_y(self) -> float:
        if self.target_y is None:
            return self.init_y
        return self.target_y


class TrackerNode(Node):
    def __init__(self, *, cfg: TrackerConfig, context: Context | None = None):
        super().__init__(type(self).__qualname__, context=context)

        self.cfg = cfg

        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        self.odom_sub = self.create_subscription(
            Odometry, self.cfg.odom_sub_topic, self.odom_callback, 1
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/vesc/high_level/input/nav_0", 1
        )

        self.line_xy = _ck_line((0.0, 1.0, -self.cfg.init_y))

        self.color_filter = jnp.array(load_color_filter())

        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        cmap = plt.get_cmap("turbo")
        x = cmap(0)

        shifts_len = len(cfg.shifts)

        self.image_plot = ImagPlot(self.ax1)
        self.lines_plot = [
            LinePlot(self.ax1, color=cmap(i / shifts_len))
            for i, _ in enumerate(cfg.shifts)
        ]

        setup_xy_plot(self.ax2)
        self.lines_plot_xy = [
            LinePlotXY(self.ax2, color=cmap(i / shifts_len), linewidth=6)
            for i, _ in enumerate(cfg.shifts)
        ]

        self._counter = 0

    def odom_callback(self, msg: Odometry):
        check(msg.child_frame_id, self.cfg.base_frame)

        assert msg.twist.twist.linear.y == 0.0
        if self.cfg.invert_odom:
            msg.twist.twist.linear.x = -msg.twist.twist.linear.x
            msg.twist.twist.angular.z = -msg.twist.twist.angular.z

        # print("msg.header.stamp", msg.header.stamp)

        if self.cfg.time_overwrite:
            msg.header.stamp = self.get_clock().now().to_msg()

        # controller.odom_callback(msg)

        # print("pose:", controller.get_pose())
        # print("particles:", controller.get_particles())

        # self.do_publish()
        # print()

    def pure_pursuit(self, target_line: Line):
        x, y = np.array(point_coord(get_foot((0, 0), target_line)))

        print("foot", x, y)

        dist = np.linalg.norm([x, y])

        lookahead = 10

        forward_dist = lookahead**2 - dist**2
        if forward_dist < 0:
            print("forward_dist negative", forward_dist)
            return
        forward_dist = np.sqrt(forward_dist)

        forward_dir = np.array([-y, x])
        forward_dir = forward_dir / np.linalg.norm(forward_dir)
        if forward_dir[0] < 0:
            forward_dir = -forward_dir

        forward_point = np.array([x, y]) + forward_dist * forward_dir
        print("forward_point", forward_point)

        drive_cmd = AckermannDriveStamped()

        drive = AckermannDrive()
        drive.steering_angle = forward_point[1] / forward_point[0] / 4 - 0.035

        # drive.steering_angle = -0.04
        drive.speed = 3.0
        # drive.speed = 0.5
        # drive.steering_angle = 0.0
        # drive.speed = 0.0

        drive_cmd.drive = drive

        self.drive_pub.publish(drive_cmd)

    def image_callback(self, msg: Image):
        print("image_callback")
        self._counter += 1

        with timer.create() as t:
            image = ImageMsg.parse(msg)

            color_mask = color_counter.apply_filter(self.color_filter, image.image)
            color_mask = (
                uniform_filter(
                    color_mask.astype(np.float32), size=3, mode="constant", cval=0.0
                )
                > 1e-8
            )

            self.line_xy = update_line(
                color_mask, self.line_xy, jnp.array(self.cfg.shifts)
            )
            jax.block_until_ready(self.line_xy)

            print("image cb: took", t.update())

            target_line = shift_line(self.line_xy, self.cfg.get_target_y())

            self.pure_pursuit(target_line)

            # if self._counter % 10 != 0:
            #     return

            # self.image_plot.set_imag(image.image)

            # for s, l in zip(self.cfg.shifts, self.lines_plot):
            #     l.set_line(xy_to_uv_line(shift_line(self.line_xy, s)))

            # for s, l in zip(self.cfg.shifts, self.lines_plot_xy):
            #     l.set_line(shift_line(self.line_xy, s))

            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()

            # print("matplotlib: took", t.update())
