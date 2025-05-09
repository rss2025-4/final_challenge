from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jsonpickle
import numpy as np
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Twist
from jax import numpy as jnp
from nav_msgs.msg import Odometry
from rclpy import Context
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from termcolor import colored
from tf2_ros import (
    Node,
)

from libracecar.ros_utils import time_msg_to_float
from libracecar.utils import time_function

from ..homography import (
    Line,
    ck_line,
    get_foot,
    homography_line,
    homography_point,
    line_direction,
    line_to_tuple,
    line_y_equals,
    matrix_rot,
    matrix_trans,
    point_coord,
    shift_line,
)
from .colors import load_color_filter
from .controller import cached_controller
from .ros import ImageMsg
from .tracker import update_with_image


@dataclass
class TrackerConfig:
    LANE_WIDTH: ClassVar[float] = 1.05

    #: list of parellel lines, in meters
    shifts: list[float] = field(
        default_factory=lambda: [x * TrackerConfig.LANE_WIDTH for x in range(3, -4, -1)],
    )

    #: initial y location; currently assuming heading exactly +x
    init_y: float = 0.0

    target_y: float | None = None

    odom_sub_topic: str = "/vesc/odom"

    log_topic: str = "/tracker_log"

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
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        self.odom_sub = self.create_subscription(
            Odometry, self.cfg.odom_sub_topic, self.odom_callback, 10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/vesc/high_level/input/nav_0", 1
        )
        self.log_pub = self.create_publisher(
            String,
            self.cfg.log_topic,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )

        ######################################################

        self._cur_twist = Twist()
        self._cur_time: float = -1.0
        self._pending_odoms: list[tuple[float, Odometry]] = []

        ######################################################

        self.line_xy: Line = ck_line(line_y_equals(-self.cfg.init_y))

        self.color_filter = jnp.array(load_color_filter())

        self._image_count = 0

        ######################################################

        self.controller_cache = cached_controller.load()

        # self.controller_cb = self.create_timer(1 / 50, self.controller_timer_callback)

    ######################################################
    # reorder messages we receieve that is out of order
    ######################################################

    def __advance_time_one(self, new_time: float, warn_tag: str) -> None:
        delta = new_time - self._cur_time
        try:
            if delta < -1e-8:
                print(
                    colored(
                        f"warning: ({warn_tag}) time went backwards by: {delta:.5f}",
                        "red",
                    )
                )
                return
            if delta < 1e-8:
                return

            if delta > 0.5:
                print(colored(f"warning: ({warn_tag}) {delta:.2f}s with no messages", "red"))

            # print(f"applying twist with duration {delta}")
            self.line_xy = self.handle_twist(
                self.line_xy,
                self._cur_twist,
                min(delta, 0.5),
            )

        finally:
            self._cur_time = new_time

    def __advance_time(self, new_time: float) -> None:

        while len(self._pending_odoms) > 0:
            if self._pending_odoms[0][0] > new_time:
                break

            odom_time, odom_msg = self._pending_odoms.pop(0)
            self.__advance_time_one(odom_time, "__advance_time_1")
            self._cur_twist = odom_msg.twist.twist

        self.__advance_time_one(new_time, "__advance_time_1")

    def odom_callback(self, msg: Odometry) -> None:
        # print("odom_callback!!", time_msg_to_float(msg.header.stamp))
        self._pending_odoms.append((time_msg_to_float(msg.header.stamp), msg))

        # self.__maybe_process_messages()

        if len(self._pending_odoms) > 50:
            _ = self._pending_odoms.pop(0)

    @time_function
    def image_callback(self, msg: Image) -> None:

        self.__advance_time(time_msg_to_float(msg.header.stamp))

        self.handle_image(msg)

    ######################################################

    # @time_function
    def handle_twist(self, prev_line: Line, twist: Twist, duration: float) -> Line:

        assert twist.linear.y == 0.0
        linear_x = twist.linear.x
        angular_z = twist.angular.z

        if self.cfg.invert_odom:
            linear_x = -linear_x
            angular_z = -angular_z

        # if twist.linear.x != 0:
        #     print("ratio!!", angular_z / linear_x)

        angular_z += 0.155 * linear_x

        translation = matrix_trans(-linear_x * duration)
        rotation = matrix_rot(-angular_z * duration)

        return homography_line(rotation @ translation, prev_line)

    def controller_timer_callback(self):
        self.controller()

    def get_forecast_delta(self) -> float | None:
        delta = time_msg_to_float(self.get_clock().now().to_msg()) - self._cur_time

        if delta > 1:
            # probably replaying a rosbag

            if len(self._pending_odoms) > 0:
                delta = self._pending_odoms[-1][0] - self._cur_time
            else:
                return None

        if delta > 0:
            return delta

        return None

    def forecast_line_xy(self) -> tuple[float | None, Line]:
        forecast_line_xy = self.line_xy

        delta = self.get_forecast_delta()
        # print("forecast(2):", delta)
        if delta is not None:
            forecast_line_xy = self.handle_twist(
                forecast_line_xy,
                self._cur_twist,
                min(delta, 0.5),
            )

        return delta, forecast_line_xy

    def controller(self):

        forecast_delta, forecast_line_xy = self.forecast_line_xy()

        target_line = shift_line(forecast_line_xy, self.cfg.get_target_y())

        lx, ly = np.array(line_direction(target_line))
        line_ang = np.arctan2(ly, lx)

        foot_rot = homography_point(
            matrix_rot(-line_ang), point_coord(get_foot((0, 0), target_line))
        )

        _x_zero, dist = np.array(point_coord(foot_rot))
        # print("x_zero", x_zero)

        control_ang = self.controller_cache.get(-dist, -line_ang)

        print("dist, line_ang", dist, line_ang)
        print("control_ang", control_ang)

        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()

        drive = AckermannDrive()

        drive.steering_angle = control_ang - 0.035
        # drive.steering_angle = -0.04

        drive.speed = min(4.0, 5.0 - 10 * abs(control_ang))

        # drive.speed = 0.5
        # drive.steering_angle = 0.0
        # drive.speed = 0.0

        drive_cmd.drive = drive

        self.drive_pub.publish(drive_cmd)

        self.publish_log(
            "controller",
            {
                "target_line": line_to_tuple(target_line),
                "forecast_line_used": (forecast_delta, line_to_tuple(forecast_line_xy)),
                "control_ang": control_ang,
            },
        )

    # @time_function
    def handle_image(self, msg_ros: Image):
        self._image_count += 1
        # if self._image_count % 10 != 0:
        #     self.publish_log("image")
        #     return

        msg = ImageMsg.parse(msg_ros)

        res = self._handle_image(msg)

        self.controller()

        self.publish_log("image", {"fit_scores": str(res)})

        # if self._counter % 2 == 0:
        # self.matplotlib_plot(msg)
        # self.publish_line()

        print()
        print()
        print()

    # @time_function
    def _handle_image(self, msg: ImageMsg):

        self.line_xy, res = update_with_image(
            self.line_xy, msg.image, self.color_filter, self.cfg.shifts
        )

        # print("res", res)
        jax.block_until_ready(self.line_xy)

        return res

    def publish_log(self, tag: str, data: Any = None):
        fd, fl = self.forecast_line_xy()
        data = {
            #
            "python_time": time.time(),
            "ros_clock": self.get_clock().now().nanoseconds,
            "controller_time": self._cur_time,  # a float produced by time_msg_to_float
            "_pending_odoms_times": [t for t, _ in self._pending_odoms],
            #
            "line_xy": line_to_tuple(self.line_xy),
            "forecast_line_xy": (fd, line_to_tuple(fl)),
            #
            "tag": tag,
            "data": data,
        }

        msg = String()
        msg.data = jsonpickle.encode(data)
        self.log_pub.publish(msg)
