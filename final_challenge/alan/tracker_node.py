from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jsonpickle
import numpy as np
import PIL.Image
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Twist, Vector3
from jax import Array, numpy as jnp
from nav_msgs.msg import Odometry
from rclpy import Context
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.ndimage import uniform_filter
from sensor_msgs.msg import Image
from std_msgs.msg import String
from termcolor import colored
from tf2_ros import (
    Node,
)

from final_challenge.alan.tracker import update_with_image
from libracecar.ros_utils import time_msg_to_float
from libracecar.utils import time_function

from ..homography import (
    Line,
    ck_line,
    get_foot,
    homography_image,
    homography_line,
    homography_mask,
    line_y_equals,
    matrix_rot,
    matrix_trans,
    matrix_xy_to_xy_img,
    point_coord,
    shift_line,
)
from .colors import color_counter, load_color_filter
from .detect_lines_sweep import ScoreCtx, update_line
from .ros import ImageMsg


@dataclass
class TrackerConfig:
    #: list of parellel lines, in meters
    shifts: list[float] = field(default_factory=lambda: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    #: initial y location; currently assuming heading exactly +x
    init_y: float = 0.5

    target_y: float | None = None

    base_frame: str = "base_link"

    odom_sub_topic: str = "/vesc/odom"

    visualization_topic: str = "/visualization"

    time_overwrite: bool = False
    invert_odom: bool = True

    matplotlib: bool = False

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
            "/tracker_log",
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
            self.handle_twist(self._cur_twist, min(delta, 0.5))

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
    def handle_twist(self, twist: Twist, duration: float):

        assert twist.linear.y == 0.0
        if self.cfg.invert_odom:
            twist.linear.x = -twist.linear.x
            twist.angular.z = -twist.angular.z

        # print("twist.angular.z", twist.angular.z)

        # if twist.linear.x != 0:
        #     print("ratio!!", twist.angular.z / twist.linear.x)

        twist.angular.z += 0.16 * twist.linear.x

        translation = matrix_trans(-twist.linear.x * duration)
        rotation = matrix_rot(-twist.angular.z * duration)

        self.line_xy = homography_line(rotation @ translation, self.line_xy)

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
        drive.steering_angle = forward_point[1] / forward_point[0] / 6 - 0.035

        # drive.steering_angle = -0.04
        drive.speed = 4.0
        # drive.speed = 0.5
        # drive.steering_angle = 0.0
        # drive.speed = 0.0

        drive_cmd.drive = drive

        self.drive_pub.publish(drive_cmd)

    def get_target_line(self):
        return shift_line(self.line_xy, self.cfg.get_target_y())

    # @time_function
    def handle_image(self, msg_ros: Image):
        self._image_count += 1
        # if self._image_count % 20 != 0:
        #     self.publish_log("image")
        #     return

        msg = ImageMsg.parse(msg_ros)

        res = self._handle_image(msg)

        self.pure_pursuit(self.get_target_line())

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

        # print("image cb: took", t.update())

        # target_line = shift_line(self.line_xy, self.cfg.get_target_y())

        # if self._counter % 10 != 0:
        #     return

    def publish_log(self, tag: str, data: Any = None):
        data = {
            #
            "python_time": time.time(),
            "ros_clock": self.get_clock().now().nanoseconds,
            "controller_time": self._cur_time,  # a float produced by time_msg_to_float
            "_pending_odoms_times": [t for t, _ in self._pending_odoms],
            #
            "line_xy": tuple(np.array(self.line_xy).tolist()),
            #
            "tag": tag,
            "data": data,
        }

        msg = String()
        msg.data = jsonpickle.encode(data)
        self.log_pub.publish(msg)
