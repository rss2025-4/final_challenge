from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np
import PIL.Image
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Vector3
from jax import Array
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
from scipy.ndimage import uniform_filter
from sensor_msgs.msg import Image
from tf2_ros import (
    Node,
)

from libracecar.utils import time_function

from ..homography import (
    Line,
    _ck_line,
    get_foot,
    homography_image,
    homography_mask,
    line_y_equals,
    matrix_xy_to_xy_img,
    point_coord,
    shift_line,
)
from .colors import color_counter, load_color_filter
from .detect_lines_sweep import ScoreCtx, update_line
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
            Odometry, self.cfg.odom_sub_topic, self.odom_callback, 1
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/vesc/high_level/input/nav_0", 1
        )
        self.line_pub = self.create_publisher(Vector3, "/tracker_line", 1)

        self.line_xy: Line = _ck_line(line_y_equals(-self.cfg.init_y))

        self.color_filter = jnp.array(load_color_filter())
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

    def get_target_line(self):
        return shift_line(self.line_xy, self.cfg.get_target_y())

    @time_function
    def image_callback(self, msg_ros: Image):
        self._counter += 1
        msg = ImageMsg.parse(msg_ros)

        self._image_callback(msg)

        # self.pure_pursuit(self.get_target_line())

        # if self._counter % 2 == 0:
        # self.matplotlib_plot(msg)
        self.publish_line()

        print()
        print()
        print()

    @time_function
    def _image_callback(self, msg: ImageMsg):

        weights = process_image(msg.image, self.color_filter)

        self.line_xy, res = update_line(
            ScoreCtx(
                weights=jnp.array(weights),
                weights_mask=jnp.array(homography_mask((msg.image.height, msg.image.width))),
                homography=jnp.array(matrix_xy_to_xy_img()),
            ),
            self.line_xy,
            jnp.array(self.cfg.shifts),
        )
        print("res", res)
        jax.block_until_ready(self.line_xy)

        # print("image cb: took", t.update())

        # target_line = shift_line(self.line_xy, self.cfg.get_target_y())

        # if self._counter % 10 != 0:
        #     return

    def publish_line(self):
        x, y, z = jnp.array(self.line_xy).tolist()
        msg = Vector3()
        msg.x = x
        msg.y = y
        msg.z = z
        self.line_pub.publish(msg)


def process_image(image: np.ndarray | PIL.Image.Image, color_filter: Array) -> np.ndarray:

    image = color_counter.apply_filter(color_filter, image)

    image = uniform_filter(image.astype(np.float32), size=3, mode="constant", cval=0.0) > 1e-8

    # print("homography_image", image)
    image = homography_image(image.astype(np.float32))
    # print("result", image)

    image = uniform_filter(image.astype(np.float32), size=11, mode="constant", cval=0.0)

    return image
