"""

in this module, in functions like :func:`uv_to_xy_line`, etc,

a "uv" coordinate is of images:

The top left pixel is the origin, u axis increases to right, and v axis
increases down.

"xy" is of the ground, in meters
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import AxLine
from PIL import Image

from .alan.utils import cache, cast_unchecked_

try:
    from jax import Array
    from jax import numpy as jnp

    Arr = Union[Array, np.ndarray]
    ArrLike = Union[Array, np.ndarray, float]
except ModuleNotFoundError:
    jnp = np
    if not TYPE_CHECKING:
        Arr = np.ndarray
        ArrLike = Union[np.ndarray, float]


logger = logging.getLogger(__name__)

######################################################
PTS_IMAGE_PLANE = [
    [639.0, 200.0],
    [336.0, 177.0],
    [515.0, 176.0],
    [88.0, 199.0],
]

######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left
PTS_GROUND_PLANE = [
    [50.25, -52],
    [99.25, 0],
    [99.25, -52],
    [56, 42],
]

######################################################

METERS_PER_INCH = 0.0254


@cache
def matrix_uv_to_xy() -> np.ndarray:
    np_pts_ground = np.array(PTS_GROUND_PLANE)
    np_pts_ground = np_pts_ground * METERS_PER_INCH
    np_pts_ground = np_pts_ground[:, np.newaxis, :]

    np_pts_image = np.array(PTS_IMAGE_PLANE)
    np_pts_image = np_pts_image * 1.0
    np_pts_image = np_pts_image[:, np.newaxis, :]

    ans, err = cv2.findHomography(np_pts_image, np_pts_ground)
    logger.warning(f"matrix_uv_to_xy: findHomography: err={err}")
    return np.array(ans)


@cache
def matrix_xy_to_uv() -> np.ndarray:
    return np.linalg.inv(matrix_uv_to_xy())


#: A parameter annotated with `Point` can take one of
#:     (1) (x, y)
#:     (2) (x, y, z), in projective coordinates.
#:         this is equivalent to the euclidean point (x/z, y/z).
#:
#: These can be python or numpy.
#:
#: functions returning Point mostly returns in project coordinates.
#: use :func:`point_coord` to convert to euclidean
Point = Union[Arr, tuple[ArrLike, ArrLike]]

#: (a, b, c), represents the line ax + by + c = 0
#:
#: can be python or numpy.
#:
#: construct a Line with  :func:`line_from_slope_intersect`, :func:`line_through_points`
#:
#: The "positive side" is points (x, y) such that ax + by + c >= 0
#:
#: The "direction" of the line is, direction == up <=> left is "positive side"
Line = Union[Arr, tuple[ArrLike, ArrLike, ArrLike]]


def _ck_line(x: Line) -> Arr:
    """validate a line"""
    x = jnp.array(x)
    assert isinstance(x, jnp.ndarray)
    assert x.shape == (3,)
    return x


def _ck_point(x: Point) -> Arr:
    """validate a point"""
    x = jnp.array(x)
    assert isinstance(x, jnp.ndarray)
    if x.shape == (2,):
        x = jnp.array([x[0], x[1], 1.0])
    assert x.shape == (3,)
    return x


def point_coord(x: Point) -> tuple[float, float]:
    """get the euclidean coordinate; cannot be at infinity"""
    ans = _ck_point(x)
    z = ans[2]
    if abs(z) <= 1e-10:
        logger.warning(f"getting coordinate of far-away pont {x}")
        z = 1e-8
    return float(ans[0] / z), float(ans[1] / z)


def line_direction(line: Line, normalize: bool = False) -> Arr:
    """vector of the direction of the line"""
    a, b, _c = _ck_line(line)
    ans = jnp.array([b, -a])

    if normalize:
        return ans / jnp.linalg.norm(ans)
    else:
        return ans


W: int = 640
H: int = 360

# u == 0 line
_image_left: Line = (1.0, 0.0, 0.0)
# u == 640 line
_image_right: Line = (1.0, 0.0, -W)
# v == 0 line
_image_top: Line = (0.0, 1.0, 0.0)
# v == 360 line
_image_bottom: Line = (0.0, 1.0, -H)


def homography_point(matrix: Arr, point: Point) -> Point:
    return _ck_point(matrix @ _ck_point(point))


def xy_to_uv_point(point: Point) -> Point:
    return homography_point(matrix_uv_to_xy(), point)


def homography_line(matrix: Arr, line: Line) -> Line:
    # <prev, line> == 0
    # <=>
    # <M^-1 @ ans, line> == 0
    # <=>
    # <uv, M^T^-1 @ line> == 0
    return _ck_line(jnp.linalg.inv(matrix.T) @ _ck_line(line))


def xy_to_uv_line(line: Line) -> Line:
    return homography_line(matrix_xy_to_uv(), line)


def uv_to_xy_line(line: Line) -> Line:
    return homography_line(matrix_uv_to_xy(), line)


def line_intersect(l1: Line, l2: Line) -> Point:
    return _ck_point(jnp.cross(_ck_line(l1), _ck_line(l2)))


def line_x_equals(x0: ArrLike) -> Line:
    # x == x0 line, pointing at +y direction
    return _ck_line((-1.0, 0.0, x0))


def line_y_equals(y0: ArrLike) -> Line:
    # y == y0 line, pointing at +x direction
    return _ck_line((0.0, 1.0, -y0))


def line_through_points(l1: Point, l2: Point) -> Line:
    return _ck_line(jnp.cross(_ck_point(l1), _ck_point(l2)))


def line_from_slope_intersect(slope: float, intercept: float) -> Line:
    # mx -y + b
    return _ck_line((slope, -1.0, intercept))


def matrix_rot(ang_rad: ArrLike) -> Arr:
    sin = jnp.sin(ang_rad)
    cos = jnp.cos(ang_rad)

    return jnp.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def matrix_trans(dx: ArrLike = 0.0, dy: ArrLike = 0.0) -> Arr:
    return jnp.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ]
    )


def shift_line(l: Line, d: ArrLike) -> Line:
    """shift line to the positive side by scalar d"""
    l = _ck_line(l)
    return _ck_line(l + jnp.array([0.0, 0.0, -jnp.linalg.norm(l[:2]) * d]))


# def angle_bisector(l1: Line, l2: Line) -> Line:
#     # UNTESTED!!
#     l1 = _ck_line(l1)
#     l2 = _ck_line(l2)

#     l2 = jnp.where(
#         jnp.dot(l1[:2], l2[:2]) < 0,
#         # flip if true
#         -l2,
#         l2,
#     )

#     r1 = jnp.linalg.norm(l1[:2])
#     r2 = jnp.linalg.norm(l2[:2])

#     return _ck_line(l1 * r2 + l2 * r1)


def get_foot(point: Point, line: Line) -> Point:
    x, y, z = _ck_point(point)
    a, b, c = _ck_line(line)

    ans = (
        b * b * x - a * b * y - a * c * z,
        a * a * y - a * b * x - b * c * z,
        a * a * z + b * b * z,
    )
    return _ck_point(jnp.array(ans))


@cache
def get_horizon(x_dist: float | None = None) -> int:
    """
    line of horizon in image, in pixels from the top

    x_dist: only look this far in meters

    returns:

    x_dist==None ==> 144 (4/30, Alan)

    x_dist==10 ==> 153 (4/30, Alan)
    """
    if x_dist is None:
        line = (0.0, 0.0, 1.0)
    else:
        line = (1.0, 0.0, -x_dist)

    uv_line = xy_to_uv_line(line)
    u1, _0 = point_coord(line_intersect(uv_line, _image_left))
    u2, _640 = point_coord(line_intersect(uv_line, _image_right))

    assert abs(u1 - u2) < 3
    return int((u1 + u2) / 2)


def _get_2_points(l: Line):
    """get two distinct points on a line"""
    l = _ck_line(l)
    l = l / jnp.linalg.norm(l)

    if abs(l[0]) <= 1e-3:
        # approx y=* line
        p1 = point_coord(line_intersect(l, (1.0, 0.0, 0.0)))  # x == 0
        p2 = point_coord(line_intersect(l, (1.0, 0.0, -1.0)))  # x == 1
    else:
        p1 = point_coord(line_intersect(l, (0.0, 1.0, 0.0)))  # y == 0
        p2 = point_coord(line_intersect(l, (0.0, 1.0, -1.0)))  # y == 1

    return p1, p2


class LinePlot:
    def __init__(self, ax: Axes, **kwargs):
        self.ax = ax
        self.line: Optional[AxLine] = None
        self.kwargs = kwargs

    def set_line(self, l: Line):
        p1, p2 = _get_2_points(l)
        if self.line is None:
            self.line = self.ax.axline(xy1=p1, xy2=p2, **self.kwargs)
        else:
            self.line.set_xy1(p1)
            self.line.set_xy2(p2)


see_meters = 2


@cache
def matrix_xy_to_xy_img():
    return (
        # 50 -> sees 200 / 50 meters
        np.array(
            [
                # u = -y + 2
                # v = -x + 2
                [0, -1, 200],
                [-1, 0, 200],
                [0, 0, 1],
            ]
        )
        @ np.diag([200 / see_meters, 200 / see_meters, 1])
    )


class LinePlotXY(LinePlot):
    def set_line(self, l: Line, **kwargs):
        super().set_line(homography_line(matrix_xy_to_xy_img(), l), **kwargs)


class ImagPlot:
    def __init__(self, ax: Axes, **kwargs):
        self.ax = ax
        self.image: Optional[AxesImage] = None
        self.kwargs = kwargs

    def set_imag(self, image: Image.Image | np.ndarray):
        image_ = np.array(image)
        if self.image is None:
            self.image = self.ax.imshow(image_, **self.kwargs)
            self.ax.set_xlim(0, image_.shape[1])
            self.ax.set_ylim(image_.shape[0], 0)
        else:
            self.image.set_data(image_)


@cache
def homography_mask(shape: tuple[int, int]) -> np.ndarray:
    ans = homography_image(np.ones(shape))
    return np.array(ans) > 1e-3


def homography_image(image: Image.Image | np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        np.array(image),
        cast_unchecked_(matrix_xy_to_xy_img() @ matrix_uv_to_xy()),
        (400, 200),
    )


class ImagPlotXY(ImagPlot):
    def __init__(self, ax: Axes):
        self.ax = ax
        self.image: Optional[AxesImage] = None

    def set_uv_imag(self, image: Image.Image | np.ndarray):
        super().set_imag(homography_image(image))


def setup_xy_plot(ax: Axes):
    ax.set_xlim(0, 400)
    ax.set_ylim(200, 0)
    ax.set_aspect("equal")

    LinePlotXY(ax).set_line(uv_to_xy_line(_image_left))
    LinePlotXY(ax).set_line(uv_to_xy_line(_image_right))
    LinePlotXY(ax).set_line(uv_to_xy_line(_image_bottom))


@cache
def xy_plot_top_to_uv_line():
    return homography_line(
        matrix_xy_to_uv() @ np.linalg.inv(matrix_xy_to_xy_img()), (0.0, 1.0, 0.0)
    )
