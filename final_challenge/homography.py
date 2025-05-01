"""

in this module, in functions like :func:`uv_to_xy_line`, etc,

a "uv" coordinate is of images:

The top left pixel is the origin, u axis increases to right, and v axis
increases down.

"xy" is of the ground, in meters
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Union

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import AxLine
from PIL import Image
from typing_extensions import Never

from .alan.utils import unreachable

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


@functools.cache
def matrix_uv_to_xy():
    np_pts_ground = np.array(PTS_GROUND_PLANE)
    np_pts_ground = np_pts_ground * METERS_PER_INCH
    np_pts_ground = np_pts_ground[:, np.newaxis, :]

    np_pts_image = np.array(PTS_IMAGE_PLANE)
    np_pts_image = np_pts_image * 1.0
    np_pts_image = np_pts_image[:, np.newaxis, :]

    ans, err = cv2.findHomography(np_pts_image, np_pts_ground)
    logger.warning(f"matrix_uv_to_xy: findHomography: err={err}")
    return np.array(ans)


@functools.cache
def matrix_xy_to_uv():
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
Point = Union[np.ndarray, tuple[float, float]]

#: (a, b, c), represents the line ax + by + c = 0
#:
#: can be python or numpy.
#:
#: construct a Line with  :func:`line_from_slope_intersect`, :func:`line_through_points`
Line = Union[np.ndarray, tuple[float, float, float]]


def _ck_line(x: Line) -> np.ndarray:
    """validate a line"""
    x = np.array(x)
    assert isinstance(x, np.ndarray)
    assert x.shape == (3,)
    return x


def _ck_point(x: Point) -> np.ndarray:
    """validate a point"""
    x = np.array(x)
    assert isinstance(x, np.ndarray)
    if x.shape == (2,):
        x = np.array([x[0], x[1], 1.0])
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


def xy_to_uv_point(point: Point) -> Point:
    return _ck_point(matrix_uv_to_xy() @ _ck_point(point))


def homography_line(matrix: np.ndarray, line: Line) -> Line:
    # <prev, line> == 0
    # <=>
    # <M^-1 @ ans, line> == 0
    # <=>
    # <uv, M^T^-1 @ line> == 0
    return _ck_line(np.linalg.inv(matrix.T) @ _ck_line(line))


def xy_to_uv_line(line: Line) -> Line:
    return homography_line(matrix_xy_to_uv(), line)


def uv_to_xy_line(line: Line) -> Line:
    return homography_line(matrix_uv_to_xy(), line)


def line_intersect(l1: Line, l2: Line) -> Point:
    return _ck_point(np.cross(_ck_line(l1), _ck_line(l2)))


def line_through_points(l1: Point, l2: Point) -> Line:
    return _ck_line(np.cross(_ck_point(l1), _ck_point(l2)))


def line_from_slope_intersect(slope: float, intercept: float) -> Line:
    # mx -y + b
    return _ck_line((slope, -1.0, intercept))


def angle_bisector(l1: Line, l2: Line) -> Line:
    l1 = _ck_line(l1)
    l2 = _ck_line(l2)

    if np.dot(l1[:2], l2[:2]) < 0:
        l2 = -l2

    r1 = np.linalg.norm(l1[:2])
    r2 = np.linalg.norm(l2[:2])

    return _ck_line(l1 * r2 + l2 * r1)


@functools.cache
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
    l = l / np.linalg.norm(l)

    if abs(l[0]) <= 1e-3:
        # approx y=* line
        p1 = point_coord(line_intersect(l, (1.0, 0.0, 0.0)))  # x == 0
        p2 = point_coord(line_intersect(l, (1.0, 0.0, -1.0)))  # x == 1
    else:
        p1 = point_coord(line_intersect(l, (0.0, 1.0, 0.0)))  # y == 0
        p2 = point_coord(line_intersect(l, (0.0, 1.0, -1.0)))  # y == 1

    return p1, p2


class LinePlot:
    def __init__(self, ax: Axes):
        self.ax = ax
        self.line: Optional[AxLine] = None

    def set_line(self, l: Line):
        p1, p2 = _get_2_points(l)
        if self.line is None:
            self.line = self.ax.axline(xy1=p1, xy2=p2)
        else:
            self.line.set_xy1(p1)
            self.line.set_xy2(p2)


@functools.cache
def matrix_xy_to_xyplot():
    return (
        #
        np.diag([100, 100, 1])
        @ np.array(
            [
                [0, -1, 2],
                [-1, 0, 2],
                [0, 0, 1],
            ]
        )
    )


class LinePlotXY(LinePlot):
    def set_xy_line(self, l: Line):
        super().set_line(homography_line(matrix_xy_to_xyplot(), l))

    def set_line(self, l: Never):
        """you probably dont want to call this"""
        unreachable(l)


class ImagPlotXY:
    def __init__(self, ax: Axes):
        self.ax = ax
        self.image: Optional[AxesImage] = None

    def set_uv_imag(self, image: Image.Image):
        image_ = cv2.warpPerspective(
            np.array(image),
            matrix_xy_to_xyplot() @ matrix_uv_to_xy(),
            (400, 200),
        )
        if self.image is None:
            self.image = self.ax.imshow(image_)
        else:
            self.image.set_data(image_)


def setup_xy_plot(ax: Axes):
    ax.set_xlim(0, 400)
    ax.set_ylim(200, 0)
    ax.set_aspect("equal")

    LinePlotXY(ax).set_xy_line(uv_to_xy_line(_image_left))
    LinePlotXY(ax).set_xy_line(uv_to_xy_line(_image_right))
    LinePlotXY(ax).set_xy_line(uv_to_xy_line(_image_bottom))


@functools.cache
def xy_plot_top_to_uv_line():
    return homography_line(
        matrix_xy_to_uv() @ np.linalg.inv(matrix_xy_to_xyplot()), (0.0, 1.0, 0.0)
    )
