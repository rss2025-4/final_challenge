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
from typing import Union

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import AxLine

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
def get_homography_matrix():
    np_pts_ground = np.array(PTS_GROUND_PLANE)
    np_pts_ground = np_pts_ground * METERS_PER_INCH
    np_pts_ground = np_pts_ground[:, np.newaxis, :]

    np_pts_image = np.array(PTS_IMAGE_PLANE)
    np_pts_image = np_pts_image * 1.0
    np_pts_image = np_pts_image[:, np.newaxis, :]

    ans, err = cv2.findHomography(np_pts_image, np_pts_ground)
    logger.warning(f"err {err}")
    return np.array(ans)


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
        logger.warn(f"getting coordinate of far-away pont {x}")
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
    return _ck_point(get_homography_matrix() @ _ck_point(point))


def xy_to_uv_line(line: Line) -> Line:
    # <xy, line> == 0
    # <=>
    # <M @ uv, line> == 0
    # <=>
    # <uv, M^T @ line> == 0
    return _ck_line(get_homography_matrix().T @ _ck_line(line))


def uv_to_xy_line(line: Line) -> Line:
    # <uv, line> == 0
    # <=>
    # <M^-1 @ uv, line> == 0
    # <=>
    # <uv, M^-1^T @ line> == 0
    return _ck_line(np.linalg.inv(get_homography_matrix()).T @ _ck_line(line))


def line_intersect(l1: Line, l2: Line) -> Point:
    return _ck_point(np.cross(_ck_line(l1), _ck_line(l2)))


def line_through_points(l1: Point, l2: Point) -> Line:
    return _ck_line(np.cross(_ck_point(l1), _ck_point(l2)))


def line_from_slope_intersect(slope: float, intercept: float) -> Line:
    # mx -y + b
    return _ck_line((slope, -1.0, intercept))


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
    l = _ck_line(l)
    l = l / np.linalg.norm(l)

    if abs(l[0]) <= 1e-3:
        # approx y=* line
        p1 = point_coord(line_intersect(l, (1.0, 0.0, 0.0)))  # x == 0
        p2 = point_coord(line_intersect(l, (1.0, 0.0, -1.0)))  # x == 1
    else:
        p1 = point_coord(line_intersect(l, (0.0, 1.0, 0.0)))  # y == 0
        p2 = point_coord(line_intersect(l, (0.0, 1.0, -1.0)))  # y == 1

    # return (p1[1], p1[0]), (p2[1], p2[0])
    return p1, p2


def plot_line(ax: Axes, l: Line) -> AxLine:
    p1, p2 = _get_2_points(l)
    return ax.axline(xy1=p1, xy2=p2)


def update_plot_line(plot: AxLine, l: Line):
    p1, p2 = _get_2_points(l)
    plot.set_xy1(p1)
    plot.set_xy2(p2)


def setup_plot(ax: Axes):
    ax.set_xlim(-1, 15)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")

    plot_line(ax, uv_to_xy_line(_image_left))
    plot_line(ax, uv_to_xy_line(_image_right))
    plot_line(ax, uv_to_xy_line(_image_bottom))
