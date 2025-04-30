import functools

import cv2
import numpy as np

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
def _get_homography_matrix():
    np_pts_ground = np.array(PTS_GROUND_PLANE)
    np_pts_ground = np_pts_ground * METERS_PER_INCH
    np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

    np_pts_image = np.array(PTS_IMAGE_PLANE)
    np_pts_image = np_pts_image * 1.0
    np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

    ans, err = cv2.findHomography(np.array(np_pts_image), np.array(np_pts_ground))
    return ans


def transformUvToXy(u, v):
    """
    u and v are pixel coordinates.
    The top left pixel is the origin, u axis increases to right, and v axis
    increases down.

    Returns a normal non-np 1x2 matrix of xy displacement vector from the
    camera to the point on the ground plane.
    Camera points along positive x axis and y axis increases to the left of
    the camera.

    Units are in meters.
    """
    homogeneous_point = np.array([[u], [v], [1]])
    xy = np.dot(_get_homography_matrix(), homogeneous_point)
    scaling_factor = 1.0 / xy[2, 0]
    homogeneous_xy = xy * scaling_factor
    x = homogeneous_xy[0, 0]
    y = homogeneous_xy[1, 0]
    return x, y
