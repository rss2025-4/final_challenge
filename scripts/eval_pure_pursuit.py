import json
import time
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from jax import Array, numpy as jnp
from numpy.typing import ArrayLike
from PIL import Image

from final_challenge.alan import FrameData
from final_challenge.alan.rosbag import get_images, get_messages
from final_challenge.alan.utils import cache, cast_unchecked_
from final_challenge.homography import (
    ImagPlot,
    Line,
    LinePlot,
    angle_bisector,
    ck_line,
    ck_point,
    homography_line,
    line_direction,
    line_from_slope_intersect,
    line_intersect,
    line_through_points,
    line_x_equals,
    line_y_equals,
    matrix_rot,
    matrix_trans,
    matrix_uv_to_xy,
    point_coord,
    shift_line,
    xy_to_uv_line,
)
from libracecar.batched import batched
from libracecar.utils import debug_print, jit

see_meters = 4


@cache
def matrix_xy_to_xy_img_plot_eval():
    return np.array(
        [
            # u = -y + 2
            # v = -x + 2
            [0, -1, 400],
            [-1, 0, 400],
            [0, 0, 1],
        ]
    ) @ np.diag([400 / see_meters, 400 / see_meters, 1])


def homography_image_for_eval(
    image: Image.Image | np.ndarray, additional: np.ndarray | None = None
) -> np.ndarray:
    ans = matrix_uv_to_xy()
    if additional is not None:
        ans = additional @ ans

    return cv2.warpPerspective(
        np.array(image),
        cast_unchecked_(matrix_xy_to_xy_img_plot_eval() @ ans),
        (800, 400),
    )


@jit
def fit_line(image: ArrayLike) -> Line:
    image = jnp.array(image)
    vs, us, ws = (
        batched.create(image, image.shape).enumerate(lambda w, v, u: (v, u, w)).reshape(-1).uf
    )
    ans = jnp.polyfit(vs.astype(np.float32), us.astype(np.float32), 1, w=ws)
    assert isinstance(ans, Array)
    assert ans.shape == (2,)
    a, b, c = jnp.array(line_from_slope_intersect(ans[0], ans[1]))
    return (b, a, c)


def viz_data():
    bagpath = Path("/home/alan/6.4200/rosbags_5_3/out_bag3")
    annotated = Path(
        "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_5_3_labeled_v1/bag3"
    )

    msgs = get_messages(bagpath)

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot = ImagPlot(ax1)
    line1 = LinePlot(ax1)
    line2 = LinePlot(ax1)

    plot2 = ImagPlot(ax2, xlim=(200, 600))
    plot2_line = [LinePlot(ax2, linewidth=5) for _ in range(4)]

    prev_stamp = msgs[0].image_time
    start_t = time.time()

    for i, msg in enumerate(msgs):
        frame = FrameData.load(annotated, msg.image_idx)

        plot.set_imag(msg.image.image)
        line1.set_line(xy_to_uv_line(shift_line(msg.line, -1)))
        line2.set_line(xy_to_uv_line(shift_line(msg.line, -2)))

        l1 = fit_line(homography_image_for_eval(frame.out_left_bool.astype(np.float32)))
        l1 = homography_line(np.linalg.inv(matrix_xy_to_xy_img_plot_eval()), l1)

        l2 = fit_line(homography_image_for_eval(frame.out_right_bool.astype(np.float32)))
        l2 = homography_line(np.linalg.inv(matrix_xy_to_xy_img_plot_eval()), l2)

        r1 = point_coord(line_intersect(l1, line_x_equals(0)))
        r2 = point_coord(line_intersect(l2, line_x_equals(0)))

        p1 = point_coord(line_intersect(l1, line_x_equals(1)))
        p2 = point_coord(line_intersect(l2, line_x_equals(1)))

        intersect = line_intersect(l1, l2)

        ix, iy, iz = ck_point(intersect)
        # iz /= 5
        iz = 0
        intersect_to = jnp.array((ix, iy, iz))

        l1_to = line_through_points(r1, intersect_to)
        l2_to = line_through_points(r2, intersect_to)

        p1_to = point_coord(line_intersect(l1_to, line_through_points(p1, (0, 0))))
        p2_to = point_coord(line_intersect(l2_to, line_through_points(p2, (0, 0))))

        print("p1, p2", p1, p2)
        print("p1_to, p2_to", p1_to, p2_to)

        H, err = cv2.findHomography(
            np.array([(0, 0), (0, 1), (0, -1), (0, 10), (0, -10), p1, p2]),
            np.array([(0, 0), (0, 1), (0, -1), (0, 10), (0, -10), p1_to, p2_to]),
        )
        print("homography", H, err)

        lmid_to = homography_line(H, angle_bisector(l1, l2))

        _, shift_y = point_coord(line_intersect(lmid_to, line_x_equals(0)))
        shift_angx, shift_angy = line_direction(lmid_to)
        H = np.array(
            #
            matrix_rot(-jnp.arctan2(shift_angy, shift_angx))
            @ matrix_trans(dy=-shift_y)
            @ H
        )

        plot2.set_imag(homography_image_for_eval(frame.in_img, H))

        for l, s in zip(plot2_line, (0, -1, -2, -3)):
            l.set_line(
                homography_line(matrix_xy_to_xy_img_plot_eval() @ H, shift_line(msg.line, s))
            )

        time.sleep(max(0, (msg.image_time - prev_stamp) - (time.time() - start_t)) * 2)
        prev_stamp = msg.image_time
        start_t = time.time()

        fig.canvas.draw()
        fig.canvas.flush_events()
