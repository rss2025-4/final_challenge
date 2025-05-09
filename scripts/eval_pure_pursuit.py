import math
import time
from pathlib import Path

import cv2
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, numpy as jnp
from numpy.typing import ArrayLike
from PIL import Image
from scipy.ndimage import uniform_filter

from final_challenge.alan import FrameData
from final_challenge.alan.image import color_image, xy_line_to_xyplot_image
from final_challenge.alan.rosbag import get_messages
from final_challenge.alan.utils import cache, cast_unchecked_
from final_challenge.homography import (
    ArrowPlot,
    ImagPlot,
    Line,
    LinePlot,
    Point,
    ck_line,
    ck_point,
    dual_l,
    homography_image_rev,
    homography_line,
    homography_point,
    line_from_slope_intersect,
    line_to_slope_intersect,
    line_x_equals,
    matrix_rot,
    matrix_trans,
    matrix_uv_to_xy,
    normalize,
    point_coord,
    shift_line,
)
from libracecar.batched import batched
from libracecar.utils import jit

np.set_printoptions(precision=5, suppress=True)
# jax.config.update("jax_enable_x64", True)

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


def homography_image2(image: Image.Image | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        np.array(image),
        cast_unchecked_(matrix),
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


# def testfn():
#     return solve_homography(batched.create(((1, 0), np.array((1, 0, 0)))))


@jit
def solve_homography(points: batched[tuple[Point, Point, ArrayLike]]) -> Array:

    def inner(h_: Array):
        h = h_.reshape(3, 3) + jnp.diag(jnp.ones(3))
        eqns = points.tuple_map(
            lambda x, y, w: jnp.array(w)
            * jnp.cross(h @ ck_point(normalize(x)), ck_point(normalize(y)))
        )
        ans = eqns.uf.reshape(-1)
        return ans, ans

    M, b = jax.jacfwd(inner, has_aux=True)(jnp.zeros(9))

    x, resid, rank, s = jnp.linalg.lstsq(M, -b, rcond=0.1)
    return x.reshape(3, 3) + jnp.diag(jnp.ones(3))


def main():
    bagpath = Path("/home/alan/6.4200/rosbags_5_3/out_bag3")
    annotated = Path(
        "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_5_3_labeled_v1/bag3_2"
    )

    msgs = get_messages(bagpath)

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot = ImagPlot(ax1)
    line1 = LinePlot(ax1)
    line2 = LinePlot(ax1)

    plot2 = ImagPlot(ax2, xlim=(200, 600), ylim=(450, 0))
    plot2_line = [LinePlot(ax2, linewidth=5, color=(0, 0, 1)) for _ in range(4)]
    plot2_line_truth = [LinePlot(ax2, linewidth=5, color="yellow", alpha=0.5) for _ in range(2)]

    arrow_plot = ArrowPlot(ax2, linewidth=10, head_width=20, length_includes_head=True)
    arrow_plot_drive = ArrowPlot(
        ax2, linewidth=5, head_width=10, length_includes_head=True, color="green"
    )

    prev_stamp = msgs[0].image_time
    start_t = time.time()

    for i, msg in enumerate(msgs):
        frame = FrameData.load(annotated, msg.image_idx)

        ######################################################

        xy_image = np.array(
            xy_line_to_xyplot_image(
                jnp.array(msg.line), jnp.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            )
        )
        xy_image = (
            uniform_filter(xy_image.astype(np.float32), size=7, mode="constant", cval=0.0) > 1e-6
        )
        overlay = homography_image_rev(np.array(color_image(xy_image, color=(0, 0, 255, 127))))

        plot.set_imag(
            Image.alpha_composite(msg.image.image, Image.fromarray(overlay))
        )  # line1.set_line(xy_to_uv_line(shift_line(msg.line, -1)))
        # line2.set_line(xy_to_uv_line(shift_line(msg.line, -2)))

        ######################################################

        matrix_for_fit_from_xy = matrix_xy_to_xy_img_plot_eval()

        def _fit_mask(m: np.ndarray) -> Line:
            l = fit_line(
                homography_image2(
                    (m > 0).astype(np.float32),
                    matrix_for_fit_from_xy @ matrix_uv_to_xy(),
                )
            )
            return homography_line(np.linalg.inv(matrix_for_fit_from_xy), l)

        ls = [_fit_mask(m) for m in frame.out_masks]

        ######################################################

        slopes_intercepts = [line_to_slope_intersect(l) for l in ls]

        slope = float(np.mean([s for s, _ in slopes_intercepts]))
        intercept = float(np.mean([y0 for _, y0 in slopes_intercepts]))

        tilt_rat = math.sqrt(1 + slope**2)

        print("dist:", (slopes_intercepts[0][1] - slopes_intercepts[1][1]) / tilt_rat)
        dist = 1.05

        # print("slope, intercept", slope, intercept)

        intercepts_offset = [0, -tilt_rat * dist, tilt_rat * dist]

        lines_to = [line_from_slope_intersect(slope, intercept + y) for y in intercepts_offset]

        perserve: list[tuple[Line, float]] = [
            (line_x_equals(0), 1.0),
            (line_x_equals(1), 1.0),
            (line_x_equals(2), 1.0),
            ((1.0, 1.0, 0.0), 1.0),
            ((1.0, -1.0, 0.0), 1.0),
        ]
        # perserve: list[Line] = [(1.0, 1.0, 0.0), (1.0, -1.0, 0.0)]

        lines = batched.create_stack(
            [
                *[(l, l_to, 2.0) for (l, l_to) in zip(ls[:2], lines_to[:2])],
                *[(l, l_to, 0.8) for (l, l_to) in zip(ls[2:], lines_to[2:])],
                *[(ck_line(x), ck_line(x), w) for x, w in perserve],
            ]
        )

        shake_correction_dual = solve_homography(
            lines.tuple_map(lambda x, y, w: (dual_l(x), dual_l(y), w))
        )
        # print("shake_correction_dual", shake_correction_dual)
        shake_correction = np.linalg.inv(shake_correction_dual).T

        ######################################################

        shake_correction = np.diag(np.ones(3))

        centering = np.array(
            matrix_rot(-jnp.arctan(slope))
            @ matrix_trans(dy=-(intercept + (intercepts_offset[0] + intercepts_offset[1]) / 2))
        )

        # for pl, l in zip(plot2_line_truth, lines_to[:2]):
        #     pl.set_line(homography_line(matrix_xy_to_xy_img_plot_eval() @ centering, l))
        for pl, l in zip(plot2_line_truth, ls[:2]):
            pl.set_line(homography_line(matrix_xy_to_xy_img_plot_eval() @ centering, l))

        plot2.set_imag(
            homography_image2(
                frame.in_img,
                matrix_xy_to_xy_img_plot_eval() @ centering @ shake_correction @ matrix_uv_to_xy(),
            )
        )

        for l, s in zip(plot2_line, (0, -1, -2, -3)):
            l.set_line(
                homography_line(
                    matrix_xy_to_xy_img_plot_eval() @ centering @ shake_correction,
                    shift_line(msg.line, s),
                )
            )

        sx, sy = point_coord(
            homography_point(matrix_xy_to_xy_img_plot_eval() @ centering, (-0.5, 0))
        )
        ex, ey = point_coord(homography_point(matrix_xy_to_xy_img_plot_eval() @ centering, (0, 0)))
        arrow_plot.set_arrow(sx, sy, ex - sx, ey - sy)

        steering = (msg.drive.drive.steering_angle + 0.035) * 4
        steering_x = 1 / math.sqrt(1 + steering**2)
        steering_y = steering_x * steering

        sx, sy = point_coord(homography_point(matrix_xy_to_xy_img_plot_eval() @ centering, (0, 0)))
        ex, ey = point_coord(
            homography_point(
                matrix_xy_to_xy_img_plot_eval() @ centering,
                (steering_x * 3, steering_y * 3),
            )
        )
        arrow_plot_drive.set_arrow(sx, sy, ex - sx, ey - sy)

        time.sleep(max(0, (msg.image_time - prev_stamp) * 3 - (time.time() - start_t)))
        prev_stamp = msg.image_time
        start_t = time.time()

        plt.tight_layout()

        fig.canvas.draw()
        fig.canvas.flush_events()
