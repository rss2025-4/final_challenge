from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np
import PIL.Image
from jax import Array
from jax import numpy as jnp
from scipy.ndimage import uniform_filter
from termcolor import colored

from libracecar.batched import batched
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
from .detect_lines_sweep import ScoreCtx, score_line_res, update_line


def process_image(image: np.ndarray | PIL.Image.Image, color_filter: Array) -> np.ndarray:

    # image = uniform_filter(image, size=3, mode="constant", cval=0.0)

    image = homography_image(image)

    image = color_counter.apply_filter(color_filter, image)

    image = uniform_filter(image.astype(np.float32), size=7, mode="constant", cval=0.0)

    return image


def update_with_image(
    line_xy: Line, image: PIL.Image.Image, color_filter: Array, shifts: list[float]
) -> tuple[Line, batched[score_line_res]]:
    weights = process_image(image, color_filter)
    line_xy, res = update_line(
        ScoreCtx(
            weights=jnp.array(weights),
            weights_mask=jnp.array(homography_mask((image.height, image.width))),
            homography=jnp.array(matrix_xy_to_xy_img()),
        ),
        line_xy,
        jnp.array(shifts),
    )
    return line_xy, res
