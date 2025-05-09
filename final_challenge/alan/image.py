from __future__ import annotations

import numpy as np
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from final_challenge.homography import (
    Line,
    homography_line,
    matrix_xy_to_xy_img,
    shift_line,
)
from libracecar.batched import batched
from libracecar.utils import cast, jit, tree_select


def _get_line(
    shape: tuple[int, int], mask: Array, line: Line
) -> batched[tuple[tuple[Array, Array], Array]]:
    # ax + by + c = 0
    # y = -(a/b) x - (c/b)

    a, b, c = line / jnp.linalg.norm(line)
    b = tree_select(jnp.abs(b) < 1e-6, on_true=1e-6, on_false=b)

    assert shape == mask.shape

    def inner(x: Array):
        y = cast[Array]()((-a / b) * x - (c / b))
        y_round = y.astype(jnp.int32)
        valid_point = (0 < y) & (y < shape[1]) & (mask.at[x, y_round].get(mode="promise_in_bounds"))
        return (x, y_round), valid_point

    return batched.arange(shape[0]).map(inner)


def get_line(
    width: int, height: int, mask: Array, line: Line
) -> batched[tuple[tuple[Array, Array], Array]]:
    """
    given shape (x, y),
    get the set of points in the rectangle (0, 0) to (x, y)
    that is on line line, masking with mask
    """

    assert mask.shape == (height, width)
    assert mask.dtype == np.bool_

    a, b, c = jnp.array(line)

    part1 = _get_line((height, width), mask, jnp.array([b, a, c]))
    part2 = _get_line((width, height), mask.T, jnp.array([a, b, c])).tuple_map(
        lambda p, m: ((p[1], p[0]), m)
    )

    return batched.concat([part1, part2])


def draw_lines(width: int, height: int, mask: Array, lines: batched[Line]) -> Array:
    """returns bool array"""
    (xs, ys), point_mask = (
        lines.map(lambda line: get_line(width, height, mask, line)).reshape(-1).uf.uf
    )
    return jnp.zeros((height, width), dtype=np.bool_).at[xs, ys].max(point_mask, mode="drop")


@jit
def xy_line_to_xyplot_image(line_xy: Line, shifts: Array) -> Array:
    """returns bool array"""

    def inner(s: Array):
        assert s.shape == ()
        line_xy_s = shift_line(line_xy, s)
        return homography_line(matrix_xy_to_xy_img(), line_xy_s)

    lines = batched.create(shifts, (len(shifts),)).map(inner)
    # return draw_lines(400, 200, jnp.array(homography_mask((640, 360))), lines)
    return draw_lines(400, 200, jnp.ones((200, 400), dtype=np.bool_), lines)


@jit
def color_image(image: ArrayLike, color: tuple[int, int, int, int] = (255, 0, 0, 127)) -> Array:
    image = jnp.array(image)
    return (
        batched.create(image, image.shape)
        .map(
            lambda b: tree_select(
                b,
                on_true=jnp.array(list(color), dtype=np.uint8),
                on_false=jnp.array([0, 0, 0, 0], dtype=np.uint8),
            )
        )
        .uf
    )
