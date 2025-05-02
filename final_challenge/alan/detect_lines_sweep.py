from __future__ import annotations

import itertools
import time
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from jax import Array, lax
from jax import numpy as jnp
from jax.typing import ArrayLike
from PIL import Image
from scipy.ndimage import uniform_filter

from final_challenge.alan import FrameData
from final_challenge.alan.utils import cast
from final_challenge.homography import (
    ImagPlot,
    Line,
    LinePlot,
    LinePlotXY,
    homography_line,
    homography_point,
    line_from_slope_intersect,
    matrix_rot,
    matrix_xy_to_uv,
    setup_xy_plot,
    shift_line,
    uv_to_xy_line,
    xy_to_uv_line,
)
from libracecar.utils import jit, pformat_repr, tree_at_, tree_select


class _score_line_res(eqx.Module):
    hits: Array
    total: Array

    __repr__ = pformat_repr


def _score_line(weights: Array, line: Array) -> _score_line_res:
    # ax + by + c = 0
    # y = -(a/b) x - (c/b)

    a, b, c = line / jnp.linalg.norm(line)
    b = tree_select(jnp.abs(b) < 1e-6, on_true=1e-6, on_false=b)

    assert len(weights.shape) == 2

    def inner(x: Array):
        y = (-a / b) * x - (c / b)
        return tree_select(
            (0 < y) & (y < weights.shape[1]),
            on_true=(
                weights.at[x, y.astype(jnp.int32)].get(mode="promise_in_bounds"),
                1.0,
            ),
            on_false=(0.0, 0.0),
        )

    hits, total = jax.vmap(inner)(jnp.arange(len(weights)))
    return _score_line_res(jnp.sum(hits), jnp.sum(total))


@jit
def score_line(weight_uncropped: ArrayLike, xy_line: Line):
    a, b, c = jnp.array(xy_to_uv_line(xy_line))
    weight_uncropped = jnp.array(weight_uncropped).astype(jnp.float32)

    # au + bv + c == 0
    # =>
    # au + b(v'+153) + c == 0
    # au + bv'+ (c + 153b) == 0
    weight_cropped = weight_uncropped[200:, :]
    c = c + 200 * b

    s1 = _score_line(weight_cropped, jnp.array([b, a, c]))
    s2 = _score_line(weight_cropped.T, jnp.array([a, b, c]))

    # return s1, s2
    return _score_line_res(s1.hits + s2.hits, s1.total + s2.total)


def _get_max(values: Array, keys: Array):
    (n,) = keys.shape
    assert len(values) == n
    i = jnp.argmax(keys)
    return values[i], keys[i]


@jit
def update_line(weight_uncropped: ArrayLike, xy_line: Line, shifts: Array) -> Array:
    xy_line = xy_line / jnp.linalg.norm(xy_line)

    def per_rot(a_rad: Array):
        line_rot = homography_line(matrix_rot(a_rad), xy_line)

        def per_shift_err(s1: Array):
            shifted_once = shift_line(line_rot, s1)

            def per_lane(s2: Array) -> tuple[Array, Array]:
                # returns (hits, total) for one white line
                line_rot_shift = shift_line(shifted_once, s2)
                score = score_line(weight_uncropped, line_rot_shift)
                return score.hits, score.total

            hits, total = jax.vmap(per_lane)(shifts)
            return jnp.array(shifted_once), jnp.sum(hits) / (jnp.sum(total) + 100)

        lines, scores = jax.vmap(per_shift_err)(jnp.linspace(-0.2, 0.2, num=21))
        return _get_max(lines, scores)

    lines, scores = jax.vmap(per_rot)(jnp.linspace(-0.1, 0.1, num=11))
    line, _score = _get_max(lines, scores)
    return line
