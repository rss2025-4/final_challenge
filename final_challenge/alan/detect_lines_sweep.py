from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jax import Array, numpy as jnp

from libracecar.batched import batched
from libracecar.utils import fval, jit, pformat_repr, tree_select

from ..homography import (
    Line,
    ck_line,
    homography_line,
    matrix_rot,
    normalize_line,
    shift_line,
)


class score_line_res(eqx.Module):
    hits: Array

    extra: Any = None

    __repr__ = pformat_repr


def _score_line(weights: Array, line: Array) -> score_line_res:
    # ax + by + c = 0
    # y = -(a/b) x - (c/b)

    a, b, c = line / jnp.linalg.norm(line)
    b = tree_select(jnp.abs(b) < 1e-6, on_true=1e-6, on_false=b)

    assert len(weights.shape) == 2

    def inner(x: Array):
        y = (-a / b) * x - (c / b)
        y_round = y.astype(jnp.int32)
        return tree_select(
            (0 < y) & (y < weights.shape[1]),
            on_true=weights.at[x, y_round].get(mode="promise_in_bounds"),
            on_false=0.0,
        )

    hits = jax.vmap(inner)(jnp.arange(len(weights)))
    return score_line_res(jnp.sum(hits))


def _get_max(values: Array, keys: Array):
    (n,) = keys.shape
    assert len(values) == n
    i = jnp.argmax(keys)
    return values[i], keys[i]


class ScoreCtx(eqx.Module):
    #: higher means positions in the image where lines are supposed to show up
    weights: Array

    #: homography matrix from xy coordinates to the `weights` image
    homography: Array

    @jit
    def score(self, xy_line: Line):
        xy_line_pixels = homography_line(jnp.array(self.homography), xy_line)

        a, b, c = jnp.array(xy_line_pixels)
        weights = self.weights.astype(jnp.float32)

        s1 = _score_line(weights, jnp.array([b, a, c]))
        s2 = _score_line(weights.T, jnp.array([a, b, c]))

        # return s1, s2
        return score_line_res(s1.hits + s2.hits, (s1, s2))


@jit
def update_line_one(ctx: ScoreCtx, xy_line: Line) -> tuple[Line, score_line_res]:
    xy_line = xy_line / jnp.linalg.norm(xy_line)

    n_try = 21
    try_rotations = batched.create(jnp.linspace(-0.2, 0.2, num=n_try), (n_try,))
    try_shifts = batched.create(jnp.linspace(-0.2, 0.2, num=n_try), (n_try,))

    def try_one_rot(a_rad: Array):
        line_rot = homography_line(matrix_rot(a_rad), xy_line)

        def try_one_shift(shift: Array) -> tuple[Array, tuple[Line, score_line_res]]:
            try_this = shift_line(line_rot, shift)
            ans = ctx.score(try_this)

            is_center = (jnp.abs(a_rad) < 1e-4) & (jnp.abs(shift) < 1e-4)
            return ans.hits + is_center * 5, (try_this, ans)

        return try_shifts.map(try_one_shift).max(lambda item: item[0])

    _best_score, (line, per_lane_res) = try_rotations.map(try_one_rot).max(lambda item: item[0])

    return line, per_lane_res


@jit
def weighted_average(lines: batched[tuple[Line, fval]]) -> Line:

    tot, tot_w = lines.tuple_map(lambda l, w: (ck_line(normalize_line(l)) * w, w)).sum().uf

    return tot / tot_w


@jit
def update_line(
    ctx: ScoreCtx, xy_line: Line, shifts: Array
) -> tuple[Line, batched[tuple[Line, score_line_res]]]:
    """
    update a line using picture `weights`.

    Args:
        ctx: image observations and homography info

        xy_line: previous estimate of the line

        shifts: array of distances; for each shift s, theres supposed to be a line s away from xy_line

    Returns:
        A tuple (line, line_result)

        line is the best line tried

        line_result: for each shift, how well that line shows up in the image
    """

    xy_line = ck_line(xy_line / jnp.linalg.norm(xy_line))

    def handle_one(shift: Array):
        line, res = update_line_one(ctx, shift_line(xy_line, shift))
        return (shift_line(line, -shift), res.hits), (line, res)

    weighted_lines, res = batched.create_array(shifts).map(handle_one).split_tuple()
    ans = weighted_average(
        batched.concat([weighted_lines, batched.create((xy_line, jnp.array(10.0))).reshape(-1)])
    )

    return ans, res
