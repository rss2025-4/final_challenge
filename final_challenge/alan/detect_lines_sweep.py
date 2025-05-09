from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jax import Array, numpy as jnp

from final_challenge.homography import (
    Line,
    homography_line,
    matrix_rot,
    shift_line,
)
from libracecar.batched import batched
from libracecar.utils import jit, pformat_repr, tree_select


class score_line_res(eqx.Module):
    hits: Array

    extra: Any = None

    __repr__ = pformat_repr


def _score_line(weights: Array, weights_mask: Array, line: Array) -> score_line_res:
    # ax + by + c = 0
    # y = -(a/b) x - (c/b)

    a, b, c = line / jnp.linalg.norm(line)
    b = tree_select(jnp.abs(b) < 1e-6, on_true=1e-6, on_false=b)

    assert len(weights.shape) == 2
    assert weights.shape == weights_mask.shape
    print("weights_mask", weights_mask.dtype)

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

    #: boolean array, False ==> weights is ignored
    weights_mask: Array

    #: homography matrix from xy coordinates to the `weights` image
    homography: Array

    @jit
    def score(self, xy_line: Line):
        xy_line_pixels = homography_line(jnp.array(self.homography), xy_line)

        a, b, c = jnp.array(xy_line_pixels)
        weights = self.weights.astype(jnp.float32)

        s1 = _score_line(weights, self.weights_mask, jnp.array([b, a, c]))
        s2 = _score_line(weights.T, self.weights_mask.T, jnp.array([a, b, c]))

        # return s1, s2
        return score_line_res(s1.hits + s2.hits, (s1, s2))


@jit
def update_line(
    ctx: ScoreCtx, xy_line: Line, shifts: Array
) -> tuple[Line, batched[score_line_res]]:
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

    xy_line = xy_line / jnp.linalg.norm(xy_line)

    n_try = 21
    try_rotations = batched.create(jnp.linspace(-0.2, 0.2, num=n_try), (n_try,))
    try_shifts = batched.create(jnp.linspace(-0.2, 0.2, num=n_try), (n_try,))

    def try_one_rot(a_rad: Array):
        line_rot = homography_line(matrix_rot(a_rad), xy_line)

        def try_one_shift(s1: Array) -> tuple[Array, tuple[Line, batched[score_line_res]]]:
            # returns (score, (line, per_lane_results))
            try_this = shift_line(line_rot, s1)

            def per_lane(s2: Array) -> score_line_res:
                # returns (hits, total) for one white line
                line_rot_shift = shift_line(try_this, s2)
                return ctx.score(line_rot_shift)

            per_lane_res = batched.create(shifts, (len(shifts),)).map(per_lane)

            is_center = (jnp.abs(a_rad) < 1e-4) & (jnp.abs(s1) < 1e-4)

            tot = per_lane_res.sum().unwrap()
            return tot.hits, (try_this, per_lane_res)

        return try_shifts.map(try_one_shift).max(lambda item: item[0])

    _best_score, (line, per_lane_res) = try_rotations.map(try_one_rot).max(lambda item: item[0])

    return line, per_lane_res
