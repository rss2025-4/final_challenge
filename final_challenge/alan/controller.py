from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import equinox as eqx
import jax
import numpy as np
import optax
from beartype import beartype as typechecker
from jax import Array, lax, numpy as jnp, random, tree_util as jtu
from jaxtyping import jaxtyped
from optax.losses import huber_loss

from libracecar.batched import batched, batched_zip
from libracecar.cone_inference import compute_posterior, cone_dist, cone_location
from libracecar.plot import plot_ctx, plot_style
from libracecar.specs import path, path_segment, position, turn_angle_limit
from libracecar.utils import (
    debug_print,
    flike,
    fval,
    jit,
    pformat_dataclass,
    time_function,
    timer,
    tree_at_,
)


class compute_score(eqx.Module):
    start: position

    prev_a: flike

    def calc_loss(self, p: path) -> fval:
        final_point, points = p.move(self.start)

        points_mid = batched_zip(points, p.parts).tuple_map(
            lambda point, part: (
                #
                path_segment(part.angle, part.length / 2).move(point)
            )
        )

        points = batched.concat([points, points_mid, batched.create(final_point).reshape(1)])

        loss_points = points.map(lambda x: huber_loss(x.tran.y)).sum().unwrap()
        # return loss_points

        loss_final = -final_point.tran.x * 4

        # loss_turns = (
        #     batched_zip(p.parts[:-1], p.parts[1:])
        #     .tuple_map(lambda a, b: jnp.square(a.angle - b.angle))
        #     .sum()
        #     .unwrap()
        #     * 10
        # )

        # loss_turns1 = jnp.square(self.prev_a - p.parts[0].unwrap().angle) * 20

        loss_turns = p.parts.map(lambda p: jnp.square(p.angle) * 10).sum().unwrap()

        return loss_points + loss_final + loss_turns


class patheval(Protocol):
    def calc_loss(self, p: path) -> fval: ...


@jit
def gradient_descent_one(init_path: path, scoring: patheval) -> tuple[path, fval]:

    optim = optax.adamw(learning_rate=0.01)

    def optim_buffers(p: path) -> list[Array]:
        return [p.parts.uf.angle]

    def from_buffers(buffers: list[Array]) -> path:
        return tree_at_(optim_buffers, init_path, buffers)

    def update(buffers: list[Array], opt_state: optax.OptState):
        loss, grads = jax.value_and_grad(
            lambda bs: scoring.calc_loss(from_buffers(bs)),
        )(buffers)

        # debug_print("loss", loss)

        updates, opt_state = optim.update(grads, opt_state, buffers)
        buffers = eqx.apply_updates(buffers, updates)

        buffers = optim_buffers(from_buffers(buffers).clip())

        return (buffers, opt_state), loss

    opt_state = optim.init(optim_buffers(init_path))

    ((ans_bufs, _), losses) = lax.scan(
        lambda c, _: update(c[0], c[1]),
        xs=None,
        init=(optim_buffers(init_path), opt_state),
        length=500,
    )

    ans = from_buffers(ans_bufs)

    # debug_print("losses:", losses[:10], losses[-10:])
    # debug_print("final loss:", scoring.calc_loss(ans))

    return ans, scoring.calc_loss(ans)


@jit
def compute_path(scoring: patheval) -> tuple[path, fval]:
    init_path = path(
        batched.create(path_segment(angle=0.00, length=1.0)).repeat(10),
    )
    return gradient_descent_one(init_path, scoring)


@jit
def compute_path_all_(ys: Array, angles: Array):

    # angles = jnp.linspace(-math.pi, math.pi, 101)[:-1]

    ans = batched.create_array(ys).map(
        lambda y: batched.create_array(angles).map(
            lambda a: compute_path(
                compute_score(
                    start=position.create((0, y), a),
                    prev_a=0.0,
                )
            )
        ),
    )
    return ans.uf


cached_controller_file = Path(__file__).parent / "cached_controller_data.pkl"


@dataclass
class cached_controller:
    ys: np.ndarray
    angles: np.ndarray

    # (ys, angles) ==> value
    data_first_drive: np.ndarray

    def get(self, y: float, a: float) -> float:
        y = float(y)
        a = float(a)

        y_idx = np.argmin(np.abs(self.ys - y))
        a_idx = np.argmin(np.abs(self.angles - a))

        return float(self.data_first_drive[y_idx, a_idx])

    def save(self):
        cached_controller_file.write_bytes(pickle.dumps(self))

    @staticmethod
    def load() -> cached_controller:
        ans = pickle.loads(cached_controller_file.read_bytes())
        assert isinstance(ans, cached_controller)
        return ans


@time_function
def compute_path_all() -> cached_controller:
    # ys = jnp.linspace(-5, 5, 401)
    # angles = jnp.linspace(-math.pi, math.pi, 361)

    ys = jnp.linspace(-5, 5, 201)
    angles = jnp.linspace(-math.pi, math.pi, 201)

    ans = (
        compute_path_all_(ys, angles)
        .tuple_map(
            lambda p, _: p.parts[0].unwrap().angle,
        )
        .uf
    )
    jax.block_until_ready(ans)
    return cached_controller(
        np.array(ys),
        np.array(angles),
        np.array(ans),
    )
