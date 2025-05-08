import math
from typing import Protocol

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import rclpy
from beartype import beartype as typechecker
from jax import Array, lax, numpy as jnp, random, tree_util as jtu
from jaxtyping import jaxtyped
from optax.losses import huber_loss

from final_challenge.alan.controller import (
    cached_controller,
    compute_path,
    compute_path_all,
    compute_path_all_,
    compute_score,
)
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

np.set_printoptions(precision=5, suppress=True)
jax.config.update("jax_enable_x64", True)


def main():
    compute_path_all().save()
    plot()


def plot():

    data = cached_controller.load()

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.imshow(
        data.data_first_drive,
        cmap="viridis",
        extent=(data.angles[0], data.angles[-1], data.ys[-1], data.ys[0]),
    )
    fig.colorbar(cax)


def testfn():
    init = position.create((0, 0), -math.pi + 0.001)

    with timer.create() as t:
        ans, loss = compute_path(compute_score(start=init, prev_a=0.1))
        jax.block_until_ready(ans)
        print("took:", t.val)

    print(ans)
    _, points = ans.move(init)

    return points.map(lambda p: p.tran.as_arr()).uf
