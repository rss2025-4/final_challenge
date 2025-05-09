import math

import jax
import matplotlib.pyplot as plt
import numpy as np

from final_challenge.alan.controller import (
    cached_controller,
    compute_path,
    compute_path_all,
    compute_score,
)
from libracecar.specs import position
from libracecar.utils import (
    timer,
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
    init = position.create((0, -0.1), -0.1)

    with timer.create() as t:
        ans, loss = compute_path(compute_score(start=init, prev_a=0.1))
        jax.block_until_ready(ans)
        print("took:", t.val)

    print(ans)
    _, points = ans.move(init)

    return points.map(lambda p: p.tran.as_arr()).uf
