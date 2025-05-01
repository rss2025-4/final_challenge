from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from final_challenge.alan.rosbag import get_images
from final_challenge.homography import (
    ImagPlotXY,
    LinePlot,
    setup_xy_plot,
    xy_plot_top_to_uv_line,
)

np.set_printoptions(precision=7, suppress=True)


def plot_data():
    bagpath = Path("/home/alan/6.4200/rosbags_4_29/bag2")
    messages = get_images(bagpath)

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    it = iter(messages)

    # for _ in range(1000):
    #     _ = next(it)

    first = next(it)

    print("first.width", first.image.width)

    viz_img = ax1.imshow(np.array(first.image))
    ax1.set_ylim((first.image.height, 0))
    LinePlot(ax1).set_line(xy_plot_top_to_uv_line())

    setup_xy_plot(ax2)
    wrap_img = ImagPlotXY(ax2)

    prev_stamp = first.time
    start_t = time.time()

    for cur in it:

        viz_img.set_data(np.array(cur.image))

        wrap_img.set_uv_imag(cur.image)

        time.sleep(max(0, (cur.time - prev_stamp) - (time.time() - start_t)))
        prev_stamp = cur.time
        start_t = time.time()

        fig.canvas.draw()
        fig.canvas.flush_events()
