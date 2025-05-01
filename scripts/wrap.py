from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames_from_jpg_images
from scipy.ndimage import convolve, uniform_filter
from sklearn.svm import SVC

from final_challenge.alan import FrameData
from final_challenge.alan.rosbag import get_images
from final_challenge.alan.sam2_video_predictor_example import (
    get_mask,
    show_points,
)
from final_challenge.alan.utils import cast_unchecked_
from final_challenge.homography import (
    get_homography_matrix,
    line_from_slope_intersect,
    plot_line,
    setup_plot,
    update_plot_line,
    uv_to_xy_line,
)

np.set_printoptions(precision=7, suppress=True)


def warp_perspective(image: Image.Image):
    return cv2.warpPerspective(
        np.array(image),
        np.diag([100, 100, 1])
        @ np.array(
            [
                [0, -1, 2],
                [-1, 0, 2],
                [0, 0, 1],
            ]
        )
        @ get_homography_matrix(),
        (400, 200),
    )


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

    warped_image = warp_perspective(first.image)
    wrap_img = ax2.imshow(warped_image)

    prev_stamp = first.time
    start_t = time.time()

    for cur in it:

        viz_img.set_data(np.array(cur.image))

        warped_image = warp_perspective(cur.image)
        wrap_img.set_data(np.array(warped_image))

        time.sleep(max(0, (cur.time - prev_stamp) - (time.time() - start_t)))
        prev_stamp = cur.time
        start_t = time.time()

        fig.canvas.draw()
        fig.canvas.flush_events()
