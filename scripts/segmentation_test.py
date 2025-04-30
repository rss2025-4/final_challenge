from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory

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
from libracecar.utils import jit, tree_at_

np.set_printoptions(precision=7, suppress=True)


class color_counter(eqx.Module):
    full: Array
    lines: Array

    @staticmethod
    def new():
        blank = jnp.zeros((128, 128, 128), dtype=jnp.int32)
        return color_counter(blank, blank)

    @staticmethod
    def _push_one(count: Array, xs: Array, counts: ArrayLike):
        return count.at[xs[:, 0], xs[:, 1], xs[:, 2]].add(counts)

    @jit
    def push_image(self, img: Array, mask: Array):
        h, w, _ = img.shape

        assert img.dtype == jnp.uint8
        img = (img // 2)[:, :, :3]
        h_ = int(h * 0.3)
        img = img[h_:]
        mask = mask[h_:]
        h = img.shape[0]

        self = tree_at_(
            lambda me: me.full,
            self,
            replace_fn=lambda x: self._push_one(x, img.reshape(h * w, 3), 1),
        )
        self = tree_at_(
            lambda me: me.lines,
            self,
            replace_fn=lambda x: self._push_one(x, img.reshape(h * w, 3), mask.reshape(h * w)),
        )

        return self

    @jit
    def get_ratios(self) -> Array:
        lines = self.lines + 1e-10
        full = self.full + 1

        lines /= jnp.sum(lines)
        full /= jnp.sum(full)

        log_ratios = jnp.log(lines / full)
        avg_log_ratios = jnp.sum(lines * log_ratios)

        # return avg_log_ratios

        return log_ratios > avg_log_ratios

    @staticmethod
    @jit
    def _apply_filter(color_filter: Array, img: Array) -> Array:

        def _one(color: Array):
            color = (color // 2)[:3]
            return color_filter[color[0], color[1], color[2]]

        return jax.vmap(jax.vmap(_one))(jnp.array(img))

    @staticmethod
    def apply_filter(color_filter: Array, img: Image.Image) -> np.ndarray:
        return np.array(color_counter._apply_filter(color_filter, jnp.array(img)))


def get_color_filter() -> Array:
    data_dirs = [
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_given_labeled"),
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part1"),
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part2"),
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part3"),
    ]

    counter = color_counter.new()

    for data_dir in data_dirs:
        for frame in tqdm.tqdm(FrameData.load_all(data_dir)):
            counter = counter.push_image(jnp.array(frame.in_img), jnp.array(frame.out_left_bool))
            counter = counter.push_image(jnp.array(frame.in_img), jnp.array(frame.out_right_bool))

    return counter.get_ratios()


def update_mask(memory: tuple[np.ndarray, np.ndarray], filter: np.ndarray, threshold=5e-5):
    moving_avg, prev_mask = memory
    mask = moving_avg / 2 + prev_mask / 2
    mask = (
        #
        uniform_filter(mask.astype(np.float32), size=21, mode="constant", cval=0.0)
        / uniform_filter(np.ones_like(mask), size=21, mode="constant", cval=0.0)
    )
    h = mask.shape[0]
    # print("height", h)
    mask = mask * (np.maximum(np.arange(h) / h - 0.3, 0.0) / 0.7).reshape(h, 1)
    mask = mask * filter

    mask = mask / np.sum(mask)

    mask = (mask * filter) > threshold
    r = 0.1
    return ((prev_mask * r + moving_avg * (1 - r)), mask), mask


def mask_to_line(mask: np.ndarray):
    # return (top_pt, bot_pt)
    xs, ys = np.where(mask)
    coefficients = np.polyfit(xs, ys, 1)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    other = intercept + slope * len(mask)
    return intercept, other


# def get_history_filter(color_filter: Array, threshold=5e-5):
#     data_dir = Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_given_labeled")

#     it = iter(FrameData.load_all(data_dir))

#     prev = next(it)

#     for cur in tqdm.tqdm(it):
#         prev_mask = color_counter.apply_filter(color_filter, prev.in_img) & prev.out_left_bool

#         cur_color = color_counter.apply_filter(color_filter, cur.in_img)
#         new_mask_weights = update_mask(prev_mask, cur_color)

#         cur_truth_mask = cur_color & cur.out_left_bool

#         mask_accepted = (new_mask_weights > threshold) & cur_truth_mask
#         print(np.sum(mask_accepted) / np.sum(cur_truth_mask))

#         # print(np.sum(new_mask_weights * cur_truth_mask) / np.sum(cur_truth_mask))

#         prev = cur


def score_one(color_mask: Array, topy: Array, boty: Array):
    def inner(x: Array):
        y_val = topy + (x / len(color_mask)) * (boty - topy)
        return color_mask.at[x, y_val.astype(np.int32)].get(mode="fill", fill_value=False)

    return jnp.sum(jax.vmap(inner)(jnp.arange(len(color_mask))))


@jit
def fit_line_to_color(color_mask: Array, topy: Array, boty: Array):
    h = color_mask.shape[0]
    color_mask = color_mask * (jnp.maximum(jnp.arange(h) / h - 0.3, 0.0) / 0.7).reshape(h, 1)

    def inner(top_cand: Array):
        def inner2(bot_cand: Array):
            return score_one(color_mask, top_cand, bot_cand)

        bot_cands = jnp.linspace(boty - 100, boty + 100, 50)
        bot_scores = jax.vmap(inner2)(bot_cands)
        am = jnp.argmax(bot_scores)
        return bot_cands[am], bot_scores[am]

    top_cands = jnp.linspace(topy - 100, topy + 100, 50)
    bot_cands, scores = jax.vmap(inner)(top_cands)
    am = jnp.argmax(scores)
    return top_cands[am], bot_cands[am]


def f2(color_filter: Array):
    data_dir = Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_given_labeled")
    # data_dir = Path(
    #     "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part3"
    # )

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    it = iter(FrameData.load_all(data_dir))

    # for _ in range(1000):
    #     _ = next(it)

    first = next(it)

    print("first.width", first.width)

    viz_img = ax1.imshow(np.array(first.viz_img))
    ax1.set_ylim((first.height, 0))

    y1, y2 = mask_to_line(first.out_left_bool)
    line = ax1.plot([y1, y2], [0, first.height], marker="o")[0]

    # fig.add_artist(lines.Line2D([0, 1], [0.47, 0.47], linewidth=3))
    # return

    # ax2.plot()

    # print(mask_to_line(first.out_right_bool))
    # assert False

    # mask_to_line

    memory_ = first.out_right_bool.astype(np.float32)
    memory = (memory_, memory_)

    cax = ax2.imshow(memory_, cmap="gray", vmin=0, vmax=6)
    # cax = ax2.imshow(memory, cmap="viridis", vmin=0, vmax=1e-5)
    fig.colorbar(cax)

    # for _ in range(100):
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #     time.sleep(0.1)

    for cur in it:
        color_mask = color_counter.apply_filter(color_filter, cur.in_img)

        y12, y22 = fit_line_to_color(jnp.array(color_mask), jnp.array(y1), jnp.array(y2))
        y1 = float(y12)
        y2 = float(y22)

        memory, mask = update_mask(memory, color_mask)
        cax.set_data(mask * 5 + color_mask)
        # mask = mask > 1.0e-5

        # y1, y2 = mask_to_line(cur.out_right_bool)
        line.set_data(([y1, y2], [0, first.height]))

        # cax.set_data(mask.astype(np.int32) * 5 + color_mask)
        viz_img.set_data(np.array(cur.viz_img))

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)
