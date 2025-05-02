from __future__ import annotations

import time
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from PIL import Image
from scipy.ndimage import uniform_filter

from final_challenge.alan import FrameData
from final_challenge.homography import (
    ImagPlot,
    Line,
    LinePlot,
    LinePlotXY,
    homography_line,
    line_from_slope_intersect,
    matrix_rot,
    setup_xy_plot,
    shift_line,
    uv_to_xy_line,
    xy_to_uv_line,
)
from libracecar.utils import jit, pformat_repr, tree_at_, tree_select

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


color_filter_path = Path(__file__).parent / "color_filter.npy"


def get_color_filter():
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

    ans = np.array(counter.get_ratios())
    np.save(color_filter_path, ans)


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
    coefficients = np.polyfit(ys, xs, 1)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    return line_from_slope_intersect(slope, intercept)


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

        lines, scores = jax.vmap(per_shift_err)(jnp.linspace(-0.4, 0.4, num=41))
        return _get_max(lines, scores)

    lines, scores = jax.vmap(per_rot)(jnp.linspace(-0.3, 0.3, num=21))
    line, _score = _get_max(lines, scores)
    return line


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


def plot_data():
    # data_dir = Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_given_labeled")
    data_dir = Path(
        "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part3"
    )

    color_filter = np.load(color_filter_path)

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    it = iter(FrameData.load_all(data_dir))

    # it = itertools.islice(it, 10)

    # for _ in range(1000):
    #     _ = next(it)

    first = next(it)

    # print("first.width", first.width)

    viz_img = ImagPlot(ax1)
    # viz_img = ax1.imshow(np.array(first.viz_img))
    # viz_img = ax1.imshow(np.array(first.viz_img))
    # ax1.set_ylim((first.height, 0))

    line = uv_to_xy_line(mask_to_line(first.out_right_bool))
    # l2 = mask_to_line(first.out_right_bool)

    setup_xy_plot(ax2)

    shifts = [-2, -1, 0, 1, 2]
    lines = [LinePlot(ax1) for _ in shifts]

    line_left_xy = LinePlotXY(ax2)
    line_right_xy = LinePlotXY(ax2)

    for cur in it:

        color_mask = color_counter.apply_filter(color_filter, cur.in_img)
        color_mask = (
            uniform_filter(color_mask.astype(np.float32), size=3, mode="constant", cval=0.0) > 1e-8
        )

        # viz_img.set_imag(cur.viz_img)
        viz_img.set_imag(color_mask)

        line = update_line(color_mask, line, jnp.array(shifts))

        # print("color_mask", color_mask)

        # l1 = mask_to_line(cur.out_left_bool)
        # l2 = mask_to_line(cur.out_right_bool)

        # print(score_line(color_mask, line))
        # print(score_line(color_mask, uv_to_xy_line(l2)))
        print()
        print()
        print()
        print()

        for s, l in zip(shifts, lines):
            l.set_line(xy_to_uv_line(shift_line(line, s)))

        # line_right.set_line(l2)

        line_left_xy.set_line(line)
        # line_right_xy.set_xy_line(uv_to_xy_line(l2))

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)


# def f2(color_filter: Array):
#     data_dir = Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_given_labeled")
#     # data_dir = Path(
#     #     "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part3"
#     # )
#     plt.ion()
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax2 = fig.add_subplot(1, 2, 2)

#     it = iter(FrameData.load_all(data_dir))

#     # for _ in range(1000):
#     #     _ = next(it)

#     first = next(it)

#     print("first.width", first.width)

#     viz_img = ax1.imshow(np.array(first.viz_img))
#     ax1.set_ylim((first.height, 0))

#     # y1, y2 = mask_to_line(first.out_left_bool)
#     # line = ax1.plot([y1, y2], [0, first.height], marker="o")[0]

#     # fig.add_artist(lines.Line2D([0, 1], [0.47, 0.47], linewidth=3))
#     # return

#     # ax2.plot()

#     # print(mask_to_line(first.out_right_bool))
#     # assert False

#     # mask_to_line

#     memory_ = first.out_right_bool.astype(np.float32)
#     memory = (memory_, memory_)

#     cax = ax2.imshow(memory_, cmap="gray", vmin=0, vmax=6)
#     # cax = ax2.imshow(memory, cmap="viridis", vmin=0, vmax=1e-5)
#     fig.colorbar(cax)

#     # for _ in range(100):
#     #     fig.canvas.draw()
#     #     fig.canvas.flush_events()
#     #     time.sleep(0.1)

#     for cur in it:
#         color_mask = color_counter.apply_filter(color_filter, cur.in_img)
#         # crop
#         # color_mask[:153, :] = 0

#         y12, y22 = fit_line_to_color(jnp.array(color_mask), jnp.array(y1), jnp.array(y2))
#         y1 = float(y12)
#         y2 = float(y22)

#         memory, mask = update_mask(memory, color_mask)
#         cax.set_data(mask * 5 + color_mask)
#         # mask = mask > 1.0e-5

#         # y1, y2 = mask_to_line(cur.out_right_bool)
#         # line.set_data(([y1, y2], [0, first.height]))

#         # cax.set_data(mask.astype(np.int32) * 5 + color_mask)
#         viz_img.set_data(np.array(cur.viz_img))

#         fig.canvas.draw()
#         fig.canvas.flush_events()

#         time.sleep(0.1)
