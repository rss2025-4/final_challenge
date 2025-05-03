from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import tqdm
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from PIL import Image

from final_challenge.alan import FrameData
from final_challenge.homography import homography_image
from libracecar.utils import jit, tree_at_


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

    def push_image(self, img: Image.Image, mask: np.ndarray):
        return self._push_image(
            img=jnp.array(homography_image(img)),
            mask=jnp.array(homography_image(mask.astype(np.float32))),
        )

    @jit
    def _push_image(self, img: Array, mask: Array):
        assert img.dtype == jnp.uint8
        img = (img // 2)[:, :, :3]

        # img = homography_image(img)
        # mask = homography_image(mask)
        # img = img[175:]
        # mask = mask[175:]

        h, w, _ = img.shape

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
    def apply_filter(color_filter: Array, img: Image.Image | np.ndarray) -> np.ndarray:
        return np.array(color_counter._apply_filter(color_filter, jnp.array(img)))


color_filter_path = Path(__file__).parent / "color_filter.npy"


def compute_color_filter():
    data_dirs = [
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_given_labeled"),
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part1"),
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part2"),
        Path("/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part3"),
    ]

    counter = color_counter.new()

    for data_dir in data_dirs:
        for frame in tqdm.tqdm(FrameData.load_all(data_dir)):
            counter = counter.push_image(frame.in_img, frame.out_left_bool)
            counter = counter.push_image(frame.in_img, frame.out_right_bool)

    ans = np.array(counter.get_ratios())
    np.save(color_filter_path, ans)


def load_color_filter():
    ans = np.load(color_filter_path)
    assert isinstance(ans, np.ndarray)
    return ans
