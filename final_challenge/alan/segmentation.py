from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class FrameData:
    """
    see :meth:`FrameData.load`
    """

    #: image from ros message
    in_img: Image.Image
    #: time stamp in the ros message
    time: float

    #: array same shape as image;
    #: >0 can be interpreted as being part of the left lane boundary;
    #: <0 otherwise. higher means more confident that the pixel is in the region.
    out_masks: list[np.ndarray]

    #: debug image with stuff drawn on it
    viz_img: Image.Image

    @property
    def out_left(self):
        return self.out_masks[0]

    @property
    def out_right(self):
        return self.out_masks[1]

    @staticmethod
    def load(data_dir: Path, idx: int):
        """
        load frame `idx` of the segmentation output from a directory

        example:

        .. code:: python

            from pathlib import Path

            from final_challenge.alan import FrameData

            frame = FrameData.load(Path("/where/you/unzipped/data"), 10)

        """
        outs = []
        for j in itertools.count():
            p = data_dir / f"out_{idx}_obj{j}.npy"
            if p.exists():
                outs.append(np.load(p))
            else:
                break

        return FrameData(
            in_img=Image.open(data_dir / f"in_{idx}.png"),
            time=json.loads((data_dir / f"in_{idx}_meta.json").read_bytes())["time"],
            out_masks=outs,
            viz_img=Image.open(data_dir / f"out_{idx}_viz.png"),
        )

    @staticmethod
    def count(data_dir: Path) -> int:
        i = 0
        while (data_dir / f"in_{i}.png").exists():
            i += 1
        return i

    @property
    def size(self) -> tuple[int, int]:
        return self.in_img.size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def out_left_bool(self) -> np.ndarray:
        return self.out_left > 0

    @property
    def out_right_bool(self) -> np.ndarray:
        return self.out_right > 0

    @staticmethod
    def load_all(data_dir: Path) -> Iterable[FrameData]:
        assert data_dir.exists()
        for i in itertools.count():
            if (data_dir / f"in_{i}.png").exists():
                yield FrameData.load(data_dir, i)
            else:
                return


@dataclass
class FrameDataV2:

    #: image from ros message
    in_img: Image.Image
    #: time stamp in the ros message
    time: float

    out_mask: np.ndarray

    #: debug image with stuff drawn on it
    viz_img: Image.Image

    @staticmethod
    def load(data_dir: Path, idx: int):
        return FrameDataV2(
            in_img=Image.open(data_dir / f"in_{idx}.png"),
            time=json.loads((data_dir / f"in_{idx}_meta.json").read_bytes())["time"],
            out_mask=np.load(data_dir / f"out_{idx}_mask.npy"),
            viz_img=Image.open(data_dir / f"out_{idx}_viz.png"),
        )

    @staticmethod
    def count(data_dir: Path) -> int:
        i = 0
        while (data_dir / f"in_{i}.png").exists():
            i += 1
        return i

    @property
    def size(self) -> tuple[int, int]:
        return self.in_img.size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @staticmethod
    def load_all(data_dir: Path) -> Iterable[FrameDataV2]:
        assert data_dir.exists()
        for i in itertools.count():
            if (data_dir / f"in_{i}.png").exists():
                yield FrameDataV2.load(data_dir, i)
            else:
                return
