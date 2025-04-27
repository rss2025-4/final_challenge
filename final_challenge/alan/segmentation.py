import json
from dataclasses import dataclass
from pathlib import Path

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
    out_left: np.ndarray

    #: see :attr:`FrameData.out_left`
    out_right: np.ndarray

    #: debug image with stuff drawn on it
    viz_img: Image.Image

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
        return FrameData(
            in_img=Image.open(data_dir / f"in_{idx}.png"),
            time=json.loads((data_dir / f"in_{idx}_meta.json").read_bytes())["time"],
            out_left=np.load(data_dir / f"out_{idx}_obj0.npy"),
            out_right=np.load(data_dir / f"out_{idx}_obj1.npy"),
            viz_img=Image.open(data_dir / f"out_{idx}_viz.png"),
        )
