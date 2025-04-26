from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.empty import builtin_interfaces__msg__Time

from .utils import unique


@dataclass
class BagImage:
    image: Image.Image
    stamp: builtin_interfaces__msg__Time

    @property
    def time(self):
        return self.stamp.sec + self.stamp.nanosec * 1e-9


def get_images(
    bagpath: Path, topic: str = "/zed/zed_node/rgb/image_rect_color"
) -> Iterable[BagImage]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connection = unique(x for x in reader.connections if x.topic == topic)
        for _connection, _timestamp, rawdata in reader.messages(connections=[connection]):
            msg: Any = reader.deserialize(rawdata, connection.msgtype)

            stamp = msg.header.stamp
            assert isinstance(stamp, builtin_interfaces__msg__Time)

            data = msg.data
            assert isinstance(data, np.ndarray)
            assert data.shape == (921600,)

            bgra_data = data.reshape(360, 640, 4)
            rgba_data = bgra_data[..., [2, 1, 0, 3]]  # Swap B and R channels

            image = Image.fromarray(rgba_data, "RGBA")
            yield BagImage(image=image, stamp=stamp)
