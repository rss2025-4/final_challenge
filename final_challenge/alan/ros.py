from dataclasses import dataclass

import numpy as np
from builtin_interfaces.msg import Time
from PIL import Image
from sensor_msgs.msg import Image as RosImage


@dataclass
class ImageMsg:
    image: Image.Image
    stamp: Time

    @property
    def time(self):
        return self.stamp.sec + self.stamp.nanosec * 1e-9

    @staticmethod
    def parse(msg: RosImage):
        stamp = msg.header.stamp

        data = np.array(msg.data)
        assert data.shape == (921600,)

        bgra_data = data.reshape(360, 640, 4)
        rgba_data = bgra_data[..., [2, 1, 0, 3]]  # Swap B and R channels

        image = Image.fromarray(rgba_data, "RGBA")
        return ImageMsg(image=image, stamp=stamp)
