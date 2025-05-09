from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tqdm
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_humble_rss import (
    ackermann_msgs__msg__AckermannDriveStamped as DriveMsg,
    builtin_interfaces__msg__Time as TimeMsg,
    geometry_msgs__msg__Vector3 as TrackerMsg,
    sensor_msgs__msg__Image as ImageMsg,
)
from termcolor import colored

from ..homography import Line
from .utils import unique

Msgs = DriveMsg | TimeMsg | TrackerMsg | ImageMsg


@dataclass
class BagImage:
    image: Image.Image
    stamp: TimeMsg

    @property
    def time(self):
        return self.stamp.sec + self.stamp.nanosec * 1e-9

    @staticmethod
    def parse(msg: ImageMsg) -> BagImage:
        stamp = msg.header.stamp

        data = msg.data
        assert data.shape == (921600,)

        bgra_data = data.reshape(360, 640, 4)
        rgba_data = bgra_data[..., [2, 1, 0, 3]]  # Swap B and R channels

        image = Image.fromarray(rgba_data, "RGBA")
        return BagImage(image=image, stamp=stamp)


def get_images(
    bagpath: Path, topic: str = "/zed/zed_node/rgb/image_rect_color"
) -> Iterable[BagImage]:
    typestore = get_typestore(Stores.ROS2_HUMBLE_RSS)
    with Reader(bagpath) as reader:
        connection = unique(x for x in reader.connections if x.topic == topic)
        for _connection, _timestamp, rawdata in reader.messages(connections=[connection]):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            assert isinstance(msg, ImageMsg)
            yield BagImage.parse(msg)


@dataclass
class message_group:
    #: source image
    image: BagImage
    #: time rosbag recorder got the image
    image_time: float
    #: index in all images; some are thrown out
    image_idx: int

    #: line estimate by controller
    line_msg: TrackerMsg

    #: drive produced by pure pursuit based on line
    drive: DriveMsg
    #: time rosbag recorder got the drive
    drive_time: float

    @property
    def line(self) -> Line:
        return (self.line_msg.x, self.line_msg.y, self.line_msg.z)


def get_messages(bagpath: Path) -> list[message_group]:
    typestore = get_typestore(Stores.ROS2_HUMBLE_RSS)
    with Reader(bagpath) as reader:

        # for x in reader.connections:
        #     print(x)
        #     print()

        topics = [
            "/zed/zed_node/rgb/image_rect_color",
            "/tracker_line",
            "/vesc/high_level/input/nav_0",
        ]

        counts = {}

        def _get_msgs():
            prev_timestamp = 0
            image_idx = 0
            for connection, timestamp, rawdata in reader.messages(
                connections=[x for x in reader.connections if x.topic in topics]
            ):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                counts.setdefault(type(msg), 0)
                counts[type(msg)] += 1
                assert isinstance(msg, Msgs)
                if isinstance(msg, ImageMsg):
                    setattr(msg, "image_idx", image_idx)
                    image_idx += 1
                # print(type(msg), (timestamp - prev_timestamp) * 1e-9)
                if isinstance(msg, TrackerMsg):
                    yield (timestamp * 1e-9 + 0.001, msg)
                yield (timestamp * 1e-9, msg)
                prev_timestamp = timestamp

        msgs: list[tuple[float, Msgs]] = list(tqdm.tqdm(_get_msgs()))

        for x, y in counts.items():
            print(x, y)
        print()

        print("sorting...")
        msgs.sort(key=lambda x: x[0], reverse=True)
        print("done")

        # image_stack: list[ImageMsg] = []

        # image -> .. -> nav -> tracker_line

        next_line: tuple[float, TrackerMsg] | None = None
        next_line_drive: list[tuple[float, TrackerMsg, DriveMsg]] = []
        ans: list[message_group] = []

        for time, msg in msgs:
            # print("time", time)

            if isinstance(msg, TrackerMsg):
                next_line = time, msg

            elif isinstance(msg, DriveMsg):
                assert next_line is not None
                next_line_offset = next_line[0] - time
                if next_line_offset > 0.015:
                    print(colored(f"next_line_offset: {next_line_offset}", "red"))
                # assert next_line[0] < time + 0.005
                next_line_drive.insert(0, (time, next_line[1], msg))
                next_line = None

            elif isinstance(msg, ImageMsg):
                if len(next_line_drive) > 0:
                    drive_time, line, drive = next_line_drive.pop()
                    # print("delay:", drive_time - time)

                    ans.append(
                        message_group(
                            image=BagImage.parse(msg),
                            image_time=time,
                            image_idx=getattr(msg, "image_idx"),
                            line_msg=line,
                            drive=drive,
                            drive_time=drive_time,
                        )
                    )
                else:
                    print("missed image message!")

        ans.reverse()
        return ans
