import json
import time
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames_from_jpg_images

from final_challenge.alan import FrameData
from final_challenge.alan.rosbag import get_images
from final_challenge.alan.sam2_video_predictor_example import (
    get_mask,
)
from final_challenge.alan.utils import cast_unchecked_


def viz_data():
    bagpath = Path("/home/alan/6.4200/rosbags_4_29/bag2")

    it = iter(get_images(bagpath))

    it = islice(it, 1300, 1500)

    first = next(it)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot = ax.imshow(first.image)

    prev_stamp = first.time

    for i, msg in enumerate(it):
        start_t = time.time()
        plot.set_data(cast_unchecked_(msg.image))

        fig.canvas.draw()
        fig.canvas.flush_events()

        print("offset", i, (msg.time - prev_stamp))
        time.sleep(max(0, (msg.time - prev_stamp) - (time.time() - start_t)))
        prev_stamp = msg.time


def build_predictor():
    # https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints
    checkpoint = "/home/alan/6.4200/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # checkpoint = "/home/alan/6.4200/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        model_cfg,
        checkpoint,
        # vos_optimized=True,
    )
    return predictor


def main(predictor: SAM2VideoPredictor | None = None):
    bagpath = Path("/home/alan/6.4200/rosbags_4_29/bag2")
    out_dir = Path(__file__).parent.parent / "data" / "johnson_track_rosbag_4_29_labeled/part3"
    # out_dir = Path(__file__).parent.parent / "data" / "johnson_track_rosbag_4_29_labeled_test"
    out_dir.mkdir(parents=True, exist_ok=False)
    # out_dir.mkdir(parents=True, exist_ok=True)

    if predictor is None:
        predictor = build_predictor()

    # it = islice(it, 1300, 1500)
    messages = list(
        # islice(get_images(bagpath), 1300, 1301),
        islice(get_images(bagpath), 1300, 1500),
        # islice(get_images(bagpath), 105, 106),
    )
    # messages = messages[:300]
    # messages = messages[:10]
    # messages = messages[::2]

    # # https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot = ax.imshow(messages[0].image)

    with (
        TemporaryDirectory() as video_dir,
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        # see
        _ = load_video_frames_from_jpg_images
        for i, x in enumerate(messages):
            x.image.convert("RGB").save(Path(video_dir) / f"{i}.jpg")

        inference_state = predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=True,
        )

        ann_frame_idx = 0

        prompts = [
            np.array([[167, 212, 1]]),
            np.array([[505, 225, 1]]),
            # np.array([[516, 231, 1]]),
        ]

        for obj_id, prompt in enumerate(prompts):
            points = prompt[:, :2].astype(np.float32)
            labels = prompt[:, 2].astype(np.int32)

            # _ = show_points
            # show_points(points, labels, ax)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            cur_msg = messages[out_frame_idx]
            (out_dir / f"in_{out_frame_idx}_meta.json").write_text(
                json.dumps(
                    {
                        "time": cur_msg.time,
                    },
                    indent=4,
                )
            )

            cur_image = cur_msg.image
            cur_image.save(out_dir / f"in_{out_frame_idx}.png")

            for out_obj_id, out_mask_logit in zip(out_obj_ids, out_mask_logits):
                out_mask_logit = out_mask_logit.cpu().numpy()
                assert out_mask_logit.shape == (1, 360, 640)
                np.save(out_dir / f"out_{out_frame_idx}_obj{out_obj_id}.npy", out_mask_logit[0])

                image_mask = (get_mask(out_mask_logit > 0, obj_id=out_obj_id) * 255).astype(
                    np.uint8
                )
                cur_image = Image.alpha_composite(cur_image, Image.fromarray(image_mask, "RGBA"))

            plot.set_data(cast_unchecked_(cur_image))
            cur_image.save(out_dir / f"out_{out_frame_idx}_viz.png")

            fig.canvas.draw()
            fig.canvas.flush_events()


def example_plot():
    data_dir = Path(
        "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_4_29_labeled/part2"
    )

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    first = FrameData.load(data_dir, 0)

    img2 = ax1.imshow(np.array(first.viz_img))

    cax = ax2.imshow(first.out_left, cmap="viridis")
    fig.colorbar(cax, ax=[ax1, ax2])

    prev_stamp = first.time

    for i in range(1504):
        start_t = time.time()
        cur = FrameData.load(data_dir, i)

        cax.set_data(cur.out_left)
        img2.set_data(cast_unchecked_(cur.viz_img))

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(max(0, (cur.time - prev_stamp) - (time.time() - start_t)))
        prev_stamp = cur.time
