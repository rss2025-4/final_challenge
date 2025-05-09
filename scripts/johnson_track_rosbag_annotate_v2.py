import json
import time
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from jax import numpy as jnp
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames_from_jpg_images
from scipy.ndimage import uniform_filter

from final_challenge.alan.colors import color_counter, load_color_filter
from final_challenge.alan.image import xy_line_to_xyplot_image
from final_challenge.alan.rosbag import get_images
from final_challenge.alan.sam2_video_predictor_example import (
    get_mask,
)
from final_challenge.alan.segmentation import FrameDataV2
from final_challenge.alan.tracker import update_with_image
from final_challenge.alan.utils import cast_unchecked_
from final_challenge.homography import (
    ImagPlot,
    LinePlot,
    homography_image,
    homography_point,
    line_y_equals,
    matrix_xy_to_uv,
    matrix_xy_to_xy_img,
    point_coord,
    shift_line,
    xy_to_uv_line,
)


def viz_data():
    bagpath = Path("/home/alan/6.4200/rosbags_5_3/out_bag1")

    it = iter(get_images(bagpath))

    # it = islice(it, 50, 1950)

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
    bagpath = Path("/home/alan/6.4200/rosbags_5_3/out_bag3")
    out_dir = (
        Path(__file__).parent.parent / "data" / "johnson_track_rosbag_5_3_labeled/bag3/bag3_part1"
    )
    # out_dir = Path(__file__).parent.parent / "data" / "johnson_track_rosbag_5_3_labeled_test"
    out_dir.mkdir(parents=True, exist_ok=False)
    # out_dir.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    end_idx = 500

    # car pos relative to ground
    starting_lane = -1.5

    shifts = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    if predictor is None:
        predictor = build_predictor()

    # 270 372
    # it = islice(it, 1300, 1500)
    messages = list(islice(get_images(bagpath), start_idx, end_idx))
    # messages = messages[:300]
    # messages = messages[:10]
    # messages = messages[::2]

    # # https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot = ImagPlot(ax)

    shift_debug_lines = [LinePlot(ax) for _ in shifts]

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

        # line pos relative to car
        init_line_xy = line_y_equals(-starting_lane)
        color_filter = jnp.array(load_color_filter())

        line_xy, _ = update_with_image(init_line_xy, messages[0].image, color_filter, shifts)

        for i, msg in tqdm.tqdm(enumerate(messages)):
            cur_image = msg.image
            line_xy, _ = update_with_image(line_xy, cur_image, color_filter, shifts)

            for p, s in zip(shift_debug_lines, shifts):
                p.set_line(xy_to_uv_line(shift_line(line_xy, s)))

            plot.set_imag(cur_image)

            fig.canvas.draw()
            fig.canvas.flush_events()

            if i % 5 == 0:
                xy_color_mask = (
                    color_counter.apply_filter(color_filter, homography_image(cur_image)) > 1e-6
                )
                for obj_id, s in enumerate(shifts):
                    xy_plot_mask = np.array(xy_line_to_xyplot_image(line_xy, jnp.array([s])))
                    xy_plot_mask = (
                        uniform_filter(
                            xy_plot_mask.astype(np.float32), size=11, mode="constant", cval=0.0
                        )
                        > 1e-6
                    )
                    xy_plot_mask = xy_plot_mask & xy_color_mask

                    xs, ys = xy_plot_mask.nonzero()
                    if len(xs) < 10:
                        continue
                    xys = np.stack([ys, xs], axis=-1)

                    idxs = np.random.choice(len(xys), 2, replace=False)
                    points = np.array(
                        [
                            point_coord(
                                homography_point(
                                    matrix_xy_to_uv() @ np.linalg.inv(matrix_xy_to_xy_img()),
                                    p,
                                )
                            )
                            for p in xys[idxs]
                        ]
                    )
                    labels = np.ones(len(points), dtype=np.int32)

                    # xy_plot_debug_image = color_image(xy_line_to_xyplot_image(line_xy, jnp.array(shifts)))
                    # uv_debug_image = homography_image_rev(np.array(xy_plot_debug_image))

                    # cur_image = Image.alpha_composite(cur_image, Image.fromarray(uv_debug_image))

                    # plot.set_data(cast_unchecked_(cur_image))

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=i,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                        clear_old_points=False,
                    )

                # show_points(points, labels, ax)
                # assert False

        line_xy, _ = update_with_image(init_line_xy, messages[0].image, color_filter, shifts)

        for plot_line in shift_debug_lines:
            plot_line.set_line(line_y_equals(-1))

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            cur_msg = messages[out_frame_idx]
            (out_dir / f"in_{out_frame_idx}_meta.json").write_text(
                json.dumps(
                    {
                        "time": cur_msg.time,
                        "_bagpath": str(bagpath),
                        "_out_dir": str(out_dir),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "_starting_lane": starting_lane,
                    },
                    indent=4,
                )
            )
            cur_image = cur_msg.image
            cur_image.save(out_dir / f"in_{out_frame_idx}.png")

            out_mask_logits_np = out_mask_logits.cpu().numpy() > 0

            for out_obj_id, out_mask_logit in zip(out_obj_ids, out_mask_logits_np):
                assert out_mask_logit.shape == (1, 360, 640)

                image_mask = (get_mask(out_mask_logit > 0, obj_id=out_obj_id) * 255).astype(
                    np.uint8
                )
                cur_image = Image.alpha_composite(cur_image, Image.fromarray(image_mask, "RGBA"))

            mask = np.squeeze(np.any(out_mask_logits_np, axis=0))
            assert mask.dtype == np.bool_
            np.save(out_dir / f"out_{out_frame_idx}_mask.npy", mask)

            # image_mask = (get_mask(mask) * 255).astype(np.uint8)
            # cur_image = Image.alpha_composite(cur_image, Image.fromarray(image_mask, "RGBA"))

            cur_image.save(out_dir / f"out_{out_frame_idx}_viz.png")

            plot.set_imag(cur_image)

            fig.canvas.draw()
            fig.canvas.flush_events()


def example_plot():
    data_dir = Path(
        "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_5_3_labeled/bag3/bag3_part1"
        # "/home/alan/6.4200/final_challenge2025/data/johnson_track_rosbag_5_3_labeled_test"
    )

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    first = FrameDataV2.load(data_dir, 0)

    img2 = ax1.imshow(np.array(first.viz_img))

    cax = ax2.imshow(first.out_mask, cmap="viridis")
    fig.colorbar(cax, ax=[ax1, ax2])

    prev_stamp = first.time

    for i in range(1504):
        start_t = time.time()
        cur = FrameDataV2.load(data_dir, i)

        cax.set_data(cur.out_mask)
        img2.set_data(cast_unchecked_(cur.viz_img))

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(max(0, (cur.time - prev_stamp) - (time.time() - start_t)))
        prev_stamp = cur.time
