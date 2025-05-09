#!/usr/bin/env python


from libracecar.sandbox import isolate


@isolate
def main():
    import time
    from pathlib import Path

    import jax
    import numpy as np

    from final_challenge.alan.plot_node import PlotConfig, PlotNode
    from final_challenge.alan.tracker_node import TrackerConfig, TrackerNode
    from libracecar.test_utils import proc_manager

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    # jax.config.update("jax_platform_name", "cpu")

    np.set_printoptions(precision=5, suppress=True)
    # jax.config.update("jax_enable_x64", True)

    # bag_dir = Path("/root/repos/rosbags_4_29")
    # bag = "bag2"
    # bag_dir = Path("/root/repos/rosbags_5_3")
    # bag = "out_bag1"
    # bag = "out_bag2"
    bag_dir = Path("/root/repos/rosbags_5_8")
    bag = "bag2"

    procs = proc_manager.new()
    procs.spin_thread()

    # procs.popen(["rviz2"])
    # procs.popen(["emacs"])

    tracker_cfg = TrackerConfig(
        init_y=-2.5 * TrackerConfig.LANE_WIDTH,
        log_topic="/tracker_log_post",
    )
    plot_cfg = PlotConfig(
        shifts=tracker_cfg.shifts,
        log_topic="/tracker_log_post",
    )

    bag_p = procs.popen(
        [
            *["ros2", "bag", "play", bag],
            # *["--rate", "0.5"],
        ],
        cwd=str(bag_dir),
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )

    procs.ros_node_subproc(PlotNode)(cfg=plot_cfg)

    procs.ros_node_thread(
        lambda context: TrackerNode(context=context, cfg=tracker_cfg),
    )

    # rclpy.init()
    # pc = PlotNode(cfg=plot_cfg)
    # rclpy.spin(pc)
    # rclpy.shutdown()

    time.sleep(10000)


if __name__ == "__main__":
    main()
