#!/bin/bash

set -euf

if [ -z "$1" ]; then
	echo "Error: No out path provided."
	exit 1
fi

ros2 bag record -o "$1" \
	/clicked_point \
	/parameter_events \
	/scan \
	/tf \
	/tf_static \
	/vesc/odom \
	/vesc/high_level/input/nav_0 \
	/vesc/sensors/imu \
	/tracker_log \
	/zed/zed_node/rgb/image_rect_color
