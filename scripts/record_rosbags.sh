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
	/zed/zed_node/left/image_rect_color \
	/zed/zed_node/rgb/image_rect_color \
	/zed/zed_node/right/image_rect_color
