#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bag="${1:-/code_ws_new/datasets/GOOSE/6_Gravel_Soil_Forest_Path.bag}"
if [[ $# -gt 0 ]]; then shift; fi
# The existing container binds /home/liu/code_ws_new to /code_ws_new.
bag="${bag/#\/home\/liu\/code_ws_new\//\/code_ws_new\/}"
exec "${script_dir}/ros.bash" rosbag play --clock -d 1.0 "${bag}" "$@" __name:=roadrunner_bag --topics \
  /sensor/camera/windshield/vis/image_rect_color \
  /sensor/camera/windshield/vis/camera_info \
  /sensor/lidar/vls128_roof/velodyne_points /tf /tf_static
