#!/usr/bin/env bash
set -eo pipefail
project_root="${ROADRUNNER_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
unset CMAKE_PREFIX_PATH ROS_PACKAGE_PATH ROSLISP_PACKAGE_DIRECTORIES PYTHONPATH
source /opt/ros/noetic/setup.bash
if [[ -f "${project_root}/devel/setup.bash" ]]; then source "${project_root}/devel/setup.bash"; fi
export PYTHONPATH="${project_root}/devel/lib/python3/dist-packages:${project_root}/src/roadrunner_mapping:/opt/ros/noetic/lib/python3/dist-packages"
export HF_HOME="${project_root}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export CUPY_CACHE_DIR="${project_root}/.cache/cupy"
export TORCH_HOME="${project_root}/.cache/torch"
export ROS_HOME="${project_root}/.cache/ros"
export ROS_LOG_DIR="${project_root}/logs/ros"
export MPLCONFIGDIR="${project_root}/.cache/matplotlib"
mkdir -p "${CUPY_CACHE_DIR}" "${ROS_HOME}" "${ROS_LOG_DIR}" "${MPLCONFIGDIR}"
cd "${project_root}"
exec "$@"
