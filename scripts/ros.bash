#!/usr/bin/env bash
# Execute in the existing ROS/CUDA installation with ONLY this workspace overlay.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container="${ROADRUNNER_CONTAINER:-roadrunner_bevfusion_container}"
container_root="${ROADRUNNER_CONTAINER_ROOT:-/code_ws_new/$(basename "${project_root}")}"
terminal_flags=(-i)
if [[ -t 0 ]]; then terminal_flags+=(-t); fi
exec docker exec "${terminal_flags[@]}" \
  -u "${ROADRUNNER_DOCKER_USER:-$(id -u):$(id -g)}" \
  -e ROADRUNNER_PROJECT_ROOT="${container_root}" \
  -e ROS_MASTER_URI="${ROADRUNNER_MASTER_URI:-http://127.0.0.1:11423}" \
  -e ROS_IP=127.0.0.1 \
  -w "${container_root}" "${container}" \
  bash --noprofile --norc "${container_root}/scripts/container_env.bash" "$@"
