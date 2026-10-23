#!/usr/bin/env bash
# Each function has its own tmux session, matching the user's terminal convention.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${project_root}/logs"
start_session() {
  local name="$1" command logfile
  shift
  if tmux has-session -t "=${name}" 2>/dev/null; then
    printf 'Existing session: %s\n' "${name}"
    return
  fi
  printf -v command '%q ' "$@"
  printf -v logfile '%q' "${project_root}/logs/${name}.log"
  tmux new-session -d -s "${name}" -n "${name}" \
    -e "ROADRUNNER_MASTER_URI=${ROADRUNNER_MASTER_URI:-http://127.0.0.1:11423}" \
    -e "ROADRUNNER_CONTAINER=${ROADRUNNER_CONTAINER:-roadrunner_bevfusion_container}" \
    -e "ROADRUNNER_CONTAINER_ROOT=${ROADRUNNER_CONTAINER_ROOT:-/code_ws_new/$(basename "${project_root}")}" \
    "${command} 2>&1 | tee -a ${logfile}"
}
if [[ ! -f "${project_root}/devel/setup.bash" ]]; then "${project_root}/scripts/build.bash"; fi
master_uri="${ROADRUNNER_MASTER_URI:-http://127.0.0.1:11423}"
master_port="${master_uri##*:}"
master_port="${master_port%/}"
start_session roadrunner_core "${project_root}/scripts/ros.bash" roscore -p "${master_port}"
ready=false
for attempt in {1..30}; do
  if "${project_root}/scripts/ros.bash" rosparam list >/dev/null 2>&1; then ready=true; break; fi
  sleep .5
done
if [[ "${ready}" != true ]]; then printf 'ROS master did not start; inspect roadrunner_core.\n' >&2; exit 1; fi
start_session roadrunner_mapping "${project_root}/scripts/run.bash" "$@"
start_session roadrunner_rviz env ROADRUNNER_DOCKER_USER=0 "${project_root}/scripts/ros.bash" rviz -d \
  "${ROADRUNNER_CONTAINER_ROOT:-/code_ws_new/$(basename "${project_root}")}/src/roadrunner_mapping/config/mapping.rviz" \
  __name:=roadrunner_rviz
printf 'Sessions: roadrunner_core, roadrunner_mapping, roadrunner_rviz\n'
printf 'Wait for "EoMT ready", then play GOOSE in session roadrunner_bag (see README).\n'
