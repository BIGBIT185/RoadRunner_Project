#!/usr/bin/env bash
# Fresh workspace, temporary container and isolated ROS master. No production tmux changes.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ROADRUNNER_IMAGE:-roadrunner-mapping:20261023}"
bag=''
case "${1:-}" in
  --bag) : "${2:?Usage: verify_docker.bash --bag /absolute/path.bag}"; bag="$(realpath "$2")"; shift 2 ;;
  --help|-h) printf 'Usage: %s [--bag /path/to/goose.bag]\nVariables: ROADRUNNER_IMAGE, ROADRUNNER_VERIFY_PORT\n' "$0"; exit 0 ;;
  '') ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then printf 'Unexpected arguments.\n' >&2; exit 2; fi
docker image inspect "${image}" >/dev/null
if [[ -n "${bag}" ]]; then
  [[ -f "${bag}" ]] || { printf 'Bag not found: %s\n' "${bag}" >&2; exit 1; }
  (cd "${project_root}/models/eomt_rugd24" && \
    sha256sum -c ../../src/roadrunner_mapping/config/eomt_rugd_model.sha256)
fi
mkdir -p "${project_root}/.cache" "${project_root}/reports"
workspace="$(mktemp -d "${project_root}/.cache/docker-verify.XXXXXX")"
report_dir="${project_root}/reports/docker-verify-$(date +%Y%m%d-%H%M%S)-$$"
container="roadrunner_verify_$$"
cid_file="${workspace}/container.cid"
cleanup() {
  if [[ -s "${cid_file}" ]]; then
    docker rm -f "$(cat "${cid_file}")" >/dev/null 2>&1 || true
  fi
  rm -rf -- "${workspace}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "${report_dir}"
cp -a "${project_root}/src" "${project_root}/scripts" "${project_root}/docker" "${workspace}/"
# catkin creates this host-specific generated link; build it again in the new workspace.
rm -f "${workspace}/src/CMakeLists.txt"
port="${ROADRUNNER_VERIFY_PORT:-11423}"
if [[ ! "${port}" =~ ^[0-9]{1,5}$ ]] || (( 10#${port} < 1024 || 10#${port} > 65535 )); then
  printf 'ROADRUNNER_VERIFY_PORT must be an integer from 1024 to 65535.\n' >&2; exit 2
fi
# All ROS participants live here; even an explicit production port cannot reach the host master.
options=(--gpus all --network none --user "$(id -u):$(id -g)"
  -e "ROS_MASTER_URI=http://127.0.0.1:${port}" -e ROS_IP=127.0.0.1
  -e ROADRUNNER_PROJECT_ROOT=/code_ws_new/RoadRunner_Project
  --mount "type=bind,src=${workspace},dst=/code_ws_new/RoadRunner_Project"
  --mount "type=bind,src=${report_dir},dst=/code_ws_new/RoadRunner_Project/reports")
if [[ -n "${bag}" ]]; then
  options+=(--mount "type=bind,src=${bag},dst=/data/goose.bag,readonly"
    --mount "type=bind,src=${project_root}/models,dst=/code_ws_new/RoadRunner_Project/models,readonly")
fi
docker image inspect "${image}" --format '{{.Id}}' > "${report_dir}/image-id.txt"
printf 'Image: %s\nPrivate-network ROS port: %s\nReports: %s\n' "${image}" "${port}" "${report_dir}"
docker run --rm --cidfile "${cid_file}" --name "${container}" "${options[@]}" "${image}" \
  bash /code_ws_new/RoadRunner_Project/docker/verify_inside.bash "${port}" "${bag:+/data/goose.bag}" \
  2>&1 | tee "${report_dir}/verification.log"
printf 'Verification passed. Reports: %s\n' "${report_dir}"
