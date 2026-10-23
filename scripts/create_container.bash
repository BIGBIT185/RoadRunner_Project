#!/usr/bin/env bash
# Create a container for scripts/{build,start,ros}.bash; never replace an existing one.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ROADRUNNER_IMAGE:-roadrunner-mapping:20261023}"
container="${ROADRUNNER_CONTAINER:-roadrunner_bevfusion_container}"
headless=false
case "${1:-}" in
  --headless) headless=true ;;
  --help|-h) printf 'Usage: %s [--headless]\nVariables: ROADRUNNER_IMAGE, ROADRUNNER_CONTAINER\n' "$0"; exit 0 ;;
  '') ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then printf 'Too many arguments.\n' >&2; exit 2; fi
if docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'Container already exists: %s\nUse docker start, or set a different ROADRUNNER_CONTAINER.\n' "${container}" >&2
  exit 1
fi
docker image inspect "${image}" >/dev/null
options=(--gpus all --network host --ipc host
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
  -v "$(dirname "${project_root}"):/code_ws_new")
if [[ "${headless}" == false ]]; then
  : "${DISPLAY:?Run from a desktop terminal, or use --headless}"
  command -v xauth >/dev/null
  auth_dir="${project_root}/.cache/docker-xauth"
  mkdir -p "${auth_dir}"
  auth_file="${auth_dir}/.docker.xauth"
  touch "${auth_file}"
  records="$(xauth nlist "${DISPLAY}")"
  if [[ -z "${records}" ]]; then printf 'No X11 authorization for DISPLAY=%s\n' "${DISPLAY}" >&2; exit 1; fi
  printf '%s\n' "${records}" | sed -e 's/^..../ffff/' | xauth -f "${auth_file}" nmerge -
  chmod a+r "${auth_file}"
  options+=(-e "DISPLAY=${DISPLAY}" -e XAUTHORITY=/tmp/roadrunner_xauth/.docker.xauth
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "${auth_dir}:/tmp/roadrunner_xauth:ro")
fi
docker run -d --name "${container}" "${options[@]}" "${image}"
printf 'Created %s from %s\nNext: ./scripts/build.bash\n' "${container}" "${image}"
