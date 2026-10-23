#!/usr/bin/env bash
# Usage: ./scripts/build_docker.bash [image-tag] [extra docker build options...]
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: %s [image-tag] [extra docker build options...]\n' "$0"
  printf 'Default image: roadrunner-mapping:20261023\n'
  exit 0
fi
image="${1:-roadrunner-mapping:20261023}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$(uname -m)" != x86_64 ]]; then
  printf 'This Dockerfile supports Linux x86_64 only.\n' >&2; exit 1
fi
docker info >/dev/null
export DOCKER_BUILDKIT=1
exec docker build --network host --progress=plain \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
  -f "${project_root}/docker/Dockerfile" -t "${image}" "$@" "${project_root}"
