#!/usr/bin/env bash
set -euo pipefail
package_dir="$(rospack find roadrunner_mapping)"
project_root="$(cd "${package_dir}/../.." && pwd)"
semantic_python="${ROADRUNNER_SEMANTIC_PYTHON:-/opt/eomt/bin/python}"
if [[ ! -x "${semantic_python}" ]]; then
  printf 'EoMT Python not found: %s; see env/README.md\n' "${semantic_python}" >&2
  exit 1
fi
export PYTHONPATH="${project_root}/devel/lib/python3/dist-packages:${package_dir}:/opt/ros/noetic/lib/python3/dist-packages:/opt/roadrunner/ros_python_compat"
export HF_HOME="${project_root}/.cache/huggingface"
export HF_HUB_OFFLINE=1
exec "${semantic_python}" "${package_dir}/scripts/semantic_node.py" "$@"
