#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/ros.bash" rosnode kill /roadrunner_bag /roadrunner_rviz /roadrunner /roadrunner_semantic 2>/dev/null || true
for name in roadrunner_bag roadrunner_rviz roadrunner_mapping roadrunner_core; do
  if tmux has-session -t "=${name}" 2>/dev/null; then
    tmux send-keys -t "${name}:0.0" C-c
    printf 'Sent Ctrl-C: %s\n' "${name}"
  fi
done
