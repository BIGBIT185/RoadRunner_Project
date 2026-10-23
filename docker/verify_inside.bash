#!/usr/bin/env bash
# Invoked only by verify_docker.bash in a temporary workspace/container.
set -euo pipefail
project_root="${ROADRUNNER_PROJECT_ROOT:?}"
runtime=(bash "${project_root}/scripts/container_env.bash")
port="${1:?}"
bag="${2:-}"
cd "${project_root}"
mkdir -p reports
printf '1/4 Fresh catkin build\n'
"${runtime[@]}" catkin_make -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF > reports/build.log 2>&1
printf '2/4 Core regression\n'
"${runtime[@]}" env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/conda/bin/python3 -m pytest \
  -q -p no:cacheprovider src/roadrunner_mapping/tests | tee reports/tests.log
printf '3/4 ROS imports, Torch CUDA and CuPy RawKernel\n'
"${runtime[@]}" /opt/conda/bin/python3 /opt/roadrunner/verify_runtime.py mapping | tee reports/mapping-runtime.json
"${runtime[@]}" /opt/eomt/bin/python /opt/roadrunner/verify_runtime.py semantic | tee reports/semantic-runtime.json
"${runtime[@]}" rospack find grid_map_rviz_plugin > reports/rviz-plugin.txt
if [[ -z "${bag}" ]]; then
  printf '4/4 Bag validation skipped (pass --bag to include actual model inference and maps).\n'
  exit 0
fi
printf '4/4 GOOSE short replay: model ready, inference, fusion, elevation and probability\n'
pids=()
cleanup() {
  if [[ ${#pids[@]} -gt 0 ]]; then kill "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
"${runtime[@]}" roscore -p "${port}" > reports/roscore.log 2>&1 &
core_pid="$!"
pids+=("${core_pid}")
ready=false
for attempt in {1..60}; do
  kill -0 "${core_pid}" 2>/dev/null || { printf 'Own ROS master exited.\n' >&2; exit 1; }
  if "${runtime[@]}" rosparam list >/dev/null 2>&1; then ready=true; break; fi
  sleep .5
done
[[ "${ready}" == true ]] || { printf 'ROS master startup failed.\n' >&2; exit 1; }
"${runtime[@]}" roslaunch roadrunner_mapping mapping.launch rviz:=false > reports/mapping.log 2>&1 &
launch_pid="$!"
pids+=("${launch_pid}")
"${runtime[@]}" timeout 180 rostopic echo -n 1 /roadrunner_semantic/ready > reports/semantic-ready.log
grep -q 'data: True' reports/semantic-ready.log
"${runtime[@]}" /opt/conda/bin/python3 scripts/verify_bag.py \
  --duration 60 --output reports/goose.json > reports/bag-check.log 2>&1 &
checker="$!"
pids+=("${checker}")
"${runtime[@]}" rosbag play --clock -r .5 -u 25 -d 2 "${bag}" --topics \
  /sensor/camera/windshield/vis/image_rect_color \
  /sensor/camera/windshield/vis/camera_info \
  /sensor/lidar/vls128_roof/velodyne_points /tf /tf_static > reports/bag-play.log 2>&1
wait "${checker}"
kill -0 "${core_pid}" "${launch_pid}"
"${runtime[@]}" timeout 10 rosnode ping -c 1 /roadrunner > reports/mapper-alive.log
"${runtime[@]}" timeout 10 rosnode ping -c 1 /roadrunner_semantic > reports/semantic-alive.log
"${runtime[@]}" /opt/conda/bin/python3 - <<'PY'
import json
p = json.load(open('reports/goose.json'))
assert p['passed'], p['errors']
for key in ('mapper_diagnostics', 'semantic_diagnostics'):
    assert p[key], 'Missing ' + key
    for name, value in p[key].items():
        if 'failures' in name:
            assert int(value) == 0, (key, name, value)
print('GOOSE passed; semantic predictions:', p['semantic_predictions'],
      '; semantic fusions:', p['mapper_diagnostics']['semantic_processed'],
      '; maps:', {k:v['count'] for k,v in p['maps'].items()})
PY
