#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
maps_dir=${MILA_MAPS_DIR:-"$script_dir/../maps"}
map_yaml=${MILA_MAP_YAML:-/maps/global_map.yaml}
initial_x=${MILA_INITIAL_X:-0.0}
initial_y=${MILA_INITIAL_Y:-0.0}
initial_yaw=${MILA_INITIAL_YAW:-0.0}
navigation_enabled=${MILA_NAVIGATION_ENABLED:-false}
goal_x=${MILA_GOAL_X:-0.0}
goal_y=${MILA_GOAL_Y:-0.0}
use_parameter_goal=false
if [[ -n ${MILA_GOAL_X+x} || -n ${MILA_GOAL_Y+x} ]]; then
    [[ -n ${MILA_GOAL_X+x} && -n ${MILA_GOAL_Y+x} ]] || {
        echo "Set both MILA_GOAL_X and MILA_GOAL_Y" >&2
        exit 2
    }
    use_parameter_goal=true
fi

test -f "$maps_dir/global_map.yaml" || {
    echo "Missing saved map: $maps_dir/global_map.yaml" >&2
    exit 2
}

terminal=(-i)
[[ -t 0 && -t 1 ]] && terminal+=(-t)

exec docker run --rm --init "${terminal[@]}" \
    --name mila_navigation --network host --ipc host \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-78}" \
    --env "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}" \
    --env PYTHONPATH=/navigation \
    --volume "$script_dir:/navigation:ro" \
    --volume "$maps_dir:/maps:ro" \
    --entrypoint /bin/bash "${JETSON_REALSENSE_IMAGE:-realsense-image:latest}" -lc \
    'source /opt/ros/humble/setup.bash && \
     python3 -B -m mila_navigation.map_server_node --ros-args \
       --params-file /navigation/config/navigation.yaml -p yaml_file:="$1" & map_pid=$!; \
     python3 -B -m mila_navigation.navigation_node --ros-args \
       --params-file /navigation/config/navigation.yaml \
       -p initial_x:="$2" -p initial_y:="$3" -p initial_yaw:="$4" \
       -p use_parameter_goal:="$6" -p goal_x:="$7" -p goal_y:="$8" & nav_pid=$!; \
     python3 -B -m mila_navigation.safety_node --ros-args \
       --params-file /navigation/config/navigation.yaml \
       -p navigation_enabled:="$5" & safety_pid=$!; \
     trap "kill -INT $map_pid $nav_pid $safety_pid 2>/dev/null || true; wait" INT TERM; \
     wait -n $map_pid $nav_pid $safety_pid; rc=$?; \
     kill -INT $map_pid $nav_pid $safety_pid 2>/dev/null || true; wait || true; exit $rc' \
    navigation "$map_yaml" "$initial_x" "$initial_y" "$initial_yaw" \
    "$navigation_enabled" "$use_parameter_goal" "$goal_x" "$goal_y"

