#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ros_setup=${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-78}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}
params_file=${GLOBAL_MAPPER_PARAMS:-"$script_dir/config/mapper.yaml"}

if [[ -f "$ros_setup" ]]; then
    source "$ros_setup"
    export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 -B -m dual_rgbd_global_mapper.node \
        --ros-args --params-file "$params_file" "$@"
fi

command -v docker >/dev/null 2>&1 || {
    echo "Neither ROS 2 Humble nor Docker is available" >&2
    exit 2
}

maps_dir=${GLOBAL_MAPPER_MAPS_DIR:-"$script_dir/maps"}
mkdir -p -- "$maps_dir"
terminal=(-i)
[[ -t 0 && -t 1 ]] && terminal+=(-t)

exec docker run --rm --init "${terminal[@]}" \
    --name dual_global_mapper --network host --ipc host \
    --env "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
    --env "ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY" \
    --env PYTHONPATH=/global_mapper \
    --volume "$script_dir:/global_mapper:ro" \
    --volume "$maps_dir:/maps" \
    --entrypoint /bin/bash \
    "${JETSON_REALSENSE_IMAGE:-realsense-image:latest}" -lc \
    'source /opt/ros/humble/setup.bash && exec python3 -B -m dual_rgbd_global_mapper.node --ros-args --params-file /global_mapper/config/mapper.yaml -p map_directory:=/maps "$@"' \
    mapper "$@"
