#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=${JETSON_LIVE_SOURCE_DIR:-"$script_dir/live_odometry"}
runtime_dir=${JETSON_IR_RUNTIME_DIR:-"$script_dir/live_ir_odometry_logs"}

test -f "$source_dir/ros_dual_ir_odometry.py" || {
    echo "Missing IR odometry source: $source_dir/ros_dual_ir_odometry.py" >&2
    exit 2
}
mkdir -p -- "$runtime_dir"

devices=(--device=/dev/bus/usb)
for device in /dev/video* /dev/hidraw*; do
    [[ -c "$device" ]] && devices+=(--device="$device")
done

terminal=(-i)
[[ -t 0 && -t 1 ]] && terminal+=(-t)

exec docker run --rm --init "${terminal[@]}" --name dual_live_odometry \
    --network host --ipc host --runtime nvidia "${devices[@]}" \
    --env "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-78}" \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=2 --env PYTHONDONTWRITEBYTECODE=1 \
    --env "LIVE_UID=$(id -u)" --env "LIVE_GID=$(id -g)" \
    --volume "$source_dir:/live_odometry:ro" \
    --volume "$script_dir:/recording_tools:ro" \
    --volume "$runtime_dir:/live_logs" \
    --entrypoint /bin/bash "${JETSON_REALSENSE_IMAGE:-realsense-image:latest}" -c \
    'source /opt/ros/humble/setup.bash && exec python3 -B /recording_tools/live_ir_odometry_session.py "$@"' ir "$@"
