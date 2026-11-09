#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# The IR odometry launcher disables RGB and therefore publishes raw depth in
# the native left-infrared geometry instead of aligned_depth_to_color.
exec "$script_dir/run_global_mapping.sh" \
    -p camera_depth_suffix:=/depth/image_rect_raw \
    -p camera_info_suffix:=/depth/camera_info \
    "$@"
