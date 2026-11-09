"""IR1 + raw-depth input preparation for the two-camera live estimator."""
from __future__ import annotations

from dual_rgbd_bag import prepare_frame
from realtime_dual_core import image_packet


IR_ENCODINGS = {'mono8', '8uc1', 'y8'}
DEPTH_ENCODINGS = {'16uc1', '32fc1'}


def prepare_ir_pair(pair, meta, maps):
    """Prepare synchronized rectified IR1/depth images for DualOdometry.

    RealSense D4xx depth is rectified in the left-infrared geometry. The ROS
    messages keep distinct optical frame names, so only the IR image is
    required to match the IR CameraInfo frame exactly; raw depth must have the
    same dimensions and a valid depth encoding.
    """
    prepared = dict(index=pair.index, time_s=pair.timestamp_ns / 1e9)
    for camera, messages in pair.cameras.items():
        infrared, depth = messages
        info = meta['cameras'][camera]
        expected_size = (info['width'], info['height'])
        if (infrared.width, infrared.height) != expected_size:
            raise ValueError('IR image does not match CameraInfo for ' + camera)
        if infrared.header.frame_id != info['frame_id']:
            raise ValueError('IR image frame_id does not match CameraInfo for ' + camera)
        if (depth.width, depth.height) != expected_size or not depth.header.frame_id:
            raise ValueError('Raw depth does not match rectified IR dimensions for ' + camera)
        if infrared.encoding.lower() not in IR_ENCODINGS:
            raise ValueError('Expected rectified mono8 IR image for ' + camera)
        if depth.encoding.lower() not in DEPTH_ENCODINGS:
            raise ValueError('Expected raw depth image for ' + camera)
        prepared[camera] = (image_packet(infrared, meta['downsample']),
                            image_packet(depth, meta['downsample']))
    return {camera: prepare_frame(prepared, camera, meta, maps) for camera in ('a', 'b')}
