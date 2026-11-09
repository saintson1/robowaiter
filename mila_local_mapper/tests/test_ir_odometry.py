import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'live_odometry'))

from live_ir_odometry_session import ir_camera_launch
from realtime_dual_core import FramePair, PairSampler, image_packet
from realtime_dual_ir_core import prepare_ir_pair
from sparse_rgbd_odometry import grayscale
from sparse_rgbd_odometry import TrackerSettings
from dual_rgbd_odometry import DualOdometry


def image(array, encoding, frame_id, stamp_ns):
    item = SimpleNamespace()
    item.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000,
                              nanosec=stamp_ns % 1_000_000_000),
        frame_id=frame_id)
    item.height, item.width = array.shape[:2]
    item.encoding = encoding
    item.is_bigendian = False
    item.step = array.strides[0]
    item.data = array.tobytes()
    return item


class IrOdometryTests(unittest.TestCase):
    def test_grayscale_accepts_ir_without_color_conversion(self):
        infrared = np.arange(20, dtype=np.uint8).reshape(4, 5)
        result = grayscale(infrared)
        self.assertIs(result, infrared)

    def test_image_packet_decodes_mono8(self):
        infrared = np.arange(24, dtype=np.uint8).reshape(4, 6)
        packet = image_packet(image(infrared, 'mono8', 'a_infra1_optical_frame', 1), 2)
        self.assertEqual(packet.encoding, 'mono8')
        np.testing.assert_array_equal(packet.array, infrared[::2, ::2])

    def test_prepare_ir_pair_uses_ir_intrinsics_and_raw_depth(self):
        meta = {
            'downsample': 1,
            'cameras': {
                camera: {'width': 6, 'height': 4, 'fx': 100., 'fy': 101.,
                         'ppx': 3., 'ppy': 2., 'coeffs': [0.] * 5,
                         'distortion_model': 'plumb_bob',
                         'frame_id': f'{camera}_infra1_optical_frame'}
                for camera in ('a', 'b')
            },
        }
        cameras = {}
        for offset, camera in enumerate(('a', 'b')):
            infrared = np.full((4, 6), 30 + offset, dtype=np.uint8)
            depth = np.full((4, 6), 1000 + offset, dtype=np.uint16)
            cameras[camera] = (
                image(infrared, 'mono8', f'{camera}_infra1_optical_frame', 1_000_000 + offset),
                image(depth, '16UC1', f'{camera}_depth_optical_frame', 1_000_000 + offset),
            )
        pair = FramePair(index=7, timestamp_ns=1_000_000, camera_skew_s=1e-9,
                         cameras=cameras)
        frames = prepare_ir_pair(pair, meta, {})
        self.assertEqual(set(frames), {'a', 'b'})
        self.assertEqual(frames['a'].rgb.shape, (4, 6))
        self.assertEqual(frames['a'].rgb.dtype, np.uint8)
        self.assertEqual(frames['a'].depth.dtype, np.uint16)
        self.assertEqual(frames['a'].depth_scale_m, .001)

    def test_ir_launcher_disables_rgb_and_enables_ir1(self):
        command = ir_camera_launch('a', '123', 60)
        self.assertIn('enable_color:=false', command)
        self.assertIn('enable_infra1:=true', command)
        self.assertIn('depth_module.infra_profile:=640x480x60', command)
        self.assertIn('enable_gyro:=false', command)
        self.assertIn('align_depth.enable:=false', command)

    def test_sampler_accepts_ir_default_stride_eight(self):
        self.assertEqual(PairSampler(stride=8).stride, 8)

    def test_fast_strategy_alternates_cameras(self):
        tracker = DualOdometry({'a': np.eye(4), 'b': np.eye(4)},
                               settings=TrackerSettings(max_corners=800),
                               camera_strategy='alternate')
        tracker.references = {'a': object(), 'b': object()}
        frames = {camera: SimpleNamespace(index=0, intrinsics=object())
                  for camera in ('a', 'b')}
        calls = []

        def fake_register(_reference, _frame, _prediction, **_kwargs):
            calls.append('registered')
            tracks = {'points': np.ones((40, 3)), 'pixels': np.ones((40, 2))}
            return np.eye(4), {'inliers': 40}, tracks

        with patch('dual_rgbd_odometry.register', side_effect=fake_register):
            _transform, quality = tracker.estimate(tracker.references, np.eye(4), frames)
        self.assertEqual(len(calls), 1)
        self.assertEqual(quality['cameras_used'], ['a'])
        calls.clear()
        with patch('dual_rgbd_odometry.register', side_effect=fake_register):
            _transform, quality = tracker.estimate(tracker.references, np.eye(4), frames)
        self.assertEqual(len(calls), 1)
        self.assertEqual(quality['cameras_used'], ['b'])


if __name__ == '__main__':
    unittest.main()
