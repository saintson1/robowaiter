"""Integration check inside ROS 2 Humble; publishes real cached RGB-D as ROS images.

No physical cameras are opened. Run separately from Windows unittest discovery.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from dual_rgbd_bag import nominal_rig
from realtime_dual_core import pose_quaternion
from ros_dual_odometry import DualOdometryNode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--rate', type=float, default=6.)
    parser.add_argument('--frames', type=int, default=16)
    args = parser.parse_args()
    rclpy.init()
    node = DualOdometryNode(parameter_overrides=[Parameter('frame_stride', value=args.stride),
        Parameter('console_status', value=False), Parameter('stale_timeout_s', value=.3)])
    probe = Node('live_odometry_test_publisher')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(probe)
    odoms, statuses, paths, transforms = [], [], [], []
    qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.BEST_EFFORT)
    latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    probe.create_subscription(Odometry, '/dual_odometry/odom', odoms.append, 10)
    probe.create_subscription(String, '/dual_odometry/status', lambda m: statuses.append(json.loads(m.data)), 10)
    probe.create_subscription(RosPath, '/dual_odometry/path', paths.append, latched)
    probe.create_subscription(TFMessage, '/tf', transforms.append, 10)
    pubs = {c: {s: probe.create_publisher(Image, f'/camera/{c}/{s}/image_raw', qos)
                 for s in ('color', 'aligned_depth_to_color')} for c in ('a', 'b')}
    info_pubs = {c: probe.create_publisher(CameraInfo, f'/camera/{c}/aligned_depth_to_color/camera_info', qos)
                 for c in ('a', 'b')}
    tf_pub = probe.create_publisher(TFMessage, '/tf_static', latched)
    meta = json.loads((args.cache/'cache.json').read_text())['configuration']['calibration']
    infos, tf_messages = {}, []
    nominal_depth = nominal_rig({c: {'frame_id': c+'_depth_optical_frame'} for c in ('a', 'b')}, {})
    for c in ('a', 'b'):
        info = meta['cameras'][c]
        m = CameraInfo()
        m.header.frame_id = info['frame_id']
        m.width, m.height = info['width'], info['height']
        m.k = [info['fx'], 0., info['ppx'], 0., info['fy'], info['ppy'], 0., 0., 1.]
        m.d, m.distortion_model = info['coeffs'], info['distortion_model']
        infos[c] = m
        factory = np.linalg.inv(nominal_depth[c]) @ np.array(meta['T_body_from_camera'][c])
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = c+'_depth_optical_frame', info['frame_id']
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, factory[:3, 3])
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, pose_quaternion(factory))
        tf_messages.append(tf)

    def spin_for(duration):
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=min(.02, max(0., deadline-time.monotonic())))

    try:
        for _ in range(15):
            tf_pub.publish(TFMessage(transforms=tf_messages))
            for c in ('a', 'b'):
                info_pubs[c].publish(infos[c])
            spin_for(.1)
        assert node.calibration.meta is not None, 'Camera calibration was not acquired through DDS'
        assert not odoms, 'Published an unobserved pose before images'
        data = []
        for path in sorted(args.cache.glob('frame_*.npz'))[:args.frames]:
            with np.load(path) as z:
                data.append({k: z[k].copy() for k in ('a_rgb', 'a_depth', 'b_rgb', 'b_depth')})
        send_started = time.monotonic()
        for i, arrays in enumerate(data):
            for c, delay in (('a', 0), ('b', 4_000_000)):
                for s, kind in (('color', 'rgb'), ('aligned_depth_to_color', 'depth')):
                    array = np.repeat(np.repeat(arrays[c+'_'+kind], 2, axis=0), 2, axis=1)
                    m = Image()
                    stamp = 1_000_000_000+i*67_000_000+delay
                    m.header.stamp.sec, m.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                    m.header.frame_id = infos[c].header.frame_id
                    m.height, m.width = array.shape[:2]
                    m.encoding = 'rgb8' if kind == 'rgb' else '16UC1'
                    m.step = m.width*(3 if kind == 'rgb' else 2)
                    m.data = array.tobytes()
                    pubs[c][s].publish(m)
            # Slow correctness tests must give DDS/executor time after EVERY
            # publish, even if Python image construction missed its deadline.
            spin_for(1/args.rate if args.rate <= 6 else
                     max(.005, send_started+(i+1)/args.rate-time.monotonic()))
        actual_input_rate = len(data)/(time.monotonic()-send_started)
        spin_for(1.5)
        assert 0 < node.sampler.stats['paired'] <= len(data), node.sampler.stats
        assert node.sampler.stats['selected'] == (node.sampler.stats['paired']+args.stride-1)//args.stride
        if args.rate <= 6:
            assert node.sampler.stats['paired'] == len(data), node.sampler.stats
            assert node.stats['overload_drops'] == 0, node.stats
            assert node.stats['processed'] == node.sampler.stats['selected'], node.stats
        assert len(odoms) >= 2, 'No usable odometry estimates'
        assert odoms[0].pose.pose.position.x == odoms[0].pose.pose.position.y == odoms[0].pose.pose.position.z == 0.
        assert odoms[0].header.frame_id == 'odom' and odoms[0].child_frame_id == 'base_link'
        assert odoms[0].twist.covariance[0] == 1e6
        assert odoms[-1].pose.covariance[0] > 0 and transforms and paths
        assert statuses[-1]['state'] in ('stale', 'tracking_lost'), statuses[-1]
        before = len(odoms)
        spin_for(.4)
        assert len(odoms) == before, 'Republished a stale pose as a new measurement'
        # A live intrinsics change must stop tracking instead of mixing calibrations.
        infos['a'].k[0] += 1.
        info_pubs['a'].publish(infos['a'])
        spin_for(1.1)
        assert node.calibration_error and statuses[-1]['state'] == 'calibration_error'
        print(json.dumps(dict(stride=args.stride, target_input_rate=args.rate, actual_input_rate=actual_input_rate,
            odometry_messages=len(odoms), last_compute_ms=node.latest.get('compute_ms'),
            pairing=node.sampler.stats, processing=node.stats, path_messages=len(paths),
            tf_messages=len(transforms), calibration_change_rejected=True, physical_cameras_used=False)), flush=True)
    finally:
        node.close()
        executor.remove_node(node)
        executor.remove_node(probe)
        node.destroy_node()
        probe.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
