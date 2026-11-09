#!/usr/bin/env python3
"""ROS 2 live two-camera RGB-D odometry; camera configuration is never written."""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from collections import deque
import json
from queue import Empty, Queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64, String
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from dual_rgbd_odometry import DualOdometry
from realtime_dual_core import (LiveCalibration, PairSampler, body_twist, pose_quaternion,
                                prepare_pair, put_latest)
from realtime_dual_ir_core import prepare_ir_pair
from sparse_rgbd_odometry import angle_deg
from sparse_rgbd_odometry import TrackerSettings


class DualOdometryNode(Node):
    def __init__(self, input_mode='rgb', **kwargs):
        super().__init__('dual_odometry', **kwargs)
        if input_mode not in ('rgb', 'ir'):
            raise ValueError("input_mode must be 'rgb' or 'ir'")
        self.input_mode = input_mode
        self.prepare_pair = prepare_pair if input_mode == 'rgb' else prepare_ir_pair
        ir_defaults = dict(max_corners=800, lk_window=15, lk_max_level=2,
                           lk_iterations=15, pnp_iterations=100,
                           camera_strategy='alternate')
        rgb_defaults = dict(max_corners=1600, lk_window=21, lk_max_level=3,
                            lk_iterations=30, pnp_iterations=200,
                            camera_strategy='all')
        tracking_defaults = ir_defaults if input_mode == 'ir' else rgb_defaults
        defaults = dict(frame_stride=2, estimator_downsample=2,
                        camera_a_prefix='/camera/a', camera_b_prefix='/camera/b',
                        odom_frame='odom', base_frame='base_link', publish_tf=True,
                        rgbd_tolerance_s=.008, camera_tolerance_s=.010,
                        keyframe_distance_m=.05, keyframe_angle_deg=5.,
                        status_hz=1., path_hz=2., max_path_points=3000, stale_timeout_s=.5,
                        console_status=True, pose_stddev_m=.05, pose_stddev_deg=5.,
                        **tracking_defaults)
        self.config = {}
        for name, value in defaults.items():
            self.config[name] = self.declare_parameter(name, value,
                ParameterDescriptor(read_only=True, description='Startup setting; restart to change.')).value
        c = self.config
        self.sampler = PairSampler(c['frame_stride'], c['rgbd_tolerance_s'], c['camera_tolerance_s'])
        for name in ('keyframe_distance_m', 'keyframe_angle_deg', 'status_hz', 'path_hz',
                     'stale_timeout_s', 'pose_stddev_m', 'pose_stddev_deg'):
            if not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(name+' must be positive and finite')
        if c['max_path_points'] < 2 or not c['odom_frame'] or not c['base_frame'] or c['odom_frame'] == c['base_frame']:
            raise ValueError('Invalid path capacity or frame names')
        if type(c['estimator_downsample']) is not int or c['estimator_downsample'] not in (1, 2):
            raise ValueError('estimator_downsample must be 1 or 2')
        for name in ('max_corners', 'lk_window', 'lk_max_level', 'lk_iterations',
                     'pnp_iterations'):
            if type(c[name]) is not int or c[name] <= 0:
                raise ValueError(name + ' must be a positive integer')
        if c['max_corners'] < 40 or c['lk_window'] % 2 != 1:
            raise ValueError('max_corners must be >=40 and lk_window must be odd')
        if c['camera_strategy'] not in ('all', 'alternate'):
            raise ValueError("camera_strategy must be 'all' or 'alternate'")
        cv2.setNumThreads(2)
        cv2.setRNGSeed(2026)
        self.calibration = LiveCalibration(c['estimator_downsample'])
        self.calibration_error = None
        self.queue, self.stop_event = Queue(maxsize=1), threading.Event()
        self.lock = threading.Lock()
        self.stats = dict(processed=0, rejected=0, overload_drops=0, before_calibration=0)
        self.latest = dict(state='waiting_calibration', reason='Waiting for CameraInfo and factory TF')
        self.path_poses = deque(maxlen=c['max_path_points'])
        self.newest_pose = None
        self.last_path_ns = None
        self.previous_pose = self.previous_stamp = None
        self.completion_times = deque(maxlen=120)
        self.last_good_monotonic = None
        self.started = time.monotonic()
        sensor_qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.BEST_EFFORT)
        static_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.odom_pub = self.create_publisher(Odometry, '~/odom', 5)
        self.distance_pub = self.create_publisher(Float64, '~/distance', 5)
        self.path_pub = self.create_publisher(Path, '~/path',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.status_pub = self.create_publisher(String, '~/status', 5)
        self.broadcaster = TransformBroadcaster(self) if c['publish_tf'] else None
        self.subscriptions_live = []
        for camera in ('a', 'b'):
            prefix = c[f'camera_{camera}_prefix'].rstrip('/')
            image_topics = ((('color', '/color/image_raw'), ('depth', '/aligned_depth_to_color/image_raw'))
                            if input_mode == 'rgb' else
                            (('color', '/infra1/image_rect_raw'), ('depth', '/depth/image_rect_raw')))
            for stream, suffix in image_topics:
                self.subscriptions_live.append(self.create_subscription(Image, prefix+suffix,
                    lambda m, camera=camera, stream=stream: self.on_image(camera, stream, m), sensor_qos))
            self.subscriptions_live.append(self.create_subscription(CameraInfo,
                prefix+('/aligned_depth_to_color/camera_info' if input_mode == 'rgb' else '/infra1/camera_info'),
                lambda m, camera=camera: self.on_info(camera, m), sensor_qos))
        self.subscriptions_live.append(self.create_subscription(TFMessage, '/tf_static', self.on_tf, static_qos))
        self.create_timer(1/c['status_hz'], self.publish_status)
        self.create_timer(1/c['path_hz'], self.publish_path)
        self.worker = threading.Thread(target=self.run_worker, name='dual-odometry-worker', daemon=True)
        self.worker.start()
        source = 'RGB-D' if input_mode == 'rgb' else 'IR1-D'
        self.get_logger().info(f"source={source}; frame_stride={c['frame_stride']}; "
                               'odom=~/odom, distance=~/distance, path=~/path. '
                               f'Waiting for both {source} cameras.')

    def on_info(self, camera, message):
        try:
            self.calibration.camera_info(camera, message)
        except ValueError as error:
            self.calibration_error = str(error)

    def on_tf(self, message):
        try:
            self.calibration.static_tf(message)
        except ValueError as error:
            self.calibration_error = str(error)

    def on_image(self, camera, stream, message):
        if self.calibration_error:
            return
        try:
            pairs = self.sampler.push(camera, stream, message)
        except ValueError as error:
            with self.lock:
                self.latest.update(state='invalid_input', reason=str(error))
            return
        for pair in pairs:
            if self.calibration.meta is None:
                self.stats['before_calibration'] += 1
                continue
            self.stats['overload_drops'] += put_latest(self.queue,
                (pair, self.calibration.meta, time.monotonic()))

    def run_worker(self):
        tracker, maps = None, {}
        while not self.stop_event.is_set():
            try:
                pair, meta, enqueued = self.queue.get(timeout=.1)
            except Empty:
                continue
            if self.calibration_error:
                continue
            started = time.monotonic()
            try:
                if tracker is None:
                    settings = TrackerSettings(max_corners=self.config['max_corners'],
                        lk_window=self.config['lk_window'],
                        lk_max_level=self.config['lk_max_level'],
                        lk_iterations=self.config['lk_iterations'],
                        pnp_iterations=self.config['pnp_iterations'])
                    tracker = DualOdometry(meta['T_body_from_camera'],
                        keyframe_distance=self.config['keyframe_distance_m'],
                        keyframe_angle=self.config['keyframe_angle_deg'],
                        settings=settings, camera_strategy=self.config['camera_strategy'])
                frames = self.prepare_pair(pair, meta, maps)
                pose, quality = tracker.track(frames)
                elapsed = time.monotonic()-started
                with self.lock:
                    self.stats['processed'] += 1
                    self.completion_times.append(time.monotonic())
                    self.latest.update(compute_ms=elapsed*1000, queue_delay_ms=(started-enqueued)*1000,
                        camera_skew_ms=pair.camera_skew_s*1000, source_pair_index=pair.index,
                        cameras_used=quality.get('cameras_used', []))
                    if pose is None:
                        self.stats['rejected'] += 1
                        self.latest.update(state='tracking_lost', reason=quality.get('reason'))
                if pose is not None and not self.stop_event.is_set() and not self.calibration_error and rclpy.ok(context=self.context):
                    self.publish_pose(pose, pair.timestamp_ns)
            except Exception as error:
                if self.stop_event.is_set() or not rclpy.ok(context=self.context):
                    break
                with self.lock:
                    self.latest.update(state='processing_error', reason=f'{type(error).__name__}: {error}')
                    self.stats['rejected'] += 1

    def publish_pose(self, pose, stamp_ns):
        stamp = Time(nanoseconds=stamp_ns).to_msg()
        quaternion = pose_quaternion(pose)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, self.config['odom_frame']
        odom.child_frame_id = self.config['base_frame']
        p = odom.pose.pose
        p.position.x, p.position.y, p.position.z = map(float, pose[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, quaternion)
        # Conservative engineering covariance, NOT an estimated uncertainty.
        pose_diagonal = [self.config['pose_stddev_m']**2]*3 + [np.deg2rad(self.config['pose_stddev_deg'])**2]*3
        odom.pose.covariance = np.diag(pose_diagonal).ravel().tolist()
        dt = (stamp_ns-self.previous_stamp)/1e9 if self.previous_stamp is not None else 0.
        twist = body_twist(self.previous_pose, pose, dt)
        odom.twist.covariance = np.diag([1e6]*6).ravel().tolist()
        if twist is not None:
            v, w = twist
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, v)
            odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, w)
            odom.twist.covariance = np.diag([1.]*3+[np.deg2rad(30.)**2]*3).ravel().tolist()
        self.odom_pub.publish(odom)
        distance = float(np.linalg.norm(pose[:3, 3]))
        self.distance_pub.publish(Float64(data=distance))
        if self.broadcaster:
            tf = TransformStamped()
            tf.header, tf.child_frame_id = odom.header, odom.child_frame_id
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pose[:3, 3])
            tf.transform.rotation = p.orientation
            self.broadcaster.sendTransform(tf)
        path_pose = PoseStamped(header=odom.header, pose=p)
        with self.lock:
            self.latest.update(state='tracking', reason=None, position_m=pose[:3, 3].tolist(),
                distance_m=distance, rotation_from_start_deg=angle_deg(pose[:3, :3]),
                measurement_stamp_ns=stamp_ns, twist_available=twist is not None)
            self.last_good_monotonic = time.monotonic()
            self.newest_pose = path_pose
            if self.last_path_ns is None or stamp_ns-self.last_path_ns >= 200_000_000:
                self.path_poses.append(path_pose)
                self.last_path_ns = stamp_ns
        self.previous_pose, self.previous_stamp = pose.copy(), stamp_ns

    def publish_path(self):
        with self.lock:
            if self.newest_pose is None:
                return
            poses, newest = list(self.path_poses), self.newest_pose
        if not poses or poses[-1] is not newest:
            poses.append(newest)
        message = Path(header=newest.header, poses=poses)
        self.path_pub.publish(message)

    def publish_status(self):
        now = time.monotonic()
        with self.lock:
            status = dict(self.latest, **self.stats)
            completions = list(self.completion_times)
            last_good = self.last_good_monotonic
        status.update(self.sampler.stats)
        status.update(frame_stride=self.config['frame_stride'],
                      estimator_downsample=self.config['estimator_downsample'],
                      tracking_source=self.input_mode,
                      camera_strategy=self.config['camera_strategy'],
                      max_corners=self.config['max_corners'],
                      uses_imu=False,
                      extrinsics_measured=False, pose_covariance_is_estimated=False,
                      processing_hz=((len(completions)-1)/(completions[-1]-completions[0])
                          if len(completions)>1 and now-completions[-1] <= self.config['stale_timeout_s'] else 0.))
        stamp = status.get('measurement_stamp_ns')
        status['measurement_age_ms'] = ((self.get_clock().now().nanoseconds-stamp)/1e6
                                        if stamp is not None else None)
        status['last_valid_age_s'] = now-last_good if last_good is not None else None
        if self.calibration_error:
            status.update(state='calibration_error', reason=self.calibration_error)
        elif self.calibration.meta is not None and status['state'] == 'waiting_calibration':
            status.update(state='waiting_images', reason='Waiting for synchronized RGB-D frames')
        elif last_good is not None and now-last_good > self.config['stale_timeout_s'] and status['state'] == 'tracking':
            status.update(state='stale', reason='No fresh valid pose; check cameras or tracking')
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))
        if self.config['console_status']:
            position = status.get('position_m')
            text = (f" x={position[0]:+.3f} y={position[1]:+.3f} z={position[2]:+.3f} m"
                    f" distance={status['distance_m']:.3f} m" if position is not None else '')
            self.get_logger().info(f"{status['state']}{text} | {status['processing_hz']:.1f} Hz"
                f" overload_drops={status['overload_drops']} rejected={status['rejected']}"
                + (f" | {status['reason']}" if status.get('reason') else ''))

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=10.)


def main(args=None, input_mode='rgb'):
    rclpy.init(args=args)
    node = None
    try:
        node = DualOdometryNode(input_mode=input_mode)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
