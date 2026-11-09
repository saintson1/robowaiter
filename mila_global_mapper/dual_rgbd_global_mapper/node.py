"""ROS 2 wrapper for odometry-based mapping from two aligned depth cameras."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import struct
import threading
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from .core import (
    CameraIntrinsics,
    MapperConfig,
    OccupancyMappingCore,
    SE2,
    make_camera_extrinsic,
    project_depth,
    wrap_angle,
)


def stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid zero quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array((
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))), dtype=np.float64)


def transform_matrix(message: TransformStamped) -> np.ndarray:
    translation, rotation = message.transform.translation, message.transform.rotation
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    result[:3, 3] = (translation.x, translation.y, translation.z)
    return result


def tf_between(edges: dict[tuple[str, str], np.ndarray], parent: str, child: str) -> np.ndarray:
    """Return T_parent_from_child using a small static-TF graph."""
    if parent == child:
        return np.eye(4, dtype=np.float64)
    pending = deque(((parent, np.eye(4, dtype=np.float64)),))
    visited: set[str] = set()
    while pending:
        node, result = pending.popleft()
        if node == child:
            return result
        if node in visited:
            continue
        visited.add(node)
        for (a, b), matrix in edges.items():
            if a == node and b not in visited:
                pending.append((b, result @ matrix))
            elif b == node and a not in visited:
                pending.append((a, result @ np.linalg.inv(matrix)))
    raise KeyError(f'no static TF chain from {parent!r} to {child!r}')


def image_to_depth_metres(message: Image, depth_scale: float) -> np.ndarray:
    """Decode common single-channel ROS depth encodings, including row padding."""
    encoding = message.encoding.upper()
    if encoding in ('16UC1', 'MONO16'):
        kind, item_size, scale = 'u2', 2, depth_scale
    elif encoding == '32FC1':
        kind, item_size, scale = 'f4', 4, 1.0
    else:
        raise ValueError(f'unsupported depth encoding {message.encoding!r}')
    if message.width < 1 or message.height < 1 or message.step < message.width * item_size:
        raise ValueError('invalid depth image dimensions or step')
    if int(message.step) % item_size:
        raise ValueError('depth image step is not aligned to its element size')
    expected = int(message.step) * int(message.height)
    if len(message.data) < expected:
        raise ValueError('truncated depth image data')
    byte_order = '>' if message.is_bigendian else '<'
    values_per_row = int(message.step) // item_size
    raw = np.frombuffer(message.data, dtype=np.dtype(byte_order + kind), count=values_per_row * message.height)
    raw = raw.reshape(message.height, values_per_row)[:, :message.width]
    return raw.astype(np.float32) * np.float32(scale)


@dataclass(frozen=True)
class DepthSample:
    stamp_ns: int
    frame_id: str
    depth_m: np.ndarray


@dataclass(frozen=True)
class CameraModel:
    frame_id: str
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class MappingTask:
    generation: int
    stamp_ns: int
    pose: SE2
    samples: dict[str, DepthSample]
    models: dict[str, CameraModel]
    transforms: dict[str, np.ndarray]


class DualRgbdGlobalMapper(Node):
    """Build a planar OccupancyGrid using timestamped visual odometry."""

    def __init__(self, **kwargs: Any):
        super().__init__('dual_rgbd_global_mapper', **kwargs)
        defaults: dict[str, Any] = {
            'camera_a_prefix': '/camera/a',
            'camera_b_prefix': '/camera/b',
            'camera_depth_suffix': '/aligned_depth_to_color/image_raw',
            'camera_info_suffix': '/aligned_depth_to_color/camera_info',
            'odom_topic': '/dual_odometry/odom',
            'odometry_status_topic': '/dual_odometry/status',
            'map_topic': '/map',
            # During odometry-based mapping map coordinates are odom coordinates.
            # The saved PGM may later be loaded in a separate map frame.
            'map_frame': 'odom',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'resolution': 0.05,
            'initial_size_x_m': 7.0,
            'initial_size_y_m': 8.0,
            'max_size_x_m': 20.0,
            'max_size_y_m': 20.0,
            'expansion_cells': 64,
            'processing_hz': 5.0,
            'sync_tolerance_s': 0.015,
            'depth_queue_size': 12,
            'pixel_stride': 8,
            'depth_scale': 0.001,
            'min_depth_m': 0.30,
            'max_depth_m': 5.0,
            'min_obstacle_height_m': 0.03,
            'max_obstacle_height_m': 0.50,
            'occupied_probability': 0.70,
            'free_probability': 0.40,
            'log_odds_min': -4.0,
            'log_odds_max': 4.0,
            'pgm_occupied_threshold': 0.65,
            'pgm_free_threshold': 0.196,
            'max_pose_jump_m': 1.0,
            'max_pose_jump_yaw_deg': 45.0,
            'map_directory': str(Path.home() / 'global_maps'),
            'map_name': 'global_map',
            'save_on_finish_loop': True,
            'require_tracking_status': False,
            'max_status_age_s': 2.0,
            'min_loop_path_length_m': 1.0,
            'max_loop_residual_translation_m': 1.5,
            'max_loop_residual_yaw_deg': 60.0,
            'camera_a_depth_frame': 'a_depth_optical_frame',
            'camera_b_depth_frame': 'b_depth_optical_frame',
            'camera_a_x': 0.2801, 'camera_a_y': -0.1178, 'camera_a_z': 0.1087,
            'camera_a_yaw_deg': -30.0, 'camera_a_pitch_deg': 10.0, 'camera_a_roll_deg': 180.0,
            'camera_b_x': 0.2801, 'camera_b_y': 0.1178, 'camera_b_z': 0.1087,
            'camera_b_yaw_deg': 30.0, 'camera_b_pitch_deg': 10.0, 'camera_b_roll_deg': 0.0,
        }
        self.params: dict[str, Any] = {}
        descriptor = ParameterDescriptor(read_only=True, description='Startup setting; restart mapper to change')
        for name, value in defaults.items():
            self.params[name] = self.declare_parameter(name, value, descriptor).value
        self._validate_parameters()

        p = self.params
        config = MapperConfig(
            resolution=float(p['resolution']),
            initial_size_x_m=float(p['initial_size_x_m']),
            initial_size_y_m=float(p['initial_size_y_m']),
            max_size_x_m=float(p['max_size_x_m']),
            max_size_y_m=float(p['max_size_y_m']),
            expansion_cells=int(p['expansion_cells']),
            occupied_probability=float(p['occupied_probability']),
            free_probability=float(p['free_probability']),
            log_odds_min=float(p['log_odds_min']),
            log_odds_max=float(p['log_odds_max']),
            pgm_occupied_threshold=float(p['pgm_occupied_threshold']),
            pgm_free_threshold=float(p['pgm_free_threshold']))
        self.core = OccupancyMappingCore(config)
        self.core_lock = threading.RLock()
        self.data_lock = threading.RLock()
        self.depth_queues = {name: deque(maxlen=int(p['depth_queue_size'])) for name in ('a', 'b')}
        self.camera_models: dict[str, CameraModel] = {}
        self.static_edges: dict[tuple[str, str], np.ndarray] = {}
        self.last_scheduled_stamp_ns = -1
        self.last_integrated_stamp_ns = -1
        self.last_scheduled_pose: SE2 | None = None
        self.generation = 0
        self.map_load_time_ns = self.get_clock().now().nanoseconds
        self.status_state: str | None = None
        self.status_received_monotonic: float | None = None
        self.tasks: Queue[MappingTask | None] = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.stats = {
            'received_depth_a': 0, 'received_depth_b': 0, 'received_odom': 0,
            'scheduled': 0, 'integrated': 0, 'sync_misses': 0, 'status_rejections': 0,
            'processing_drops': 0, 'projection_errors': 0,
        }
        self.last_error = ''

        sensor_qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.BEST_EFFORT)
        static_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_publisher = self.create_publisher(OccupancyGrid, str(p['map_topic']), map_qos)
        self.status_publisher = self.create_publisher(String, '~/status', 5)
        self.subscriptions_live = []
        for camera in ('a', 'b'):
            prefix = str(p[f'camera_{camera}_prefix']).rstrip('/')
            self.subscriptions_live.append(self.create_subscription(
                Image, prefix + str(p['camera_depth_suffix']),
                lambda message, camera=camera: self.on_depth(camera, message), sensor_qos))
            self.subscriptions_live.append(self.create_subscription(
                CameraInfo, prefix + str(p['camera_info_suffix']),
                lambda message, camera=camera: self.on_camera_info(camera, message), sensor_qos))
        self.subscriptions_live.append(self.create_subscription(
            Odometry, str(p['odom_topic']), self.on_odometry, 10))
        self.subscriptions_live.append(self.create_subscription(
            String, str(p['odometry_status_topic']), self.on_odometry_status, 5))
        self.subscriptions_live.append(self.create_subscription(TFMessage, '/tf_static', self.on_static_tf, static_qos))
        self.create_service(Trigger, '~/reset', self.on_reset)
        self.create_service(Trigger, '~/save', self.on_save)
        self.create_service(Trigger, '~/finish_loop', self.on_finish_loop)
        self.create_timer(1.0, self.publish_status)

        self.worker = threading.Thread(target=self.run_worker, name='global-mapper-worker', daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"Mapping {p['camera_a_prefix']} and {p['camera_b_prefix']} at {p['processing_hz']} Hz; "
            f"map resolution={p['resolution']} m; output={p['map_directory']}/{p['map_name']}.[pgm|yaml]")

    def _validate_parameters(self) -> None:
        p = self.params
        positive = ('resolution', 'initial_size_x_m', 'initial_size_y_m',
                    'max_size_x_m', 'max_size_y_m', 'processing_hz',
                    'sync_tolerance_s', 'depth_scale', 'min_depth_m', 'max_depth_m',
                    'max_status_age_s', 'min_loop_path_length_m', 'max_pose_jump_m',
                    'max_pose_jump_yaw_deg')
        for name in positive:
            if not math.isfinite(float(p[name])) or float(p[name]) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if int(p['depth_queue_size']) < 2 or int(p['pixel_stride']) < 1:
            raise ValueError('depth_queue_size must be >=2 and pixel_stride positive')
        for name in ('camera_depth_suffix', 'camera_info_suffix'):
            if not isinstance(p[name], str) or not p[name].startswith('/'):
                raise ValueError(name + ' must be a non-empty topic suffix starting with /')
        if float(p['max_depth_m']) <= float(p['min_depth_m']):
            raise ValueError('max_depth_m must exceed min_depth_m')
        if float(p['max_obstacle_height_m']) <= float(p['min_obstacle_height_m']):
            raise ValueError('invalid obstacle height interval')
        if not str(p['map_frame']) or not str(p['odom_frame']) or not str(p['base_frame']):
            raise ValueError('map_frame, odom_frame and base_frame may not be empty')
        for camera in ('a', 'b'):
            for suffix in ('x', 'y', 'z', 'yaw_deg', 'pitch_deg', 'roll_deg'):
                if not math.isfinite(float(p[f'camera_{camera}_{suffix}'])):
                    raise ValueError(f'camera_{camera}_{suffix} must be finite')

    def on_depth(self, camera: str, message: Image) -> None:
        try:
            sample = DepthSample(stamp_ns(message), str(message.header.frame_id),
                                 image_to_depth_metres(message, float(self.params['depth_scale'])))
            with self.data_lock:
                self.depth_queues[camera].append(sample)
                self.stats[f'received_depth_{camera}'] += 1
        except (ValueError, TypeError) as error:
            self.last_error = f'camera {camera}: {error}'

    def on_camera_info(self, camera: str, message: CameraInfo) -> None:
        try:
            model = CameraModel(str(message.header.frame_id), CameraIntrinsics(
                int(message.width), int(message.height), float(message.k[0]), float(message.k[4]),
                float(message.k[2]), float(message.k[5])))
            with self.data_lock:
                previous = self.camera_models.get(camera)
                if previous is not None and previous != model:
                    raise ValueError(f'CameraInfo for camera {camera} changed; reset/restart required')
                self.camera_models[camera] = model
        except (ValueError, TypeError) as error:
            self.last_error = str(error)

    def on_static_tf(self, message: TFMessage) -> None:
        with self.data_lock:
            for item in message.transforms:
                try:
                    self.static_edges[(str(item.header.frame_id), str(item.child_frame_id))] = transform_matrix(item)
                except ValueError as error:
                    self.last_error = f'invalid static TF: {error}'

    def on_odometry_status(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            state = str(value.get('state', ''))
            if not state:
                raise ValueError('status has no state')
            self.status_state = state
            self.status_received_monotonic = time.monotonic()
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.last_error = f'invalid odometry status: {error}'

    @staticmethod
    def _nearest(queue: deque[DepthSample], wanted_ns: int, tolerance_ns: int) -> DepthSample | None:
        if not queue:
            return None
        sample = min(queue, key=lambda item: abs(item.stamp_ns - wanted_ns))
        return sample if abs(sample.stamp_ns - wanted_ns) <= tolerance_ns else None

    def _camera_transform(self, camera: str, model: CameraModel) -> np.ndarray:
        p = self.params
        nominal_depth = make_camera_extrinsic(
            float(p[f'camera_{camera}_x']), float(p[f'camera_{camera}_y']),
            float(p[f'camera_{camera}_z']), float(p[f'camera_{camera}_yaw_deg']),
            float(p[f'camera_{camera}_pitch_deg']), float(p[f'camera_{camera}_roll_deg']))
        depth_frame = str(p[f'camera_{camera}_depth_frame'])
        if model.frame_id == depth_frame:
            return nominal_depth
        return nominal_depth @ tf_between(self.static_edges, depth_frame, model.frame_id)

    def _tracking_is_valid(self) -> bool:
        if not bool(self.params['require_tracking_status']):
            return True
        age = (math.inf if self.status_received_monotonic is None
               else time.monotonic() - self.status_received_monotonic)
        return self.status_state == 'tracking' and age <= float(self.params['max_status_age_s'])

    def on_odometry(self, message: Odometry) -> None:
        self.stats['received_odom'] += 1
        with self.core_lock:
            if self.core.finalized:
                return
        wanted_ns = stamp_ns(message)
        minimum_period_ns = int(1e9 / float(self.params['processing_hz']))
        if wanted_ns <= self.last_scheduled_stamp_ns or wanted_ns - self.last_scheduled_stamp_ns < minimum_period_ns:
            return
        if str(message.header.frame_id) != str(self.params['odom_frame']):
            self.last_error = f"unexpected odom frame {message.header.frame_id!r}"
            return
        if str(message.child_frame_id) != str(self.params['base_frame']):
            self.last_error = f"unexpected odom child frame {message.child_frame_id!r}"
            return
        if not self._tracking_is_valid():
            self.stats['status_rejections'] += 1
            return
        tolerance_ns = int(float(self.params['sync_tolerance_s']) * 1e9)
        with self.data_lock:
            if set(self.camera_models) != {'a', 'b'}:
                self.stats['sync_misses'] += 1
                return
            samples = {name: self._nearest(self.depth_queues[name], wanted_ns, tolerance_ns)
                       for name in ('a', 'b')}
            if any(sample is None for sample in samples.values()):
                self.stats['sync_misses'] += 1
                return
            if any(samples[name].frame_id != self.camera_models[name].frame_id
                   for name in ('a', 'b')):
                self.last_error = 'depth image frame_id does not match CameraInfo frame_id'
                self.stats['sync_misses'] += 1
                return
            try:
                transforms = {name: self._camera_transform(name, self.camera_models[name])
                              for name in ('a', 'b')}
            except (KeyError, ValueError, np.linalg.LinAlgError) as error:
                self.last_error = f'waiting for camera factory TF: {error}'
                self.stats['sync_misses'] += 1
                return
            models = dict(self.camera_models)
        orientation = message.pose.pose.orientation
        try:
            rotation = quaternion_matrix(orientation.x, orientation.y, orientation.z, orientation.w)
        except ValueError as error:
            self.last_error = str(error)
            return
        position = message.pose.pose.position
        pose = SE2(float(position.x), float(position.y),
                   wrap_angle(math.atan2(rotation[1, 0], rotation[0, 0])))
        if self.last_scheduled_pose is not None:
            jump_m = math.hypot(pose.x - self.last_scheduled_pose.x,
                                pose.y - self.last_scheduled_pose.y)
            jump_yaw = abs(wrap_angle(pose.yaw - self.last_scheduled_pose.yaw))
            if (jump_m > float(self.params['max_pose_jump_m'])
                    or jump_yaw > math.radians(float(self.params['max_pose_jump_yaw_deg']))):
                self.last_error = f'odometry jump rejected: {jump_m:.3f} m, {math.degrees(jump_yaw):.2f} deg'
                return
        task = MappingTask(self.generation, wanted_ns, pose, samples, models, transforms)  # type: ignore[arg-type]
        try:
            self.tasks.put_nowait(task)
        except Full:
            try:
                self.tasks.get_nowait()
                self.tasks.task_done()
            except Empty:
                pass
            self.tasks.put_nowait(task)
            self.stats['processing_drops'] += 1
        self.last_scheduled_stamp_ns = wanted_ns
        self.last_scheduled_pose = pose
        self.stats['scheduled'] += 1

    def run_worker(self) -> None:
        p = self.params
        while not self.stop_event.is_set():
            try:
                task = self.tasks.get(timeout=0.2)
            except Empty:
                continue
            if task is None:
                self.tasks.task_done()
                return
            if task.generation != self.generation:
                self.tasks.task_done()
                continue
            try:
                batches = tuple(project_depth(
                    task.samples[name].depth_m, task.models[name].intrinsics, task.transforms[name],
                    pixel_stride=int(p['pixel_stride']), min_depth_m=float(p['min_depth_m']),
                    max_depth_m=float(p['max_depth_m']),
                    min_obstacle_height_m=float(p['min_obstacle_height_m']),
                    max_obstacle_height_m=float(p['max_obstacle_height_m'])) for name in ('a', 'b'))
                with self.core_lock:
                    if task.generation != self.generation or self.core.finalized:
                        continue
                    self.core.add_frame(task.stamp_ns, task.pose, batches)
                    message = self.make_map_message(task.stamp_ns)
                self.map_publisher.publish(message)
                self.last_integrated_stamp_ns = task.stamp_ns
                self.stats['integrated'] += 1
            except Exception as error:  # keep the mapping worker alive; details are published
                self.stats['projection_errors'] += 1
                self.last_error = f'{type(error).__name__}: {error}'
            finally:
                self.tasks.task_done()

    def make_map_message(self, measurement_stamp_ns: int | None = None) -> OccupancyGrid:
        message = OccupancyGrid()
        now_ns = self.get_clock().now().nanoseconds if measurement_stamp_ns is None else measurement_stamp_ns
        message.header.stamp.sec = int(now_ns // 1_000_000_000)
        message.header.stamp.nanosec = int(now_ns % 1_000_000_000)
        message.header.frame_id = str(self.params['map_frame'])
        message.info.map_load_time.sec = int(self.map_load_time_ns // 1_000_000_000)
        message.info.map_load_time.nanosec = int(self.map_load_time_ns % 1_000_000_000)
        message.info.resolution = float(self.core.grid.config.resolution)
        message.info.width = self.core.grid.width
        message.info.height = self.core.grid.height
        message.info.origin.position.x = float(self.core.grid.origin_x)
        message.info.origin.position.y = float(self.core.grid.origin_y)
        message.info.origin.orientation.w = 1.0
        message.data = self.core.grid.occupancy_values().ravel(order='C').tolist()
        return message

    def _save_locked(self) -> tuple[Path, Path]:
        return self.core.save(str(self.params['map_directory']), str(self.params['map_name']))

    def on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.generation += 1
        while True:
            try:
                self.tasks.get_nowait()
                self.tasks.task_done()
            except Empty:
                break
        with self.core_lock:
            self.core.reset()
        with self.data_lock:
            for queue in self.depth_queues.values():
                queue.clear()
        self.last_scheduled_stamp_ns = self.last_integrated_stamp_ns = -1
        self.last_scheduled_pose = None
        self.map_load_time_ns = self.get_clock().now().nanoseconds
        response.success, response.message = True, 'map and stored keyframes reset'
        return response

    def on_save(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            with self.core_lock:
                pgm, yaml = self._save_locked()
            response.success, response.message = True, f'saved {pgm} and {yaml}'
        except (ValueError, OSError) as error:
            response.success, response.message = False, str(error)
        return response

    def on_finish_loop(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self.generation += 1
            while True:
                try:
                    self.tasks.get_nowait()
                    self.tasks.task_done()
                except Empty:
                    break
            with self.core_lock:
                result = self.core.finish_loop(
                    min_path_length_m=float(self.params['min_loop_path_length_m']),
                    max_residual_translation_m=float(self.params['max_loop_residual_translation_m']),
                    max_residual_yaw_rad=math.radians(float(self.params['max_loop_residual_yaw_deg'])))
                self.map_publisher.publish(self.make_map_message())
                saved = self._save_locked() if bool(self.params['save_on_finish_loop']) else None
            suffix = f'; saved {saved[0]} and {saved[1]}' if saved else ''
            response.success = True
            response.message = (f"closed {int(result['frames'])} frames over {result['path_length_m']:.2f} m; "
                                f"residual={result['residual_translation_m']:.3f} m, "
                                f"{math.degrees(result['residual_yaw_rad']):.2f} deg{suffix}")
        except (ValueError, RuntimeError, OSError) as error:
            response.success, response.message = False, str(error)
        return response

    def publish_status(self) -> None:
        with self.core_lock:
            payload = dict(self.stats)
            payload.update(frames=len(self.core.frames), finalized=self.core.finalized,
                           map_width=self.core.grid.width, map_height=self.core.grid.height,
                           map_resolution=self.core.grid.config.resolution,
                           observed_cells=int(np.count_nonzero(self.core.grid.observed)),
                           integration_totals=dict(self.core.integration_totals),
                           last_error=self.last_error or None,
                           odometry_state=self.status_state,
                           last_integrated_stamp_ns=(self.last_integrated_stamp_ns
                                                     if self.last_integrated_stamp_ns >= 0 else None))
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.tasks.put_nowait(None)
        except Full:
            try:
                self.tasks.get_nowait()
                self.tasks.task_done()
            except Empty:
                pass
            self.tasks.put_nowait(None)
        self.worker.join(timeout=10.0)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: DualRgbdGlobalMapper | None = None
    try:
        node = DualRgbdGlobalMapper()
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
