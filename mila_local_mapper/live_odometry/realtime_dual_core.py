"""ROS-independent bounded input handling for the live two-camera estimator."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full

import cv2
import numpy as np

from dual_rgbd_bag import ImagePacket, nominal_rig, prepare_frame, quaternion_rotation


def stamp_ns(message):
    stamp = message.header.stamp
    if not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError("Invalid image timestamp")
    return int(stamp.sec)*1_000_000_000 + int(stamp.nanosec)


@dataclass
class FramePair:
    index: int
    timestamp_ns: int
    camera_skew_s: float
    cameras: dict


class PairSampler:
    """Consume each image at most once; stride applies after all four are paired."""
    def __init__(self, stride=2, rgbd_tolerance_s=.008, camera_tolerance_s=.010, capacity=8):
        if type(stride) is not int or stride not in (1, 2, 4, 8):
            raise ValueError("frame_stride must be 1, 2, 4 or 8")
        if not 0 < rgbd_tolerance_s <= .1 or not 0 < camera_tolerance_s <= .1 or capacity < 1:
            raise ValueError("Invalid pairing tolerances/capacity")
        self.stride, self.capacity = stride, capacity
        self.rgbd_ns = round(rgbd_tolerance_s*1e9)
        self.camera_ns = round(camera_tolerance_s*1e9)
        self.pending = {c: {s: deque() for s in ('color', 'depth')} for c in ('a', 'b')}
        self.ready = {c: deque() for c in ('a', 'b')}
        self.last_stamp = {}
        self.previous_pair_stamp = None
        self.stats = dict(paired=0, selected=0, skipped_by_stride=0,
                          discarded_images=0, discarded_camera_pairs=0, nonincreasing_images=0)

    def push(self, camera, stream, message):
        stamp = stamp_ns(message)
        key = camera, stream
        if stamp <= self.last_stamp.get(key, -1):
            self.stats['nonincreasing_images'] += 1
            return []
        self.last_stamp[key] = stamp
        queue = self.pending[camera][stream]
        queue.append(message)
        if len(queue) > self.capacity:
            queue.popleft()
            self.stats['discarded_images'] += 1
        color, depth = (self.pending[camera][s] for s in ('color', 'depth'))
        while color and depth:
            skew = stamp_ns(color[0])-stamp_ns(depth[0])
            if abs(skew) <= self.rgbd_ns:
                self.ready[camera].append((color.popleft(), depth.popleft()))
                if len(self.ready[camera]) > self.capacity:
                    self.ready[camera].popleft()
                    self.stats['discarded_camera_pairs'] += 1
            else:
                (color if skew < 0 else depth).popleft()
                self.stats['discarded_images'] += 1
        output = []
        while self.ready['a'] and self.ready['b']:
            a, b = self.ready['a'][0], self.ready['b'][0]
            skew = stamp_ns(a[0])-stamp_ns(b[0])
            if abs(skew) > self.camera_ns:
                self.ready['a' if skew < 0 else 'b'].popleft()
                self.stats['discarded_camera_pairs'] += 1
                continue
            self.ready['a'].popleft()
            self.ready['b'].popleft()
            stamp = (stamp_ns(a[0])+stamp_ns(b[0]))//2
            if self.previous_pair_stamp is not None and stamp <= self.previous_pair_stamp:
                self.stats['discarded_camera_pairs'] += 2
                continue
            self.previous_pair_stamp = stamp
            index = self.stats['paired']
            self.stats['paired'] += 1
            if index % self.stride:
                self.stats['skipped_by_stride'] += 1
            else:
                self.stats['selected'] += 1
                output.append(FramePair(index, stamp, skew/1e9, {'a': a, 'b': b}))
        return output


def put_latest(queue, item):
    """A full queue drops its pending old frame, never blocks subscription callbacks."""
    dropped = 0
    while True:
        try:
            queue.put_nowait(item)
            return dropped
        except Full:
            try:
                queue.get_nowait()
                dropped += 1
            except Empty:
                pass


def image_packet(message, downsample):
    encoding = message.encoding.lower()
    if encoding in ('rgb8', 'bgr8'):
        dtype, channels = np.dtype(np.uint8), 3
    elif encoding in ('mono8', '8uc1', 'y8'):
        dtype, channels = np.dtype(np.uint8), 1
    elif encoding in ('16uc1', '32fc1'):
        dtype = np.dtype(('>' if message.is_bigendian else '<') + ('u2' if encoding == '16uc1' else 'f4'))
        channels = 1
    else:
        raise ValueError('Unsupported ROS Image encoding: '+message.encoding)
    height, width, step = message.height, message.width, message.step
    if (height < 1 or width < 1 or step < width*channels*dtype.itemsize
            or len(message.data) != height*step or downsample < 1):
        raise ValueError('Invalid ROS Image dimensions/stride/data')
    shape = (height, width, channels) if channels == 3 else (height, width)
    strides = (step, channels*dtype.itemsize, dtype.itemsize) if channels == 3 else (step, dtype.itemsize)
    array = np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides)[::downsample, ::downsample]
    if encoding == 'bgr8':
        array = array[..., ::-1]
    if channels == 1:
        target = (np.uint16 if encoding == '16uc1' else
                  np.float32 if encoding == '32fc1' else np.uint8)
        array = array.astype(target, copy=False)
    return ImagePacket(stamp_ns(message), encoding, array.copy())


class LiveCalibration:
    def __init__(self, downsample=2):
        self.downsample, self.info, self.edges, self.meta = downsample, {}, {}, None

    def camera_info(self, camera, message):
        if (message.width < 1 or message.height < 1 or message.k[0] <= 0 or message.k[4] <= 0
                or not np.isfinite(list(message.k)+list(message.d)).all() or not message.header.frame_id):
            raise ValueError('Invalid CameraInfo for '+camera)
        info = dict(width=int(message.width), height=int(message.height), fx=float(message.k[0]),
                    fy=float(message.k[4]), ppx=float(message.k[2]), ppy=float(message.k[5]),
                    coeffs=list(message.d), distortion_model=message.distortion_model,
                    frame_id=message.header.frame_id)
        if self.meta is not None and info != self.info[camera]:
            raise ValueError('CameraInfo changed: restart odometry to establish a new origin')
        self.info[camera] = info
        self.build()

    def static_tf(self, message):
        for tf in message.transforms:
            t, q = tf.transform.translation, tf.transform.rotation
            matrix = np.eye(4)
            matrix[:3, :3] = quaternion_rotation([q.x, q.y, q.z, q.w])
            matrix[:3, 3] = [t.x, t.y, t.z]
            if not np.isfinite(matrix).all():
                raise ValueError('Nonfinite static TF')
            key = tf.header.frame_id, tf.child_frame_id
            if self.meta is not None and key in self.edges and not np.allclose(matrix, self.edges[key], atol=1e-8):
                raise ValueError('Static camera geometry changed: restart odometry')
            self.edges[key] = matrix
        self.build()

    def build(self):
        if self.meta is not None or len(self.info) != 2:
            return
        try:
            extrinsics = nominal_rig(self.info, self.edges)
        except ValueError:
            return  # Wait for both drivers' latched factory depth-to-color TF.
        self.meta = dict(cameras=dict(self.info), downsample=self.downsample,
                         T_body_from_camera={c: e.tolist() for c, e in extrinsics.items()},
                         extrinsics_measured=False)


def prepare_pair(pair, meta, maps):
    prepared = dict(index=pair.index, time_s=pair.timestamp_ns/1e9)
    for camera, messages in pair.cameras.items():
        info = meta['cameras'][camera]
        for message in messages:
            if ((message.width, message.height) != (info['width'], info['height'])
                    or message.header.frame_id != info['frame_id']):
                raise ValueError('RGB/aligned depth does not match CameraInfo for '+camera)
        if messages[0].encoding.lower() not in ('rgb8', 'bgr8') or messages[1].encoding.lower() not in ('16uc1', '32fc1'):
            raise ValueError('Expected RGB and depth images')
        prepared[camera] = tuple(image_packet(m, meta['downsample']) for m in messages)
    return {c: prepare_frame(prepared, c, meta, maps) for c in ('a', 'b')}


def pose_quaternion(pose):
    """Stable xyzw quaternion, including rotations near 180 degrees."""
    rotation = np.asarray(pose)[:3, :3]
    vector = cv2.Rodrigues(rotation)[0].ravel()
    theta = np.linalg.norm(vector)
    if theta < 1e-12:
        return np.array([0., 0., 0., 1.])
    return np.r_[vector/theta*np.sin(theta/2), np.cos(theta/2)]


def body_twist(previous, current, dt):
    if previous is None or not 0 < dt <= .5:
        return None
    rotation = current[:3, :3]
    linear = rotation.T @ (current[:3, 3]-previous[:3, 3])/dt
    angular = rotation.T @ previous[:3, :3] @ cv2.Rodrigues(previous[:3, :3].T @ rotation)[0].ravel()/dt
    return linear, angular
