"""Bounded RGB-D pairing for two cameras; ROS 2 on Jetson or MCAP on Windows."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
from types import SimpleNamespace

import cv2
import numpy as np


CAMERAS = ("a", "b")


def image_topic(camera, stream):
    return f"/camera/{camera}/{stream}/image_raw"


def raw_messages(path, topics):
    path = Path(path)
    directory = path / "bag" if (path / "bag").is_dir() else path
    try:
        import rosbag2_py
    except ImportError:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
        factory = DecoderFactory()
        files = sorted(directory.glob("*.mcap"), key=lambda p: [int(x) if x.isdigit() else x
                       for x in re.split(r"(\d+)", p.name)])
        if not files:
            raise ValueError(f"No MCAP files in {directory}")
        for filename in files:
            with filename.open("rb") as stream:
                for schema, channel, message in make_reader(stream).iter_messages(topics=topics):
                    if channel.topic.endswith("/image_raw"):
                        yield channel.topic, message.data
                    else:
                        decoder = factory.decoder_for(channel.message_encoding, schema)
                        yield channel.topic, decoder(message.data)
    else:
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo
        from tf2_msgs.msg import TFMessage
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(directory), storage_id="mcap"),
                    rosbag2_py.ConverterOptions("cdr", "cdr"))
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic.endswith("/image_raw"):
                yield topic, data
            else:
                kind = TFMessage if topic == "/tf_static" else CameraInfo
                yield topic, deserialize_message(data, kind)


@dataclass
class ImagePacket:
    timestamp_ns: int
    encoding: str
    array: np.ndarray


def decode_image(data, downsample=1):
    """Decode sensor_msgs/Image CDR directly, without copying full-size pixels twice."""
    view = memoryview(data)
    if len(view) < 4 or bytes(view[:2]) not in (b"\x00\x01", b"\x00\x00"):
        raise ValueError("Unsupported CDR encapsulation")
    endian = "<" if view[1] else ">"
    offset = 4

    def integer(signed=False):
        nonlocal offset
        offset += (-(offset - 4)) % 4
        value = struct.unpack_from(endian + ("i" if signed else "I"), view, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        length = integer()
        if length < 1 or offset + length > len(view) or view[offset + length - 1] != 0:
            raise ValueError("Invalid CDR string")
        value = bytes(view[offset:offset + length - 1]).decode("utf-8")
        offset += length
        return value

    seconds, nanoseconds = integer(True), integer()
    string()  # frame_id
    height, width = integer(), integer()
    encoding = string().lower()
    bigendian = view[offset]
    offset += 1
    step, size = integer(), integer()
    if downsample < 1 or not width or not height or size != height * step or offset + size > len(view):
        raise ValueError("Invalid ROS Image dimensions or byte count")
    pixels = view[offset:offset + size]
    if encoding in ("rgb8", "bgr8"):
        if step < width * 3:
            raise ValueError("Invalid RGB row stride")
        array = np.ndarray((height, width, 3), np.uint8, pixels, strides=(step, 3, 1))
        array = array[::downsample, ::downsample]
        if encoding == "bgr8":
            array = array[..., ::-1]
    elif encoding in ("16uc1", "32fc1"):
        itemsize = 2 if encoding == "16uc1" else 4
        if step < width * itemsize:
            raise ValueError("Invalid depth row stride")
        dtype = np.dtype((">" if bigendian else "<") + ("u2" if itemsize == 2 else "f4"))
        array = np.ndarray((height, width), dtype, pixels, strides=(step, itemsize))[::downsample, ::downsample]
        array = array.astype(np.uint16 if itemsize == 2 else np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    # Own the small decimated array; do not keep a full-resolution CDR message alive.
    return ImagePacket(seconds * 1_000_000_000 + nanoseconds, encoding, array.copy())


def quaternion_rotation(q):
    q = np.asarray(q, float)
    norm = np.linalg.norm(q)
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("Invalid TF quaternion")
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def calibration(path, downsample=2):
    topics = [f"/camera/{c}/aligned_depth_to_color/camera_info" for c in CAMERAS] + ["/tf_static"]
    info, edges = {}, {}
    for topic, message in raw_messages(path, topics):
        if topic == "/tf_static":
            for tf in message.transforms:
                t, q = tf.transform.translation, tf.transform.rotation
                matrix = np.eye(4)
                matrix[:3, :3] = quaternion_rotation([q.x, q.y, q.z, q.w])
                matrix[:3, 3] = [t.x, t.y, t.z]
                edges[(tf.header.frame_id, tf.child_frame_id)] = matrix
        else:
            camera = topic.split("/")[2]
            if camera not in info:
                info[camera] = {"width": int(message.width), "height": int(message.height),
                    "fx": float(message.k[0]), "fy": float(message.k[4]),
                    "ppx": float(message.k[2]), "ppy": float(message.k[5]),
                    "coeffs": list(message.d), "distortion_model": str(message.distortion_model),
                    "frame_id": str(message.header.frame_id)}
        if len(info) == 2 and all(any(c + "_" in a for a, b in edges) for c in CAMERAS):
            break
    if len(info) != 2:
        raise ValueError("Missing camera_info for one of the two cameras")
    transforms = nominal_rig(info, edges)
    return {"cameras": info, "T_body_from_camera": {c: transforms[c].tolist() for c in CAMERAS},
            "extrinsics_source": "user CAD/URDF nominal depth origins + factory TF depth-to-colour",
            "extrinsics_measured": False, "downsample": downsample,
            "body_axes": "x forward, y left, z up", "camera_a": "right, upside down",
            "camera_b": "left"}


def tf_between(edges, parent, child):
    queue = deque([(parent, np.eye(4))])
    visited = set()
    while queue:
        node, transform = queue.popleft()
        if node == child:
            return transform
        if node in visited:
            continue
        visited.add(node)
        for (a, b), matrix in edges.items():
            if a == node and b not in visited:
                queue.append((b, transform @ matrix))
            if b == node and a not in visited:
                queue.append((a, transform @ np.linalg.inv(matrix)))
    raise ValueError(f"Missing factory TF from {parent} to {child}")


def nominal_rig(info, edges):
    optical_to_link = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
    result = {}
    for camera, yaw, roll, y in (("a", -30, 180, -.1178), ("b", 30, 0, .1178)):
        yaw, pitch, roll = np.radians([yaw, 10., roll])
        rx = np.array([[1., 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1., 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
        depth_pose = np.eye(4)
        depth_pose[:3, :3] = rz @ ry @ rx @ optical_to_link
        depth_pose[:3, 3] = [.2801, y, .1087]
        factory = tf_between(edges, camera + "_depth_optical_frame", info[camera]["frame_id"])
        result[camera] = depth_pose @ factory
    return result


def synchronized_frames(path, *, stride=4, downsample=2, stats=None, max_camera_skew_s=.010):
    stats = stats if stats is not None else {}
    stats.update(rgbd_pairs={c: 0 for c in CAMERAS}, paired_cameras=0, emitted=0,
                 discarded_images=0, discarded_camera_frames=0, max_camera_skew_s=0.)
    pending = {c: {s: deque() for s in ("color", "aligned_depth_to_color")} for c in CAMERAS}
    ready = {c: deque() for c in CAMERAS}
    topics = [image_topic(c, s) for c in CAMERAS for s in ("color", "aligned_depth_to_color")]
    first = None
    previous = None
    for topic, data in raw_messages(path, topics):
        _, _, camera, stream, _ = topic.split("/")
        packet = decode_image(data, downsample)
        pending[camera][stream].append(packet)
        color, depth = pending[camera]["color"], pending[camera]["aligned_depth_to_color"]
        while color and depth:
            skew = color[0].timestamp_ns - depth[0].timestamp_ns
            if abs(skew) <= 8_000_000:
                rgb, z = color.popleft(), depth.popleft()
                if rgb.array.shape[:2] != z.array.shape:
                    raise ValueError("RGB and aligned depth size mismatch")
                ready[camera].append((rgb, z))
                stats["rgbd_pairs"][camera] += 1
            else:
                (color if skew < 0 else depth).popleft()
                stats["discarded_images"] += 1
        while ready["a"] and ready["b"]:
            a, b = ready["a"][0], ready["b"][0]
            skew = a[0].timestamp_ns - b[0].timestamp_ns
            if abs(skew) <= int(max_camera_skew_s * 1e9):
                ready["a"].popleft()
                ready["b"].popleft()
                stamp = (a[0].timestamp_ns + b[0].timestamp_ns) // 2
                if previous is not None and stamp <= previous:
                    raise ValueError("Non-increasing synchronized timestamps")
                previous = stamp
                first = stamp if first is None else first
                index = stats["paired_cameras"]
                stats["paired_cameras"] += 1
                stats["max_camera_skew_s"] = max(stats["max_camera_skew_s"], abs(skew) / 1e9)
                if index % stride == 0:
                    stats["emitted"] += 1
                    yield {"index": index, "time_s": (stamp-first) / 1e9,
                           "camera_skew_s": skew / 1e9, "a": a, "b": b}
            else:
                ready["a" if skew < 0 else "b"].popleft()
                stats["discarded_camera_frames"] += 1
        if max(len(q) for queues in pending.values() for q in queues.values()) > 180 or max(map(len, ready.values())) > 180:
            raise ValueError("Arrival gap exceeds bounded frame buffer")
    stats["discarded_images"] += sum(len(q) for queues in pending.values() for q in queues.values())
    stats["discarded_camera_frames"] += sum(map(len, ready.values()))


def prepare_frame(pair, camera, meta, maps=None):
    rgb, depth = pair[camera]
    info = meta["cameras"][camera]
    scale = meta["downsample"]
    intr = SimpleNamespace(**{k: info[k] / scale for k in ("fx", "fy", "ppx", "ppy")})
    if np.any(np.asarray(info["coeffs"]) != 0):
        if info["distortion_model"] != "plumb_bob":
            raise ValueError("Nonzero unsupported distortion model")
        key = camera
        if maps is None:
            maps = {}
        if key not in maps:
            k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
            maps[key] = cv2.initUndistortRectifyMap(k, np.asarray(info["coeffs"]), None, k,
                                                 rgb.array.shape[1::-1], cv2.CV_32FC1)
        rgb_array = cv2.remap(rgb.array, *maps[key], cv2.INTER_LINEAR)
        z_array = cv2.remap(depth.array, *maps[key], cv2.INTER_NEAREST)
    else:
        rgb_array, z_array = rgb.array, depth.array
    return SimpleNamespace(index=pair["index"], timestamp=pair["time_s"], rgb=rgb_array,
                           depth=z_array, depth_scale_m=.001 if depth.encoding == "16uc1" else 1., intrinsics=intr)
