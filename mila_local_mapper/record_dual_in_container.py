"""Record two USB 3 D435i cameras into one MCAP bag (inside the existing image)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


MIN_FREE_BYTES = 3_000_000_000


def configure_transport(session):
    # The default Fast DDS SHM segment is smaller than a single RGB image.
    # Keep all local ROS participants on transports sized for RGB-D payloads.
    profile = session / "fastdds_profile.xml"
    profile.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>rgbd_shm</transport_id><type>SHM</type>
      <maxMessageSize>4194304</maxMessageSize>
      <segment_size>67108864</segment_size>
      <port_queue_capacity>1024</port_queue_capacity>
    </transport_descriptor>
    <transport_descriptor>
      <transport_id>local_udp</transport_id><type>UDPv4</type>
      <interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="dual_rgbd" is_default_profile="true">
    <rtps>
      <userTransports><transport_id>rgbd_shm</transport_id><transport_id>local_udp</transport_id></userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
''', encoding="utf-8")
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(profile)
    return profile


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def camera_launch(name, serial, fps):
    return ["ros2", "launch", "realsense2_camera", "rs_launch.py",
            "camera_namespace:=camera", f"camera_name:={name}", f"serial_no:=_{serial}",
            "enable_color:=true", "enable_depth:=true", "enable_infra1:=false", "enable_infra2:=false",
            f"rgb_camera.color_profile:=640x480x{fps}", f"depth_module.depth_profile:=640x480x{fps}",
            "enable_gyro:=true", "enable_accel:=true", "unite_imu_method:=0",
            "enable_sync:=true", "align_depth.enable:=true", "pointcloud.enable:=false",
            "publish_tf:=true", "wait_for_device_timeout:=20.0"]


def topic_spec():
    required = {}
    optional = []
    for name in ("a", "b"):
        prefix = f"/camera/{name}/"
        for stream in ("color", "depth", "aligned_depth_to_color"):
            image = "image_rect_raw" if stream == "depth" else "image_raw"
            required[prefix + stream + "/" + image] = "Image"
            required[prefix + stream + "/camera_info"] = "CameraInfo"
        required[prefix + "gyro/sample"] = "Imu"
        required[prefix + "accel/sample"] = "Imu"
        optional += [prefix + item for item in (
            "color/metadata", "depth/metadata", "gyro/metadata", "accel/metadata",
            "gyro/imu_info", "accel/imu_info", "extrinsics/depth_to_color",
            "extrinsics/depth_to_gyro", "extrinsics/depth_to_accel")]
    required["/tf_static"] = "TFMessage"
    return required, optional


def camera_inventory(serials):
    import pyrealsense2 as rs

    context = rs.context()
    result = {}
    for device in context.query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        if serial not in serials:
            continue
        fields = {"name": rs.camera_info.name, "serial": rs.camera_info.serial_number,
                  "usb_type": rs.camera_info.usb_type_descriptor,
                  "firmware": rs.camera_info.firmware_version,
                  "physical_port": rs.camera_info.physical_port}
        item = {key: device.get_info(field) for key, field in fields.items() if device.supports(field)}
        if not item.get("usb_type", "").startswith("3"):
            raise RuntimeError(f"Camera {serial} is not connected over USB 3: {item}")
        item["hardware_sync_modes"] = []
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.inter_cam_sync_mode):
                item["hardware_sync_modes"].append(sensor.get_option(rs.option.inter_cam_sync_mode))
            if sensor.is_depth_sensor():
                item["depth_scale_m"] = sensor.as_depth_sensor().get_depth_scale()
        result[serial] = item
    missing = set(serials) - set(result)
    if missing:
        raise RuntimeError(f"Missing USB 3 camera(s): {sorted(missing)}")
    # All SDK handles leave scope before separate ROS drivers open the devices.
    return result


def stop_process(process, *, recorder=False):
    if process is None:
        return
    if process.poll() is None:
        try:
            # ros2 launch forwards SIGINT to its children itself. Signalling its
            # entire group would deliver SIGINT twice to the RealSense node.
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    if recorder:
        # Never SIGKILL a recorder while it is writing indexes/metadata.
        process.wait()
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def summarize_bag(metadata, required, fps):
    counts = {entry["topic_metadata"]["name"]: entry["message_count"]
              for entry in metadata["topics_with_message_count"]}
    missing = [topic for topic in required if counts.get(topic, 0) == 0]
    if missing:
        raise RuntimeError(f"Bag is missing required messages: {missing}")
    duration = metadata["duration"]["nanoseconds"] / 1e9
    if duration <= 0:
        raise RuntimeError("Bag duration is zero")
    rates = {topic: round(counts[topic] / duration, 2)
             for topic, kind in required.items() if kind == "Image"}
    warnings = [f"Low image rate {topic}: {rate} Hz (requested {fps})"
                for topic, rate in rates.items() if rate < fps * 0.8]
    for stream in ("color/image_raw", "depth/image_rect_raw", "aligned_depth_to_color/image_raw"):
        a, b = (counts[f"/camera/{name}/{stream}"] for name in ("a", "b"))
        if abs(a - b) / max(a, b) > 0.05:
            warnings.append(f"Camera A/B frame count mismatch for {stream}: {a}/{b}")
    return {"recorded_topic_counts": counts, "bag_duration_s": duration,
            "image_rates_hz": rates, "quality_warnings": warnings,
            "timestamp_synchronization_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Relative session directory under /recordings")
    parser.add_argument("--serial-a", required=True)
    parser.add_argument("--serial-b", required=True)
    parser.add_argument("--fps", type=int, choices=(30, 60), default=60)
    parser.add_argument("--duration", type=float, default=0, help="Diagnostic seconds after ready; 0 waits for stop")
    args = parser.parse_args()
    if args.serial_a == args.serial_b:
        parser.error("Camera serial numbers must differ")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be nonnegative and finite")
    root = Path("/recordings").resolve()
    relative = Path(args.session)
    session = (root / relative).resolve()
    if relative.is_absolute() or not session.is_relative_to(root) or session == root:
        parser.error("--session must remain under /recordings")
    session.mkdir(parents=True, exist_ok=True)
    bag = session / "bag"
    if bag.exists():
        parser.error("Session already contains a bag; refusing to overwrite")
    os.environ["ROS_LOG_DIR"] = str(session / "ros_logs")
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    def stopping():
        return stopped or (session / "stop.request").exists()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    status = {"status": "starting", "created_utc": datetime.now(timezone.utc).isoformat(),
              "fps": args.fps, "width": 640, "height": 480,
              "camera_a_serial": args.serial_a, "camera_b_serial": args.serial_b,
              "physical_left_right": {"a": "right", "b": "left"}, "hardware_sync_settings_modified": False,
              "mounting_roll_about_view_axis_deg": {"a": 180.0, "b": 0.0},
              "mounting_orientation_source": "user description; full extrinsics not calibrated",
              "streams": "two cameras: RGB, raw depth, aligned depth, raw gyro/accel, calibration, TF",
              "notes": "Simultaneous recording does not verify cross-camera exposure synchronization."}
    write_json(session / "session.json", status)
    drivers = []
    recorder = node = rclpy = None
    logs = []
    subscriptions = []
    started = None
    try:
        if stopping():
            status["status"] = "cancelled"
            return 0
        if shutil.disk_usage(root).free < MIN_FREE_BYTES * 2:
            raise RuntimeError("Need at least 6 GB free before starting two cameras")
        status["cameras"] = camera_inventory((args.serial_a, args.serial_b))
        status["transport_profile"] = str(configure_transport(session))
        import rclpy
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image, CameraInfo, Imu
        from tf2_msgs.msg import TFMessage
        import yaml

        kinds = {"Image": Image, "CameraInfo": CameraInfo, "Imu": Imu, "TFMessage": TFMessage}
        required, optional = topic_spec()
        counts = dict.fromkeys(required, 0)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("dual_recording_readiness_check")
        for topic, kind in required.items():
            def receive(_message, topic=topic):
                counts[topic] += 1
            qos = (QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
                   if topic == "/tf_static" else qos_profile_sensor_data)
            subscriptions.append(node.create_subscription(kinds[kind], topic, receive, qos))
        commands = [camera_launch(name, serial, args.fps)
                    for name, serial in (("a", args.serial_a), ("b", args.serial_b))]
        status["launch_commands"] = commands
        write_json(session / "session.json", status)
        for name, command in zip(("a", "b"), commands):
            log = (session / f"driver_{name}.log").open("w")
            logs.append(log)
            drivers.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True))
            # Starting both SDK contexts together races while claiming USB
            # interfaces. Open the next camera only after this one's streams work.
            camera_topics = [topic for topic in required if topic.startswith(f"/camera/{name}/")]
            deadline = time.monotonic() + 70
            while not all(counts[topic] for topic in camera_topics):
                if stopping():
                    status["status"] = "cancelled"
                    return 0
                driver_text = (session / f"driver_{name}.log").read_text(errors="replace")
                if (drivers[-1].poll() is not None or "process has died" in driver_text
                        or time.monotonic() >= deadline):
                    missing = [topic for topic in camera_topics if not counts[topic]]
                    raise RuntimeError(f"Camera {name.upper()} failed to start: {missing}; see driver_{name}.log")
                rclpy.spin_once(node, timeout_sec=0.1)
        deadline = time.monotonic() + 70
        while not all(counts.values()):
            if stopping():
                status["status"] = "cancelled"
                return 0
            if any(driver.poll() is not None for driver in drivers) or time.monotonic() >= deadline:
                raise RuntimeError("No data on required topics: " + str([k for k, v in counts.items() if not v]))
            rclpy.spin_once(node, timeout_sec=0.1)
        warmup = time.monotonic() + 3
        while time.monotonic() < warmup:
            if stopping():
                status["status"] = "cancelled"
                return 0
            rclpy.spin_once(node, timeout_sec=0.1)
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        subscriptions.clear()
        # No compressed/theora subscriptions: raw topics avoid unnecessary encoders.
        overrides = {topic: {"history": "keep_last", "depth": 30 if kind == "Image" else 5,
                             "reliability": "best_effort", "durability": "volatile"}
                     for topic, kind in required.items()}
        overrides["/tf_static"] = {"history": "keep_last", "depth": 100,
                                    "reliability": "reliable", "durability": "transient_local"}
        qos_file = session / "recorder_qos.yaml"
        qos_file.write_text(yaml.safe_dump(overrides), encoding="utf-8")
        command = ["ros2", "bag", "record", "-s", "mcap", "-o", str(bag),
                   "--max-cache-size", "268435456", "-b", "2000000000",
                   "--qos-profile-overrides-path", str(qos_file)] + list(required) + optional
        status.update(status="subscribing", record_command=command)
        write_json(session / "session.json", status)
        recorder_log = (session / "recorder.log").open("w")
        logs.append(recorder_log)
        recorder = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=recorder_log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 35
        while True:
            if stopping():
                status["status"] = "cancelled"
                return 0
            text = (session / "recorder.log").read_text(errors="replace")
            if all(f"Subscribed to topic '{topic}'" in text for topic in required):
                break
            if (recorder.poll() is not None or any(d.poll() is not None for d in drivers)
                    or time.monotonic() >= deadline):
                raise RuntimeError("Recorder could not subscribe to all camera topics; inspect logs")
            time.sleep(0.1)
        started = time.monotonic()
        status.update(status="recording", recording_started_utc=datetime.now(timezone.utc).isoformat())
        write_json(session / "session.json", status)
        next_disk_check = started
        while not stopping() and (not args.duration or time.monotonic() - started < args.duration):
            if recorder.poll() is not None or any(driver.poll() is not None for driver in drivers):
                raise RuntimeError("Camera driver or recorder stopped unexpectedly; inspect logs")
            if time.monotonic() >= next_disk_check:
                if shutil.disk_usage(root).free < MIN_FREE_BYTES:
                    status["stop_reason"] = "low_disk_space"
                    break
                next_disk_check = time.monotonic() + 1
            time.sleep(0.1)
        status.update(status="finalizing", elapsed_recording_s=time.monotonic() - started)
        write_json(session / "session.json", status)
        stop_process(recorder, recorder=True)
        if recorder.returncode not in (0, -signal.SIGINT, 130):
            raise RuntimeError(f"Recorder failed with exit code {recorder.returncode}")
        metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
        status.update(summarize_bag(metadata, required, args.fps))
        if status.get("stop_reason") == "low_disk_space":
            status["quality_warnings"].append("Recording ended early: low disk space")
        # Full bag info is a file, never a wall of console output.
        info = subprocess.run(["ros2", "bag", "info", str(bag)], capture_output=True, text=True, check=True)
        (session / "bag_info.txt").write_text(info.stdout, encoding="utf-8")
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=str(error))
    finally:
        try:
            stop_process(recorder, recorder=True)
            for driver in drivers:
                stop_process(driver)
            if node is not None:
                node.destroy_node()
            if rclpy is not None and rclpy.ok():
                rclpy.shutdown()
            for log in logs:
                log.close()
            checksums = {}
            for path in sorted(bag.glob("*")):
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                            digest.update(chunk)
                    checksums[str(path.relative_to(session))] = digest.hexdigest()
            write_json(session / "checksums.json", checksums)
        except Exception as error:
            status.update(status="failed", error=f"Finalization failed: {error}")
        if started is not None:
            status.setdefault("elapsed_recording_s", time.monotonic() - started)
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(session / "session.json", status)
        uid, gid = int(os.environ.get("RECORDING_UID", os.getuid())), int(os.environ.get("RECORDING_GID", os.getgid()))
        for path in session.rglob("*"):
            os.chown(path, uid, gid)
        os.chown(session, uid, gid)
    return 0 if status["status"] in ("complete", "cancelled") else 1


if __name__ == "__main__":
    sys.exit(main())
