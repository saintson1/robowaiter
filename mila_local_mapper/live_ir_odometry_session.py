"""Launch two RealSense IR1/depth streams and the live IR odometry node."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from record_dual_in_container import configure_transport, stop_process


# Camera B rejects its advertised IR1+depth 30 Hz combination and is more
# sensitive to concurrent USB startup. At the verified 60 Hz profile both
# devices work independently; claim B first, then A.
SERIALS = (('b', '920312072983'), ('a', '152222073087'))


def ir_camera_launch(name, serial, fps):
    """RealSense launch command with RGB disabled and rectified IR1 enabled."""
    return ['ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'camera_namespace:=camera', f'camera_name:={name}', f'serial_no:=_{serial}',
            'enable_color:=false', 'enable_depth:=true',
            'enable_infra:=false', 'enable_infra1:=true', 'enable_infra2:=false',
            f'depth_module.depth_profile:=640x480x{fps}',
            f'depth_module.infra_profile:=640x480x{fps}',
            'depth_module.enable_auto_exposure:=true',
            'enable_gyro:=false', 'enable_accel:=false', 'unite_imu_method:=0',
            'enable_sync:=true', 'align_depth.enable:=false', 'pointcloud.enable:=false',
            'publish_tf:=true', 'wait_for_device_timeout:=20.0']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-stride', type=int, choices=(1, 2, 4, 8), default=2,
                        help='2 targets 30 odometry measurements/s from the 60 Hz streams')
    parser.add_argument('--fps', type=int, choices=(30, 60), default=60,
                        help='IR/depth rate; camera B is verified only at 60 Hz')
    parser.add_argument('--full-resolution', action='store_true',
                        help='Track at 640x480 instead of the stable 320x240 default')
    parser.add_argument('--node-only', action='store_true',
                        help='Subscribe to already running IR1/depth drivers')
    args, ros_arguments = parser.parse_known_args()
    if ros_arguments and ros_arguments[0] != '--ros-args':
        parser.error('Additional node arguments must start with --ros-args')

    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    logdir = Path('/live_logs') / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                                   + f'_{os.getpid()}')
    logdir.mkdir(parents=True)
    os.environ['ROS_LOG_DIR'] = str(logdir / 'ros_logs')
    configure_transport(logdir)
    drivers, logs = [], []
    watcher = odometry = None
    ros_initialized = False
    try:
        if not args.node_only:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.signals import SignalHandlerOptions
            from sensor_msgs.msg import CameraInfo, Image

            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            ros_initialized = True
            watcher = rclpy.create_node('live_ir_camera_startup_check')
            counts = {}
            for camera in ('a', 'b'):
                for suffix, kind in (('infra1/image_rect_raw', Image),
                                     ('depth/image_rect_raw', Image),
                                     ('infra1/camera_info', CameraInfo)):
                    topic = f'/camera/{camera}/{suffix}'
                    counts[topic] = 0

                    def receive(_message, topic=topic):
                        counts[topic] += 1

                    watcher.create_subscription(kind, topic, receive, qos_profile_sensor_data)

            for camera, serial in SERIALS:
                print(f'Starting camera {camera.upper()} with IR1/depth 640x480@{args.fps}...',
                      flush=True)
                log = (logdir / f'driver_{camera}.log').open('w')
                logs.append(log)
                driver = subprocess.Popen(ir_camera_launch(camera, serial, args.fps), stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                drivers.append(driver)
                deadline = time.monotonic() + 70
                camera_topics = [topic for topic in counts if topic.startswith(f'/camera/{camera}/')]
                while not all(counts[topic] for topic in camera_topics):
                    if stopped:
                        return 0
                    text = (logdir / f'driver_{camera}.log').read_text(errors='replace')
                    if (driver.poll() is not None or 'process has died' in text
                            or time.monotonic() > deadline):
                        missing = [topic for topic in camera_topics if not counts[topic]]
                        raise RuntimeError(f'IR camera {camera.upper()} not ready: {missing}; '
                                           f'inspect {logdir}/driver_{camera}.log')
                    rclpy.spin_once(watcher, timeout_sec=.1)
            watcher.destroy_node()
            watcher = None
            rclpy.shutdown()
            ros_initialized = False
        if stopped:
            return 0
        command = [sys.executable, '-B', '/live_odometry/ros_dual_ir_odometry.py',
                   '--ros-args', '-p', f'frame_stride:={args.frame_stride}',
                   '-p', f'estimator_downsample:={1 if args.full_resolution else 2}'] + ros_arguments
        print(f'Live IR odometry; fps={args.fps}; frame_stride={args.frame_stride}; '
              f'estimator={"640x480" if args.full_resolution else "320x240"}; '
              f'Ctrl+C to stop. Logs: {logdir}', flush=True)
        odometry = subprocess.Popen(command, start_new_session=True)
        while not stopped:
            if odometry.poll() is not None:
                return odometry.returncode
            if any(driver.poll() is not None for driver in drivers):
                raise RuntimeError('An IR camera driver exited; see driver logs')
            time.sleep(.1)
        return 0
    except Exception as error:
        print('ERROR: ' + str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        stop_process(odometry)
        for driver in drivers:
            stop_process(driver)
        if watcher is not None:
            watcher.destroy_node()
        if ros_initialized:
            rclpy.shutdown()
        for log in logs:
            log.close()
        uid = int(os.environ.get('LIVE_UID', os.getuid()))
        gid = int(os.environ.get('LIVE_GID', os.getgid()))
        for path in [logdir] + list(logdir.rglob('*')):
            os.chown(path, uid, gid)


if __name__ == '__main__':
    raise SystemExit(main())
