"""Run the existing camera launch configuration and the new live ROS node."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from record_dual_in_container import camera_launch, configure_transport, stop_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-stride', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--node-only', action='store_true', help='Subscribe to already running drivers in the same DDS domain')
    args, ros_arguments = parser.parse_known_args()
    if ros_arguments and ros_arguments[0] != '--ros-args':
        parser.error('Additional node arguments must start with --ros-args')
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    logdir = Path('/live_logs') / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}')
    logdir.mkdir(parents=True)
    os.environ['ROS_LOG_DIR'] = str(logdir/'ros_logs')
    configure_transport(logdir)
    drivers, logs = [], []
    watcher = odometry = None
    ros_initialized = False
    try:
        if not args.node_only:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.signals import SignalHandlerOptions
            from sensor_msgs.msg import Image, CameraInfo
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            ros_initialized = True
            watcher = rclpy.create_node('live_camera_startup_check')
            counts = {}
            for camera in ('a', 'b'):
                for suffix, kind in (('color/image_raw', Image), ('aligned_depth_to_color/image_raw', Image),
                                     ('aligned_depth_to_color/camera_info', CameraInfo)):
                    topic = f'/camera/{camera}/{suffix}'
                    counts[topic] = 0
                    def receive(_message, topic=topic):
                        counts[topic] += 1
                    watcher.create_subscription(kind, topic, receive, qos_profile_sensor_data)
            for camera, serial in (('a', '152222073087'), ('b', '920312072983')):
                print(f'Starting camera {camera.upper()} with existing RGB/depth 640x480@60 configuration...', flush=True)
                log = (logdir/f'driver_{camera}.log').open('w')
                logs.append(log)
                driver = subprocess.Popen(camera_launch(camera, serial, 60), stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                drivers.append(driver)
                deadline = time.monotonic()+70
                while not all(n for topic, n in counts.items() if topic.startswith(f'/camera/{camera}/')):
                    if stopped:
                        return 0
                    text = (logdir/f'driver_{camera}.log').read_text(errors='replace')
                    if driver.poll() is not None or 'process has died' in text or time.monotonic() > deadline:
                        raise RuntimeError(f'Camera {camera.upper()} not ready; inspect {logdir}/driver_{camera}.log')
                    rclpy.spin_once(watcher, timeout_sec=.1)
            watcher.destroy_node()
            watcher = None
            rclpy.shutdown()
            ros_initialized = False
        if stopped:
            return 0
        command = [sys.executable, '-B', '/live_odometry/ros_dual_odometry.py', '--ros-args',
                   '-p', f'frame_stride:={args.frame_stride}']+ros_arguments
        print(f'Live odometry; frame_stride={args.frame_stride}; Ctrl+C to stop. Logs: {logdir}', flush=True)
        odometry = subprocess.Popen(command, start_new_session=True)
        while not stopped:
            if odometry.poll() is not None:
                return odometry.returncode
            if any(driver.poll() is not None for driver in drivers):
                raise RuntimeError('A camera driver exited; see driver logs')
            time.sleep(.1)
        return 0
    except Exception as error:
        print('ERROR: '+str(error), file=sys.stderr, flush=True)
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
        uid, gid = int(os.environ.get('LIVE_UID', os.getuid())), int(os.environ.get('LIVE_GID', os.getgid()))
        for path in [logdir]+list(logdir.rglob('*')):
            os.chown(path, uid, gid)


if __name__ == '__main__':
    raise SystemExit(main())
