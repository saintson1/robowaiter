"""Short end-to-end load check with the original ROS bag, no physical cameras."""
import argparse
import json
import signal
import subprocess
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from ros_dual_odometry import DualOdometryNode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--duration', type=float, default=10.)
    args = parser.parse_args()
    rclpy.init()
    node = DualOdometryNode(parameter_overrides=[Parameter('frame_stride', value=args.stride),
                                               Parameter('console_status', value=False)])
    probe = Node('live_replay_observer')
    statuses, odoms = [], []
    probe.create_subscription(String, '/dual_odometry/status', lambda m: statuses.append(json.loads(m.data)), 20)
    probe.create_subscription(Odometry, '/dual_odometry/odom', odoms.append, 20)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(probe)
    topics = ['/tf_static']+[f'/camera/{c}/{s}' for c in ('a', 'b') for s in
        ('color/image_raw', 'aligned_depth_to_color/image_raw', 'aligned_depth_to_color/camera_info')]
    with open('/tmp/live_bag_play.log', 'w') as log:
        player = subprocess.Popen(['ros2', 'bag', 'play', args.bag, '--read-ahead-queue-size', '32',
                                   '--delay', '1', '--topics']+topics, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while time.monotonic()-started < args.duration:
                executor.spin_once(timeout_sec=.02)
                if player.poll() is not None:
                    raise RuntimeError('Bag player exited: '+open('/tmp/live_bag_play.log').read()[-2000:])
            assert len(odoms) >= 10, (node.latest, node.stats, node.sampler.stats)
            tracking = [s for s in statuses if s['state'] == 'tracking']
            assert tracking, node.latest
            print(json.dumps(dict(stride=args.stride, duration_wall_s=time.monotonic()-started,
                physical_cameras_used=False, input='original bag at playback rate 1',
                odometry_messages=len(odoms), final_status=tracking[-1], pairing=node.sampler.stats,
                processing=node.stats)), flush=True)
        finally:
            player.send_signal(signal.SIGINT)
            try:
                player.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                player.kill()
                player.wait()
            node.close()
            executor.remove_node(node)
            executor.remove_node(probe)
            node.destroy_node()
            probe.destroy_node()
            executor.shutdown()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
