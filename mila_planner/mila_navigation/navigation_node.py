#!/usr/bin/env python3
"""Global planning, map/odom anchoring and forward-only path following."""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


def wrap_angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def compose(a, b):
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (a[0] + ca * b[0] - sa * b[1],
            a[1] + sa * b[0] + ca * b[1], wrap_angle(a[2] + b[2]))


def inverse(pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (-c * pose[0] - s * pose[1],
            s * pose[0] - c * pose[1], wrap_angle(-pose[2]))


class NavigationNode(Node):
    """ROS adapter for the pure-Python planner and controller backends."""

    def __init__(self):
        super().__init__('mila_navigation')
        defaults = {
            'initial_x': 0.0, 'initial_y': 0.0, 'initial_yaw': 0.0,
            'use_parameter_initial_pose': True,
            'goal_x': 0.0, 'goal_y': 0.0, 'use_parameter_goal': False,
            'control_hz': 20.0, 'goal_tolerance_m': 0.15,
            'odom_timeout_s': 0.30, 'status_timeout_s': 1.5,
            'footprint_length_m': 0.60, 'footprint_width_m': 0.45,
            'safety_margin_m': 0.30, 'min_turning_radius_m': 0.70,
            'unknown_cost': 4.0, 'occupied_threshold': 65,
            'planner_heuristic_weight': 5.0,
            'use_local_costmap': False,
        }
        self.params = {name: self.declare_parameter(name, value).value
                       for name, value in defaults.items()}
        if float(self.params['control_hz']) <= 0:
            raise ValueError('control_hz must be positive')
        self.lock = threading.RLock()
        self.map_message = None
        self.latest_odom = None
        self.odom_received = None
        self.odom_state = None
        self.status_received = None
        self.map_from_odom = None
        self.pending_initial = ((float(self.params['initial_x']), float(self.params['initial_y']),
                                 float(self.params['initial_yaw']))
                                if self.params['use_parameter_initial_pose'] else None)
        self.goal = ((float(self.params['goal_x']), float(self.params['goal_y']))
                     if self.params['use_parameter_goal'] else None)
        self.path = []
        self.plan_error = None
        self.last_control_time = time.monotonic()
        from .controller import ControllerConfig, PurePursuitController
        self.controller = PurePursuitController(ControllerConfig(
            min_turn_radius=float(self.params['min_turning_radius_m']),
            goal_tolerance=float(self.params['goal_tolerance_m'])))
        self.broadcaster = TransformBroadcaster(self)

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.create_subscription(Odometry, '/dual_odometry/odom', self.on_odom, 10)
        self.create_subscription(String, '/dual_odometry/status', self.on_odom_status, 5)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_initial_pose, 5)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 5)
        self.path_pub = self.create_publisher(Path, '/navigation/global_plan', map_qos)
        self.command_pub = self.create_publisher(Twist, '/nav_cmd_vel', 1)
        self.status_pub = self.create_publisher(String, '/navigation/status', 5)
        self.create_timer(1.0 / float(self.params['control_hz']), self.control_cycle)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info('Waiting for saved /map, odometry anchor and /goal_pose')

    @staticmethod
    def odom_pose(message):
        p = message.pose.pose
        return p.position.x, p.position.y, yaw_from_quaternion(p.orientation)

    def on_map(self, message):
        if message.header.frame_id != 'map':
            self.plan_error = f'expected map frame, got {message.header.frame_id!r}'
            return
        with self.lock:
            self.map_message = message
        self.replan()

    def on_odom(self, message):
        if message.header.frame_id != 'odom' or message.child_frame_id != 'base_link':
            self.plan_error = 'unexpected odometry frames'
            return
        anchored_now = False
        with self.lock:
            self.latest_odom = message
            self.odom_received = time.monotonic()
            if self.pending_initial is not None:
                self.map_from_odom = compose(self.pending_initial, inverse(self.odom_pose(message)))
                self.pending_initial = None
                anchored_now = True
                self.get_logger().info('Anchored saved map to current odometry')
        self.publish_map_odom_tf()
        if anchored_now:
            self.replan()

    def on_odom_status(self, message):
        try:
            state = json.loads(message.data)['state']
        except Exception as error:
            self.plan_error = f'invalid odometry status: {error}'
            return
        with self.lock:
            self.odom_state = state
            self.status_received = time.monotonic()

    def on_initial_pose(self, message):
        if message.header.frame_id not in ('', 'map'):
            self.plan_error = f'initial pose must be in map, got {message.header.frame_id!r}'
            return
        p = message.pose.pose
        desired = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self.lock:
            if self.latest_odom is None:
                self.pending_initial = desired
            else:
                self.map_from_odom = compose(desired, inverse(self.odom_pose(self.latest_odom)))
            self.path = []
        self.publish_map_odom_tf()
        self.replan()

    def on_goal(self, message):
        if message.header.frame_id not in ('', 'map'):
            self.plan_error = f'goal must be in map, got {message.header.frame_id!r}'
            return
        # Goal orientation is intentionally ignored: the robot cannot turn in place.
        with self.lock:
            self.goal = (float(message.pose.position.x), float(message.pose.position.y))
            self.path = []
        self.replan()

    def current_map_pose(self):
        if self.latest_odom is None or self.map_from_odom is None:
            return None
        return compose(self.map_from_odom, self.odom_pose(self.latest_odom))

    def _planner_backend(self):
        # Imported lazily so map/status diagnostics still run if a backend has a
        # packaging error.
        from .hybrid_astar import GridMap, HybridAStarPlanner, PlannerConfig, Pose2D
        return GridMap, HybridAStarPlanner, PlannerConfig, Pose2D

    def replan(self):
        with self.lock:
            if self.map_message is None or self.goal is None:
                return
            start = self.current_map_pose()
            if start is None:
                return
            map_message, goal = self.map_message, self.goal
        try:
            GridMap, HybridAStarPlanner, PlannerConfig, Pose2D = self._planner_backend()
            origin = map_message.info.origin
            grid = GridMap(
                np.asarray(map_message.data, dtype=np.int16).reshape(
                    map_message.info.height, map_message.info.width),
                resolution=float(map_message.info.resolution),
                origin_x=float(origin.position.x), origin_y=float(origin.position.y),
                origin_yaw=yaw_from_quaternion(origin.orientation))
            config = PlannerConfig(
                min_turning_radius_m=float(self.params['min_turning_radius_m']),
                footprint_length_m=float(self.params['footprint_length_m']),
                footprint_width_m=float(self.params['footprint_width_m']),
                safety_margin_m=float(self.params['safety_margin_m']),
                goal_tolerance_m=float(self.params['goal_tolerance_m']),
                unknown_penalty=float(self.params['unknown_cost']),
                occupied_threshold=int(self.params['occupied_threshold']),
                heuristic_weight=float(self.params['planner_heuristic_weight']))
            start_pose = Pose2D(*start)
            # Goal yaw is ignored by the search; start yaw is only used for the
            # preliminary footprint check at the requested coordinate.
            goal_pose = Pose2D(goal[0], goal[1], start[2])
            result = HybridAStarPlanner(grid, config).plan(start_pose, goal_pose)
            points = list(result)
            if not points:
                raise RuntimeError('planner returned an empty path')
            with self.lock:
                self.path = points
                self.plan_error = None
            self.publish_path()
        except Exception as error:
            with self.lock:
                self.path = []
                self.plan_error = f'{type(error).__name__}: {error}'
            self.get_logger().error('Planning failed: ' + self.plan_error)

    @staticmethod
    def point_components(point):
        if hasattr(point, 'x'):
            return float(point.x), float(point.y), float(point.yaw)
        return float(point[0]), float(point[1]), float(point[2])

    def publish_path(self):
        message = Path()
        message.header.frame_id = 'map'
        message.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            points = list(self.path)
        for point in points:
            x, y, yaw = self.point_components(point)
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = x, y
            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
            pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
            message.poses.append(pose)
        self.path_pub.publish(message)

    def publish_map_odom_tf(self):
        with self.lock:
            transform = self.map_from_odom
        if transform is None:
            return
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        message.child_frame_id = 'odom'
        message.transform.translation.x = transform[0]
        message.transform.translation.y = transform[1]
        message.transform.rotation.z = math.sin(transform[2] / 2.0)
        message.transform.rotation.w = math.cos(transform[2] / 2.0)
        self.broadcaster.sendTransform(message)

    def control_cycle(self):
        now = time.monotonic()
        dt = max(1e-3, min(0.2, now - self.last_control_time))
        self.last_control_time = now
        with self.lock:
            pose = self.current_map_pose()
            path = list(self.path)
            goal = self.goal
            odom_age = math.inf if self.odom_received is None else now - self.odom_received
            status_age = math.inf if self.status_received is None else now - self.status_received
            state = self.odom_state
        command = Twist()
        if (pose is None or not path or goal is None
                or odom_age > float(self.params['odom_timeout_s'])
                or status_age > float(self.params['status_timeout_s']) or state != 'tracking'):
            self.command_pub.publish(command)
            return
        if math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= float(self.params['goal_tolerance_m']):
            with self.lock:
                self.path = []
            self.command_pub.publish(command)
            return
        try:
            from .controller import Pose2D
            current_speed = (max(0.0, float(self.latest_odom.twist.twist.linear.x))
                             if self.latest_odom is not None else 0.0)
            velocity = self.controller.compute(
                path, Pose2D(*pose), current_speed=current_speed, dt=dt)
            command.linear.x = float(velocity.linear)
            command.angular.z = float(velocity.angular)
        except Exception as error:
            self.plan_error = f'controller: {type(error).__name__}: {error}'
        self.command_pub.publish(command)

    def publish_status(self):
        with self.lock:
            payload = {
                'map_received': self.map_message is not None,
                'anchored': self.map_from_odom is not None,
                'goal': self.goal,
                'path_points': len(self.path),
                'odometry_state': self.odom_state,
                'last_error': self.plan_error,
                'local_costmap_enabled': bool(self.params['use_local_costmap']),
            }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NavigationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

