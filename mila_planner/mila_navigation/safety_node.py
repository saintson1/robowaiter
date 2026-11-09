#!/usr/bin/env python3
"""Final velocity gate; the sole autonomous publisher on /cmd_vel."""
from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .controller import VelocityCommand
from .safety import SafetyConfig, SafetySupervisor


class SafetyNode(Node):
    def __init__(self):
        super().__init__('mila_safety_supervisor')
        defaults = {
            'navigation_enabled': False,
            'publish_hz': 20.0, 'command_timeout_s': 0.25,
            'odometry_timeout_s': 0.30, 'status_timeout_s': 1.5,
            'minimum_linear_mps': 0.04, 'maximum_linear_mps': 0.15,
            'maximum_acceleration_mps2': 0.10,
            'maximum_deceleration_mps2': 0.20,
            'minimum_turning_radius_m': 0.70,
        }
        self.params = {name: self.declare_parameter(name, value).value
                       for name, value in defaults.items()}
        self.supervisor = SafetySupervisor(SafetyConfig(
            min_turn_radius=float(self.params['minimum_turning_radius_m']),
            max_speed=float(self.params['maximum_linear_mps']),
            max_acceleration=float(self.params['maximum_acceleration_mps2']),
            max_deceleration=float(self.params['maximum_deceleration_mps2']),
            command_timeout=float(self.params['command_timeout_s']),
            odometry_timeout=float(self.params['odometry_timeout_s'])))
        self.odom_received = None
        self.status_received = None
        self.odom_state = None
        self.reason = 'starting'
        self.last_output = VelocityCommand()
        self.create_subscription(Twist, '/nav_cmd_vel', self.on_command, 1)
        self.create_subscription(Odometry, '/dual_odometry/odom', self.on_odom, 10)
        self.create_subscription(String, '/dual_odometry/status', self.on_status, 5)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.status_pub = self.create_publisher(String, '/safety/status', 5)
        self.create_timer(1.0 / float(self.params['publish_hz']), self.tick)
        self.create_timer(1.0, self.publish_status)

    def on_command(self, message):
        self.supervisor.submit_command(
            VelocityCommand(float(message.linear.x), float(message.angular.z)),
            time.monotonic())

    def on_odom(self, _message):
        self.odom_received = time.monotonic()

    def on_status(self, message):
        try:
            self.odom_state = json.loads(message.data)['state']
            self.status_received = time.monotonic()
        except Exception:
            self.odom_state = 'invalid_status'

    def tick(self):
        now = time.monotonic()
        if not bool(self.params['navigation_enabled']):
            self.reason = 'navigation_disabled'
            self.last_output = VelocityCommand()
            output = Twist()
            self.publisher.publish(output)
            return
        status_fresh = (self.status_received is not None
                        and now - self.status_received <= float(self.params['status_timeout_s']))
        if self.odom_received is not None:
            self.supervisor.update_odometry(
                self.odom_received, valid=status_fresh and self.odom_state == 'tracking')
        decision = self.supervisor.evaluate(now)
        self.reason = decision.reason
        self.last_output = decision.command
        output = Twist()
        output.linear.x = decision.command.linear
        output.angular.z = decision.command.angular
        self.publisher.publish(output)

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps({
            'state': self.reason,
            'linear_output_mps': self.last_output.linear,
            'odometry_state': self.odom_state,
        }, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SafetyNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            zero = Twist()
            for _ in range(3):
                node.publisher.publish(zero)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

