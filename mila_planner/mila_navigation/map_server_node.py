#!/usr/bin/env python3
"""Small ROS 2 PGM/YAML map server without a Nav2 runtime dependency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from .map_io import load_ros_map


class SavedMapServer(Node):
    def __init__(self):
        super().__init__('mila_saved_map_server')
        read_only = ParameterDescriptor(read_only=True)
        yaml_file = self.declare_parameter('yaml_file', '', read_only).value
        self.frame_id = self.declare_parameter('frame_id', 'map', read_only).value
        if not yaml_file:
            raise ValueError('yaml_file is required')
        self.grid, self.config = load_ros_map(Path(yaml_file).expanduser().resolve())
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(OccupancyGrid, '/map', qos)
        self.create_timer(1.0, self.publish_map)
        self.publish_map()
        self.get_logger().info(
            f"Loaded {yaml_file}: {self.grid.shape[1]}x{self.grid.shape[0]} at "
            f"{float(self.config['resolution']):.3f} m/cell")

    def publish_map(self):
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.info.map_load_time = message.header.stamp
        message.info.resolution = float(self.config['resolution'])
        message.info.width = self.grid.shape[1]
        message.info.height = self.grid.shape[0]
        origin = self.config['origin']
        message.info.origin.position.x = float(origin[0])
        message.info.origin.position.y = float(origin[1])
        yaw = float(origin[2])
        message.info.origin.orientation.z = float(np.sin(yaw / 2.0))
        message.info.origin.orientation.w = float(np.cos(yaw / 2.0))
        message.data = self.grid.ravel().astype(int).tolist()
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SavedMapServer()
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

