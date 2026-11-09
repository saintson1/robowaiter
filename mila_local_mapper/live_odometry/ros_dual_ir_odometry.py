#!/usr/bin/env python3
"""ROS 2 live two-camera IR1-depth odometry entry point."""

from ros_dual_odometry import main as _main


def main(args=None):
    _main(args=args, input_mode='ir')


if __name__ == '__main__':
    main()
