from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory('dual_rgbd_global_mapper'))
    default_params = str(package_share / 'config' / 'mapper.yaml')
    params_file = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(
            package='dual_rgbd_global_mapper',
            executable='global_mapper',
            name='dual_rgbd_global_mapper',
            output='screen',
            parameters=[params_file],
        ),
    ])
