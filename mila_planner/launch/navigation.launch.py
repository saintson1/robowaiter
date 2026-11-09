from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('mila_navigation'))
    config = str(share / 'config' / 'navigation.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('map_yaml', default_value='/maps/global_map.yaml'),
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
        DeclareLaunchArgument('goal_x', default_value='0.0'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument('use_parameter_goal', default_value='false'),
        DeclareLaunchArgument('navigation_enabled', default_value='false'),
        Node(package='mila_navigation', executable='saved_map_server',
             parameters=[config, {'yaml_file': LaunchConfiguration('map_yaml')}], output='screen'),
        Node(package='mila_navigation', executable='navigation_node',
             parameters=[config, {
                 'initial_x': ParameterValue(LaunchConfiguration('initial_x'), value_type=float),
                 'initial_y': ParameterValue(LaunchConfiguration('initial_y'), value_type=float),
                 'initial_yaw': ParameterValue(LaunchConfiguration('initial_yaw'), value_type=float),
                 'goal_x': ParameterValue(LaunchConfiguration('goal_x'), value_type=float),
                 'goal_y': ParameterValue(LaunchConfiguration('goal_y'), value_type=float),
                 'use_parameter_goal': ParameterValue(
                     LaunchConfiguration('use_parameter_goal'), value_type=bool),
             }], output='screen'),
        Node(package='mila_navigation', executable='safety_node',
             parameters=[config, {'navigation_enabled': ParameterValue(
                 LaunchConfiguration('navigation_enabled'), value_type=bool)}], output='screen'),
    ])

