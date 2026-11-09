from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('frame_stride', default_value='2', choices=['1', '2', '4', '8']),
        Node(package='dual_realsense_odometry', executable='dual_odometry', output='screen',
             parameters=[{'frame_stride': ParameterValue(LaunchConfiguration('frame_stride'), value_type=int)}]),
    ])
