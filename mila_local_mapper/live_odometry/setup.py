from glob import glob
from setuptools import setup

setup(name='dual_realsense_odometry', version='0.1.0',
      py_modules=['ros_dual_odometry', 'realtime_dual_core', 'dual_rgbd_odometry',
                  'dual_rgbd_bag', 'sparse_rgbd_odometry', 'ros_dual_ir_odometry',
                  'realtime_dual_ir_core'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/dual_realsense_odometry']),
                  ('share/dual_realsense_odometry', ['package.xml']),
                  ('share/dual_realsense_odometry/launch', glob('launch/*.launch.py')),
                  ('share/dual_realsense_odometry/config', glob('config/*'))],
      install_requires=['setuptools', 'numpy'], zip_safe=False,
      maintainer='Project maintainer', maintainer_email='maintainer@example.invalid',
      description='Live two-camera RGB-D odometry', license='Proprietary',
      entry_points={'console_scripts': [
          'dual_odometry = ros_dual_odometry:main',
          'dual_ir_odometry = ros_dual_ir_odometry:main']})
