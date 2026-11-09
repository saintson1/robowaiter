from glob import glob
from setuptools import find_packages, setup


package_name = 'dual_rgbd_global_mapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=('test',)),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='ivan',
    maintainer_email='ivan@localhost',
    description='Odometry-based global occupancy mapping from two RGB-D cameras.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'global_mapper = dual_rgbd_global_mapper.node:main',
        ],
    },
)
