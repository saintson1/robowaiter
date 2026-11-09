from glob import glob
from setuptools import find_packages, setup


setup(
    name='mila_navigation',
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/mila_navigation']),
        ('share/mila_navigation', ['package.xml']),
        ('share/mila_navigation/config', glob('config/*.yaml')),
        ('share/mila_navigation/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=False,
    maintainer='MILA project',
    maintainer_email='maintainer@example.invalid',
    description='Forward-only global navigation and velocity safety',
    license='Proprietary',
    entry_points={'console_scripts': [
        'saved_map_server = mila_navigation.map_server_node:main',
        'navigation_node = mila_navigation.navigation_node:main',
        'safety_node = mila_navigation.safety_node:main',
    ]},
)

