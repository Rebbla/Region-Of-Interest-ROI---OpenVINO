import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_pkg')

    return LaunchDescription([
        Node(
            package='vision_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[os.path.join(pkg_share, 'config', 'vision.yaml')],
        ),
    ])
