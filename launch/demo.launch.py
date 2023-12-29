from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    v4l2_camera_node = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="v4l2_camera_node",
        output="screen",
    )

    detector_node = Node(
        package="ultralytics_ros",
        executable="detector",
        name="detector",
        output="screen",
    )

    rqt_image_view_node = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_image_view",
    )

    return LaunchDescription([v4l2_camera_node, detector_node, rqt_image_view_node])
