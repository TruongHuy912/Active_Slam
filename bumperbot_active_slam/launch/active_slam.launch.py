from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_navigation = LaunchConfiguration("enable_navigation")
    global_costmap_topic = LaunchConfiguration("global_costmap_topic")
    algorithm_mode = LaunchConfiguration("algorithm_mode")
    config_file = PathJoinSubstitution(
        [
            FindPackageShare("bumperbot_active_slam"),
            "config",
            "active_slam.yaml",
        ]
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true.",
    )
    enable_navigation_arg = DeclareLaunchArgument(
        "enable_navigation",
        default_value="true",
        description="Send selected frontiers to Nav2 NavigateToPose if true.",
    )
    global_costmap_topic_arg = DeclareLaunchArgument(
        "global_costmap_topic",
        default_value="/global_costmap/costmap",
        description="Nav2 global costmap OccupancyGrid topic used for frontier safety filtering.",
    )
    algorithm_mode_arg = DeclareLaunchArgument(
        "algorithm_mode",
        default_value="aslam_original",
        description="Active SLAM algorithm mode: aslam_original or legacy_nav2_safe.",
    )

    active_slam_node = Node(
        package="bumperbot_active_slam",
        executable="active_slam_node",
        name="active_slam_explorer",
        output="screen",
        parameters=[
            config_file,
            {"use_sim_time": use_sim_time},
            {"enable_navigation": ParameterValue(enable_navigation, value_type=bool)},
            {"global_costmap_topic": global_costmap_topic},
            {"algorithm_mode": algorithm_mode},
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_navigation_arg,
        global_costmap_topic_arg,
        algorithm_mode_arg,
        active_slam_node,
    ])
