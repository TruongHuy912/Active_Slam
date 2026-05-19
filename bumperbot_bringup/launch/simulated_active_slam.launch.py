import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def package_launch(package_name, *relative_path):
    return PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(package_name), *relative_path)
    )


def generate_launch_description():
    world_name = LaunchConfiguration("world_name")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    active_slam_params_file = LaunchConfiguration("active_slam_params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    algorithm_mode = LaunchConfiguration("algorithm_mode")
    planner_validation_period_sec = LaunchConfiguration("planner_validation_period_sec")
    enable_navigation = LaunchConfiguration("enable_navigation")

    active_slam_share = get_package_share_directory("bumperbot_active_slam")
    bringup_share = get_package_share_directory("bumperbot_bringup")

    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="small_house",
        description="Gazebo world name without .world extension.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock.",
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="true",
        description="Start RViz with the Active SLAM debug layout.",
    )
    active_slam_params_file_arg = DeclareLaunchArgument(
        "active_slam_params_file",
        default_value=os.path.join(active_slam_share, "config", "active_slam.yaml"),
        description="Parameter file for bumperbot_active_slam active_slam_node.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(bringup_share, "rviz", "active_slam_debug.rviz"),
        description="RViz config for Active SLAM, Nav2, costmap, footprint, and scan debugging.",
    )
    algorithm_mode_arg = DeclareLaunchArgument(
        "algorithm_mode",
        default_value="aslam_original",
        description="Active SLAM algorithm mode: aslam_original or legacy_nav2_safe.",
    )
    planner_validation_period_sec_arg = DeclareLaunchArgument(
        "planner_validation_period_sec",
        default_value="0.0",
        description="Override Active SLAM planner validation period in seconds.",
    )
    enable_navigation_arg = DeclareLaunchArgument(
        "enable_navigation",
        default_value="true",
        description="Enable Active SLAM NavigateToPose goal sending.",
    )

    gazebo = IncludeLaunchDescription(
        package_launch("bumperbot_description", "launch", "gazebo.launch.py"),
        launch_arguments={"world_name": world_name}.items(),
    )

    controller = IncludeLaunchDescription(
        package_launch("bumperbot_controller", "launch", "controller.launch.py"),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_simple_controller": "False",
            "use_python": "False",
        }.items(),
    )

    slam_toolbox = IncludeLaunchDescription(
        package_launch("bumperbot_mapping", "launch", "slam.launch.py"),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    navigation = IncludeLaunchDescription(
        package_launch("bumperbot_navigation", "launch", "navigation.launch.py"),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    active_slam = Node(
        package="bumperbot_active_slam",
        executable="active_slam_node",
        name="active_slam_explorer",
        output="screen",
        parameters=[
            active_slam_params_file,
            {"use_sim_time": use_sim_time},
            {"algorithm_mode": algorithm_mode},
            {"planner_validation_period_sec": ParameterValue(planner_validation_period_sec, value_type=float)},
            {"enable_navigation": ParameterValue(enable_navigation, value_type=bool)},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="active_slam_debug_rviz",
        arguments=["-d", rviz_config],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        world_name_arg,
        use_sim_time_arg,
        launch_rviz_arg,
        active_slam_params_file_arg,
        rviz_config_arg,
        algorithm_mode_arg,
        planner_validation_period_sec_arg,
        enable_navigation_arg,
        gazebo,
        controller,
        TimerAction(period=3.0, actions=[slam_toolbox]),
        TimerAction(period=6.0, actions=[navigation]),
        TimerAction(period=10.0, actions=[active_slam]),
        TimerAction(period=12.0, actions=[rviz]),
    ])
