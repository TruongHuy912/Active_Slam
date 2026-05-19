"""ROS 2 Active SLAM orchestration node.

The heavy helpers live in sibling modules so this file stays focused on ROS I/O,
TF lookup, state-machine decisions, and wiring the current algorithm together.
"""

from __future__ import annotations

import math
from typing import Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from bumperbot_active_slam.active_slam_types import CandidateGoal, ScoredFrontier
from bumperbot_active_slam.aslam_core import compute_aslam_utility
from bumperbot_active_slam.entropy_utils import (
    Cell,
    GridMeta,
    Point2D,
    bresenham,
    compute_path_entropy,
    distance_decay,
    euclidean_distance,
    grid_meta_from_map_info,
    map_to_world,
    path_occupancy_values,
    world_to_map,
)
from bumperbot_active_slam.exploration_state import ExplorationState
from bumperbot_active_slam.frontier_detector import FrontierCluster, detect_frontier_clusters
from bumperbot_active_slam.frontier_filter import (
    FrontierPointFilter,
    clusters_from_points,
    costmap_value_at,
    find_nearest_free_cell,
    information_gain,
    is_acceptable_cost,
    path_has_blocked_costmap_cell,
    path_has_occupied_cell,
)
from bumperbot_active_slam.marker_publisher import ActiveSlamMarkerPublisher
from bumperbot_active_slam.rrt_frontier_detector import RrtFrontierDetector
from bumperbot_active_slam.nav2_adapter import (
    Nav2ActionAdapter,
    goal_status_name,
    path_cells_from_poses,
    path_length,
    yaw_to_quaternion,
)


class ActiveSlamExplorer(Node):
    """Detect, score, visualize, and optionally navigate to frontiers."""

    def __init__(self) -> None:
        super().__init__("active_slam_explorer")
        self.callback_group = ReentrantCallbackGroup()

        self.declare_active_slam_parameters()
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.last_selected_cell: Optional[Cell] = None
        self.last_selected_log_time = self.get_clock().now()
        self.last_goal_send_time = self.get_clock().now()
        self.active_goal_handle = None
        self.active_goal_xy: Optional[Point2D] = None
        self.active_goal_sent_time: Optional[Time] = None
        self.navigation_status = "idle"
        self.last_rejection_counts: dict[str, int] = {}
        self._last_goal_rejection_reason = "goal_invalid"
        self._last_planner_rejection_reason = "planner_failed"
        self._last_throttle_log: dict[str, Time] = {}

        self.state = ExplorationState(
            blacklist_radius=self.blacklist_radius,
            blacklist_timeout_sec=self.blacklist_timeout_sec,
            visited_radius=self.visited_radius,
            visited_timeout_sec=self.visited_timeout_sec,
            planner_cache_radius=self.planner_cache_radius,
            planner_cache_ttl_sec=self.planner_cache_ttl_sec,
            planner_validation_period_sec=self.planner_validation_period_sec,
            post_goal_settle_time_sec=self.post_goal_settle_time_sec,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav2 = Nav2ActionAdapter(
            self,
            self.callback_group,
            global_frame=self.global_frame,
            planner_action_name=self.planner_action_name,
            navigate_action_name=self.navigate_action_name,
            planner_id=self.planner_id,
        )
        self.markers = ActiveSlamMarkerPublisher(
            self,
            global_frame=self.global_frame,
            marker_topic=self.marker_topic,
            debug_visited_goals=self.debug_visited_goals,
        )
        self.rrt_detector = RrtFrontierDetector(
            eta=self.rrt_eta,
            samples_per_cycle=self.rrt_samples_per_cycle,
            seed=self.rrt_random_seed if self.rrt_random_seed >= 0 else None,
        )
        self.frontier_filter = FrontierPointFilter(
            timeout_sec=self.frontier_buffer_timeout_sec,
            merge_radius=self.frontier_merge_radius,
            min_cluster_size=1,
        )
        self.exploration_state_name = "COLLECTING_FRONTIERS"
        self.frontier_collection_started = self.get_clock().now()

        self.create_ros_io()
        self.log_startup()

    def declare_active_slam_parameters(self) -> None:
        self.algorithm_mode = str(self.declare_parameter("algorithm_mode", "aslam_original").value)
        if self.algorithm_mode not in ("aslam_original", "legacy_nav2_safe"):
            self.get_logger().warn(f"Unknown algorithm_mode '{self.algorithm_mode}', using 'aslam_original'.")
            self.algorithm_mode = "aslam_original"

        self.map_topic = self.declare_parameter("map_topic", "/map").value
        self.global_costmap_topic = self.declare_parameter("global_costmap_topic", "/global_costmap/costmap").value
        self.global_frame = self.declare_parameter("global_frame", "map").value
        self.robot_frame = self.declare_parameter("robot_frame", "base_link").value
        self.frontier_min_cluster_size = int(self.declare_parameter("frontier_min_cluster_size", 5).value)
        self.frontier_connectivity = int(self.declare_parameter("frontier_connectivity", 8).value)
        self.lambda_decay = float(self.declare_parameter("lambda_decay", 0.6).value)
        self.unknown_probability = float(self.declare_parameter("unknown_probability", 0.1).value)
        self.known_probability = float(self.declare_parameter("known_probability", 0.45).value)
        self.w_entropy = float(self.declare_parameter("w_entropy", 1.0).value)
        self.w_distance = float(self.declare_parameter("w_distance", 1.0).value)
        self.enable_navigation = bool(self.declare_parameter("enable_navigation", True).value)
        self.decision_rate_hz = float(self.declare_parameter("decision_rate_hz", 0.2).value)
        self.goal_reached_distance = float(self.declare_parameter("goal_reached_distance", 0.3).value)
        self.goal_timeout_sec = float(self.declare_parameter("goal_timeout_sec", 180.0).value)
        self.frontier_goal_offset = float(self.declare_parameter("frontier_goal_offset", 0.3).value)
        self.goal_search_radius_cells = int(self.declare_parameter("goal_search_radius_cells", 8).value)
        self.costmap_clearing_threshold = int(self.declare_parameter("costmap_clearing_threshold", 70).value)
        self.allow_unknown_goal = bool(self.declare_parameter("allow_unknown_goal", False).value)
        self.info_radius = float(self.declare_parameter("info_radius", 1.0).value)
        self.information_threshold = float(self.declare_parameter("information_threshold", 0.45).value)
        self.planner_action_name = self.declare_parameter("planner_action_name", "compute_path_to_pose").value
        self.planner_id = self.declare_parameter("planner_id", "GridBased").value
        self.planner_validation_period_sec = float(self.declare_parameter("planner_validation_period_sec", 2.0).value)
        self.planner_request_timeout_sec = float(self.declare_parameter("planner_request_timeout_sec", 3.0).value)
        self.planner_cache_ttl_sec = float(self.declare_parameter("planner_cache_ttl_sec", 5.0).value)
        self.planner_cache_radius = float(self.declare_parameter("planner_cache_radius", 0.25).value)
        self.min_frontier_distance = float(self.declare_parameter("min_frontier_distance", 0.35).value)
        self.visited_radius = float(self.declare_parameter("visited_radius", 0.60).value)
        self.visited_timeout_sec = float(self.declare_parameter("visited_timeout_sec", 180.0).value)
        self.post_goal_settle_time_sec = float(self.declare_parameter("post_goal_settle_time_sec", 3.0).value)
        self.debug_visited_goals = bool(self.declare_parameter("debug_visited_goals", True).value)
        self.min_goal_separation = float(self.declare_parameter("min_goal_separation", 0.4).value)
        self.goal_send_period_sec = float(self.declare_parameter("goal_send_period_sec", 3.0).value)
        self.blacklist_radius = float(self.declare_parameter("blacklist_radius", 0.35).value)
        self.blacklist_timeout_sec = float(self.declare_parameter("blacklist_timeout_sec", 60.0).value)
        self.rejected_blacklist_timeout_sec = float(self.declare_parameter("rejected_blacklist_timeout_sec", 20.0).value)
        self.max_goal_retries = int(self.declare_parameter("max_goal_retries", 2).value)
        self.keep_selecting_markers_while_navigating = bool(
            self.declare_parameter("keep_selecting_markers_while_navigating", True).value
        )
        self.debug_markers = bool(self.declare_parameter("debug_markers", True).value)
        self.marker_topic = self.declare_parameter("marker_topic", "/active_slam/markers").value
        self.navigate_action_name = self.declare_parameter("navigate_action_name", "navigate_to_pose").value
        default_frontier_source = "rrt" if self.algorithm_mode == "aslam_original" else "bfs"
        self.frontier_source = str(self.declare_parameter("frontier_source", default_frontier_source).value)
        if self.frontier_source not in ("rrt", "bfs"):
            self.get_logger().warn(f"Unknown frontier_source '{self.frontier_source}', using '{default_frontier_source}'.")
            self.frontier_source = default_frontier_source
        self.rrt_eta = float(self.declare_parameter("rrt_eta", 0.5).value)
        self.rrt_samples_per_cycle = int(self.declare_parameter("rrt_samples_per_cycle", 80).value)
        self.rrt_include_global = bool(self.declare_parameter("rrt_include_global", True).value)
        self.rrt_include_local = bool(self.declare_parameter("rrt_include_local", True).value)
        self.rrt_random_seed = int(self.declare_parameter("rrt_random_seed", -1).value)
        self.publish_detected_points = bool(self.declare_parameter("publish_detected_points", True).value)
        self.detected_points_topic = self.declare_parameter("detected_points_topic", "/active_slam/detected_points").value
        self.filtered_points_topic = self.declare_parameter("filtered_points_topic", "/active_slam/filtered_points").value
        self.frontier_collection_time_sec = float(self.declare_parameter("frontier_collection_time_sec", 3.0).value)
        self.min_raw_frontiers_before_planning = int(self.declare_parameter("min_raw_frontiers_before_planning", 5).value)
        self.max_goal_search_time_sec = float(self.declare_parameter("max_goal_search_time_sec", 15.0).value)
        self.frontier_merge_radius = float(self.declare_parameter("frontier_merge_radius", 0.35).value)
        self.frontier_buffer_timeout_sec = float(self.declare_parameter("frontier_buffer_timeout_sec", 20.0).value)

        if self.algorithm_mode == "aslam_original" and self.frontier_source != "rrt":
            self.get_logger().warn("aslam_original requested with non-RRT frontier_source; this is a debug fallback, not the original detector.")

        if self.decision_rate_hz <= 0.0:
            self.get_logger().warn("decision_rate_hz must be positive; using 0.2 Hz")
            self.decision_rate_hz = 0.2

    def create_ros_io(self) -> None:
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        costmap_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
            callback_group=self.callback_group,
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.global_costmap_topic,
            self.costmap_callback,
            costmap_qos,
            callback_group=self.callback_group,
        )
        self.detected_points_pub = None
        self.filtered_points_pub = None
        if self.publish_detected_points:
            self.detected_points_pub = self.create_publisher(PointStamped, self.detected_points_topic, 10)
            self.filtered_points_pub = self.create_publisher(PointStamped, self.filtered_points_topic, 10)
        self.timer = self.create_timer(1.0 / self.decision_rate_hz, self.decision_callback, callback_group=self.callback_group)

    def log_startup(self) -> None:
        self.get_logger().info(
            f"Active SLAM node started: map_topic={self.map_topic}, costmap_topic={self.global_costmap_topic}, "
            f"tf={self.global_frame}->{self.robot_frame}, markers={self.marker_topic}, "
            f"navigation={'enabled' if self.enable_navigation else 'disabled'}, algorithm_mode={self.algorithm_mode}"
        )
        if self.algorithm_mode == "aslam_original":
            self.get_logger().info(
                "aslam_original mode: costmap/information-gain/planner filters follow aslam_rosbot; "
                f"frontier_source={self.frontier_source}; BFS is only a fallback/debug source."
            )

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def costmap_callback(self, msg: OccupancyGrid) -> None:
        self.latest_costmap = msg

    def decision_callback(self) -> None:
        if self.latest_map is None:
            self.log_throttled("waiting_map", "Waiting for occupancy grid map...", level="info")
            return

        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            return

        now = self.get_clock().now()
        self.update_navigation_state(robot_xy)
        self.state.prune(now)
        self.last_rejection_counts = {}

        map_msg = self.latest_map
        meta = grid_meta_from_map_info(map_msg.info)
        raw_clusters, candidate_clusters = self.collect_frontier_candidates(map_msg, meta, robot_xy, now)

        if self.is_navigating():
            self.exploration_state_name = "NAVIGATING"
            self.navigation_status = "NAVIGATING"
            self.publish_markers(raw_clusters, [], None, meta)
            return

        if self.state.is_settling(now):
            self.exploration_state_name = "SETTLING"
            self.navigation_status = "SETTLING"
            self.publish_markers(raw_clusters, [], None, meta)
            return

        if self.should_keep_collecting(now, len(raw_clusters)):
            self.exploration_state_name = "COLLECTING_FRONTIERS"
            self.navigation_status = "COLLECTING_FRONTIERS"
            self.publish_markers(raw_clusters, [], None, meta)
            return

        self.exploration_state_name = "PLANNING_GOAL"
        scored_frontiers = self.score_frontiers(map_msg, meta, robot_xy, candidate_clusters)
        if not scored_frontiers:
            self.exploration_state_name = "NO_VALID_FRONTIER"
            self.navigation_status = "NO_VALID_FRONTIER"
            self.log_no_valid_frontiers(len(raw_clusters), len(scored_frontiers))
            self.publish_markers(raw_clusters, [], None, meta)
            return

        selected = max(scored_frontiers, key=lambda item: item.utility)
        self.navigation_status = "IDLE"
        self.publish_markers(raw_clusters, scored_frontiers, selected, meta)
        self.log_selected_frontier(selected)
        self.maybe_send_navigation_goal(selected, robot_xy)

    def lookup_robot_xy(self) -> Optional[Point2D]:
        try:
            transform = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time())
        except TransformException as exc:
            self.log_throttled(
                "tf_unavailable",
                f"Cannot lookup TF {self.global_frame}->{self.robot_frame}: {exc}",
                level="warn",
            )
            return None
        translation = transform.transform.translation
        return float(translation.x), float(translation.y)

    def collect_frontier_candidates(
        self,
        map_msg: OccupancyGrid,
        meta: GridMeta,
        robot_xy: Point2D,
        now: Time,
    ) -> tuple[list[FrontierCluster], list[FrontierCluster]]:
        if self.frontier_source == "rrt":
            return self.collect_rrt_frontiers(map_msg, meta, robot_xy, now)
        clusters = self.detect_bfs_clusters(map_msg, meta)
        self.publish_point_stream(clusters, filtered_clusters=clusters)
        return clusters, clusters

    def collect_rrt_frontiers(
        self,
        map_msg: OccupancyGrid,
        meta: GridMeta,
        robot_xy: Point2D,
        now: Time,
    ) -> tuple[list[FrontierCluster], list[FrontierCluster]]:
        raw_points = self.rrt_detector.detect(
            map_msg.data,
            meta,
            robot_xy,
            include_global=self.rrt_include_global,
            include_local=self.rrt_include_local,
        )
        now_sec = now.nanoseconds / 1.0e9
        self.frontier_filter.add_points(raw_points, now_sec)
        buffered_raw_points = self.frontier_filter.raw_points(now_sec)
        filtered_points, rejected = self.frontier_filter.filtered_points(
            now_sec,
            map_msg,
            meta,
            self.latest_costmap,
            self.global_frame,
            costmap_threshold=self.costmap_clearing_threshold,
            allow_unknown_goal=self.allow_unknown_goal,
            info_radius=self.info_radius,
            information_threshold=self.information_threshold,
        )
        for key, value in rejected.items():
            self.last_rejection_counts[key] = self.last_rejection_counts.get(key, 0) + value
        raw_clusters = clusters_from_points(buffered_raw_points, meta)
        filtered_clusters = clusters_from_points(filtered_points, meta)
        self.publish_point_stream(raw_clusters, filtered_clusters=filtered_clusters)
        return raw_clusters, filtered_clusters

    def detect_bfs_clusters(self, map_msg: OccupancyGrid, meta: GridMeta) -> list[FrontierCluster]:
        return detect_frontier_clusters(
            map_msg.data,
            width=meta.width,
            height=meta.height,
            resolution=meta.resolution,
            origin_x=meta.origin_x,
            origin_y=meta.origin_y,
            min_cluster_size=self.frontier_min_cluster_size,
            connectivity=self.frontier_connectivity,
        )

    def should_keep_collecting(self, now: Time, raw_count: int) -> bool:
        if self.frontier_source != "rrt" or self.algorithm_mode != "aslam_original":
            return False
        elapsed = (now - self.frontier_collection_started).nanoseconds / 1.0e9
        if raw_count >= self.min_raw_frontiers_before_planning and elapsed >= self.frontier_collection_time_sec:
            return False
        if elapsed >= self.max_goal_search_time_sec:
            return False
        self.log_throttled(
            "collecting_frontiers",
            f"Collecting RRT frontiers: raw={raw_count}, elapsed={elapsed:.1f}s",
            level="info",
            period_sec=2.0,
        )
        return True

    def publish_point_stream(
        self,
        raw_clusters: list[FrontierCluster],
        *,
        filtered_clusters: list[FrontierCluster],
    ) -> None:
        if not self.publish_detected_points:
            return
        now_msg = self.get_clock().now().to_msg()
        for cluster in raw_clusters:
            if self.detected_points_pub is not None:
                self.detected_points_pub.publish(self.make_point_stamped(cluster.centroid, now_msg))
        for cluster in filtered_clusters:
            if self.filtered_points_pub is not None:
                self.filtered_points_pub.publish(self.make_point_stamped(cluster.centroid, now_msg))

    def make_point_stamped(self, xy: Point2D, stamp) -> PointStamped:
        msg = PointStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = stamp
        msg.point.x = xy[0]
        msg.point.y = xy[1]
        msg.point.z = 0.0
        return msg

    def score_frontiers(
        self,
        map_msg: OccupancyGrid,
        meta: GridMeta,
        robot_xy: Point2D,
        clusters: list[FrontierCluster],
    ) -> list[ScoredFrontier]:
        robot_cell = world_to_map(robot_xy[0], robot_xy[1], meta)
        if robot_cell is None:
            self.log_throttled("robot_off_map", "Robot pose is outside the occupancy grid.", level="warn")
            return []

        scored: list[ScoredFrontier] = []
        for cluster in clusters:
            frontier_xy = cluster.centroid
            if self.state.is_blacklisted(frontier_xy):
                self.reject_candidate("blacklisted_frontier")
                continue
            if self.state.is_visited(frontier_xy):
                self.reject_candidate("visited_frontier")
                continue

            distance = euclidean_distance(robot_xy, frontier_xy)
            if distance < self.min_frontier_distance:
                self.state.add_to_visited(self.get_clock().now(), frontier_xy)
                self.reject_candidate("too_close_frontier")
                continue

            frontier_cell = world_to_map(frontier_xy[0], frontier_xy[1], meta)
            if frontier_cell is None:
                self.reject_candidate("off_map_frontier")
                continue

            info_gain = information_gain(map_msg, meta, frontier_xy, self.info_radius)
            if info_gain < self.information_threshold:
                self.reject_candidate("low_information_gain")
                continue

            nav_goal = self.compute_navigation_goal(map_msg, meta, robot_xy, robot_cell, frontier_xy)
            if nav_goal is None:
                self.reject_candidate(self._last_goal_rejection_reason)
                continue
            if self.state.is_blacklisted(nav_goal.xy):
                self.reject_candidate("blacklisted_goal")
                continue
            if self.state.is_visited(nav_goal.xy):
                self.reject_candidate("visited_goal")
                continue
            if euclidean_distance(robot_xy, nav_goal.xy) < self.goal_reached_distance:
                now = self.get_clock().now()
                self.state.add_to_visited(now, nav_goal.xy)
                self.state.add_to_visited(now, frontier_xy)
                self.reject_candidate("goal_too_close")
                continue

            if self.algorithm_mode == "aslam_original":
                aslam_utility = compute_aslam_utility(
                    map_msg.data,
                    meta,
                    robot_xy,
                    frontier_xy,
                    nav_goal.planner_path.poses,
                    info_radius=self.info_radius,
                    lambda_decay=self.lambda_decay,
                    unknown_probability=self.unknown_probability,
                    known_probability=self.known_probability,
                )
                path_cells = aslam_utility.path_cells
                entropy_reward = aslam_utility.inv_entropy
                gamma = aslam_utility.gamma
                utility = aslam_utility.utility
                spann = aslam_utility.spann
                eta = aslam_utility.eta
                entropy_value = aslam_utility.entropy
                pose_graph_fallback = aslam_utility.pose_graph_fallback
            else:
                path_cells = bresenham(robot_cell, frontier_cell)
                occupancy_values = path_occupancy_values(map_msg.data, path_cells, meta)
                _, normalized_entropy = compute_path_entropy(
                    occupancy_values,
                    unknown_probability=self.unknown_probability,
                    known_probability=self.known_probability,
                )
                entropy_reward = clamp(1.0 - normalized_entropy, 0.0, 1.0)
                gamma = distance_decay(distance, self.lambda_decay)
                utility = (self.w_entropy * entropy_reward) + (self.w_distance * gamma)
                spann = 0.0
                eta = 1.0
                entropy_value = normalized_entropy
                pose_graph_fallback = False

            scored.append(
                ScoredFrontier(
                    cluster=cluster,
                    path_cells=path_cells,
                    utility=utility,
                    entropy_reward=entropy_reward,
                    distance_reward=gamma,
                    distance=distance,
                    centroid_cell=frontier_cell,
                    nav_goal_xy=nav_goal.xy,
                    nav_goal_cell=nav_goal.cell,
                    nav_path_cells=nav_goal.map_path_cells,
                    planner_path=nav_goal.planner_path,
                    planner_path_length=nav_goal.planner_path_length,
                    information_gain=info_gain,
                    costmap_cost=nav_goal.costmap_cost,
                    used_offset_goal=nav_goal.used_offset,
                    spann=spann,
                    inv_entropy=entropy_reward,
                    eta=eta,
                    gamma=gamma,
                    entropy=entropy_value,
                    pose_graph_fallback=pose_graph_fallback,
                )
            )
        return scored

    def compute_navigation_goal(
        self,
        map_msg: OccupancyGrid,
        meta: GridMeta,
        robot_xy: Point2D,
        robot_cell: Cell,
        frontier_xy: Point2D,
    ) -> Optional[CandidateGoal]:
        self._last_goal_rejection_reason = "goal_invalid"
        centroid_candidate = self.evaluate_goal_candidate(map_msg, meta, robot_xy, robot_cell, frontier_xy, used_offset=False)
        if centroid_candidate is not None:
            return centroid_candidate

        if self.frontier_goal_offset <= 0.0:
            return None

        if self.algorithm_mode == "aslam_original":
            self.log_throttled(
                "aslam_offset_fallback",
                "ROS 2/Nav2 adaptation fallback: centroid failed costmap/planner validation, trying frontier_goal_offset.",
                level="warn",
                period_sec=10.0,
            )
        offset_xy = offset_frontier_toward_robot(frontier_xy, robot_xy, self.frontier_goal_offset)
        offset_cell = world_to_map(offset_xy[0], offset_xy[1], meta)
        if offset_cell is None:
            self._last_goal_rejection_reason = "offset_off_map"
            return None
        free_goal_cell = find_nearest_free_cell(map_msg.data, meta, offset_cell, self.goal_search_radius_cells)
        if free_goal_cell is None:
            self._last_goal_rejection_reason = "no_free_offset_goal"
            return None
        free_goal_xy = map_to_world(free_goal_cell[0], free_goal_cell[1], meta)
        return self.evaluate_goal_candidate(map_msg, meta, robot_xy, robot_cell, free_goal_xy, used_offset=True)

    def evaluate_goal_candidate(
        self,
        map_msg: OccupancyGrid,
        meta: GridMeta,
        robot_xy: Point2D,
        robot_cell: Cell,
        goal_xy: Point2D,
        *,
        used_offset: bool,
    ) -> Optional[CandidateGoal]:
        goal_cell = world_to_map(goal_xy[0], goal_xy[1], meta)
        if goal_cell is None:
            self._last_goal_rejection_reason = "off_map_goal"
            return None

        cost, reason = costmap_value_at(self.latest_costmap, self.global_frame, goal_xy)
        if reason == "waiting_costmap":
            self.log_throttled(
                "waiting_costmap",
                f"Waiting for global costmap on {self.global_costmap_topic}; rejecting candidates for safety.",
                level="warn",
            )
        elif reason == "costmap_frame_mismatch" and self.latest_costmap is not None:
            self.log_throttled(
                "costmap_frame_mismatch",
                f"Global costmap frame '{self.latest_costmap.header.frame_id}' differs from global_frame '{self.global_frame}'.",
                level="warn",
            )
        if cost is None or not is_acceptable_cost(cost, self.costmap_clearing_threshold, self.allow_unknown_goal):
            self._last_goal_rejection_reason = "costmap_rejected"
            return None

        planner_path = self.compute_nav2_path(robot_xy, goal_xy)
        if planner_path is None:
            self._last_goal_rejection_reason = self._last_planner_rejection_reason
            return None
        if not planner_path.poses:
            self._last_goal_rejection_reason = "planner_empty_path"
            return None
        if path_has_blocked_costmap_cell(
            planner_path,
            self.latest_costmap,
            self.global_frame,
            self.costmap_clearing_threshold,
            self.allow_unknown_goal,
        ):
            self._last_goal_rejection_reason = "path_blocked_costmap"
            return None

        map_path_cells = path_cells_from_poses(planner_path.poses, meta)
        if not map_path_cells:
            map_path_cells = bresenham(robot_cell, goal_cell)
        if path_has_occupied_cell(map_msg.data, meta, map_path_cells):
            self._last_goal_rejection_reason = "map_path_occupied"
            return None

        return CandidateGoal(
            xy=goal_xy,
            cell=goal_cell,
            map_path_cells=map_path_cells,
            planner_path=planner_path,
            planner_path_length=path_length(planner_path.poses),
            costmap_cost=cost,
            used_offset=used_offset,
        )

    def compute_nav2_path(self, robot_xy: Point2D, goal_xy: Point2D) -> Optional[Path]:
        self._last_planner_rejection_reason = "planner_failed"
        cached = self.state.get_cached_planner_path(goal_xy)
        if cached is not None:
            return cached
        cached_failure = self.state.get_cached_planner_failure(goal_xy)
        if cached_failure is not None:
            self._last_planner_rejection_reason = cached_failure
            return None

        now = self.get_clock().now()
        ready, reason = self.state.planner_validation_ready(now)
        if not ready:
            self._last_planner_rejection_reason = reason
            return None
        if not self.nav2.wait_for_planner(timeout_sec=0.0):
            self._last_planner_rejection_reason = "planner_unavailable"
            self.log_throttled(
                "planner_unavailable",
                f"ComputePathToPose action server '{self.planner_action_name}' is not available; rejecting frontier candidates until planner is available.",
                level="warn",
            )
            return None

        self.state.start_planner_request(now)
        planner_path, reason = self.nav2.request_path_blocking(
            robot_xy,
            goal_xy,
            timeout_sec=self.planner_request_timeout_sec,
            log_throttled=self.log_throttled,
        )
        self.state.finish_planner_request()
        self._last_planner_rejection_reason = reason or "planner_failed"
        self.state.cache_planner_result(self.get_clock().now(), goal_xy, planner_path, self._last_planner_rejection_reason)
        return planner_path

    def publish_markers(
        self,
        raw_clusters: list[FrontierCluster],
        scored_frontiers: list[ScoredFrontier],
        selected: Optional[ScoredFrontier],
        meta: GridMeta,
    ) -> None:
        if not self.debug_markers:
            return
        self.markers.publish_markers(
            raw_clusters,
            scored_frontiers,
            selected,
            meta,
            visited_goals=self.state.visited_goals,
            active_goal_xy=self.active_goal_xy,
            navigation_status=self.navigation_status,
        )

    def maybe_send_navigation_goal(self, selected: ScoredFrontier, robot_xy: Point2D) -> None:
        if not self.enable_navigation:
            self.navigation_status = "marker_only"
            return
        if self.active_goal_handle is not None:
            return

        now = self.get_clock().now()
        if (now - self.last_goal_send_time).nanoseconds / 1.0e9 < self.goal_send_period_sec:
            return
        if not self.nav2.wait_for_navigation(timeout_sec=0.0):
            self.navigation_status = "nav2_unavailable"
            self.log_throttled(
                "nav2_unavailable",
                f"NavigateToPose action server '{self.navigate_action_name}' is not available.",
                level="warn",
            )
            return
        if self.active_goal_xy is not None and euclidean_distance(selected.nav_goal_xy, self.active_goal_xy) < self.min_goal_separation:
            return

        retry_count = self.state.goal_retries.get(selected.nav_goal_cell, 0)
        if retry_count > self.max_goal_retries:
            self.state.add_to_blacklist(now, selected.nav_goal_xy)
            self.navigation_status = "blacklisted"
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.global_frame
        goal_msg.pose.header.stamp = now.to_msg()
        goal_msg.pose.pose.position.x = selected.nav_goal_xy[0]
        goal_msg.pose.pose.position.y = selected.nav_goal_xy[1]
        goal_msg.pose.pose.position.z = 0.0
        yaw = math.atan2(selected.cluster.centroid[1] - robot_xy[1], selected.cluster.centroid[0] - robot_xy[0])
        (
            goal_msg.pose.pose.orientation.x,
            goal_msg.pose.pose.orientation.y,
            goal_msg.pose.pose.orientation.z,
            goal_msg.pose.pose.orientation.w,
        ) = yaw_to_quaternion(yaw)

        self.navigation_status = "sending"
        self.last_goal_send_time = now
        self.active_goal_xy = selected.nav_goal_xy
        self.active_goal_sent_time = now
        self.state.goal_retries[selected.nav_goal_cell] = retry_count + 1
        self.get_logger().info(
            f"Sending NavigateToPose goal to ({selected.nav_goal_xy[0]:.2f}, {selected.nav_goal_xy[1]:.2f}) "
            f"for frontier ({selected.cluster.centroid[0]:.2f}, {selected.cluster.centroid[1]:.2f}), "
            f"utility={selected.utility:.3f}, spann={selected.spann:.3f}, "
            f"inv_entropy={selected.inv_entropy:.3f}, eta={selected.eta:.3f}, gamma={selected.gamma:.3f}, "
            f"info_gain={selected.information_gain:.2f}, cost={selected.costmap_cost}, "
            f"plan={selected.planner_path_length:.2f} m"
        )
        send_future = self.nav2.send_navigation_goal(goal_msg, feedback_callback=self.navigation_feedback_callback)
        send_future.add_done_callback(lambda future, scored=selected: self.navigation_goal_response_callback(future, scored))

    def navigation_goal_response_callback(self, future, selected: ScoredFrontier) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                f"NavigateToPose goal rejected at ({selected.nav_goal_xy[0]:.2f}, {selected.nav_goal_xy[1]:.2f}); trying another candidate on next cycle."
            )
            self.navigation_status = "rejected"
            self.active_goal_handle = None
            self.active_goal_xy = None
            self.active_goal_sent_time = None
            now = self.get_clock().now()
            self.state.add_to_blacklist(now, selected.nav_goal_xy, timeout_sec=self.rejected_blacklist_timeout_sec)
            self.state.add_to_blacklist(now, selected.cluster.centroid, timeout_sec=self.rejected_blacklist_timeout_sec)
            return

        self.active_goal_handle = goal_handle
        self.navigation_status = "navigating"
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda result, scored=selected: self.navigation_result_callback(result, scored))

    def navigation_result_callback(self, future, selected: ScoredFrontier) -> None:
        result = future.result()
        status = result.status
        self.active_goal_handle = None
        self.active_goal_xy = None
        self.active_goal_sent_time = None
        now = self.get_clock().now()

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.navigation_status = "SETTLING"
            self.state.start_settling(now)
            self.frontier_collection_started = now
            self.state.add_to_visited(now, selected.nav_goal_xy)
            self.state.add_to_visited(now, selected.cluster.centroid)
            self.get_logger().info(
                f"NavigateToPose succeeded at ({selected.nav_goal_xy[0]:.2f}, {selected.nav_goal_xy[1]:.2f}); "
                f"settling for {self.post_goal_settle_time_sec:.1f}s"
            )
            return

        self.navigation_status = goal_status_name(status).lower()
        self.get_logger().warn(
            f"NavigateToPose finished with {goal_status_name(status)} at "
            f"({selected.nav_goal_xy[0]:.2f}, {selected.nav_goal_xy[1]:.2f}); temporarily blacklisting this candidate."
        )
        self.state.add_to_blacklist(now, selected.nav_goal_xy, timeout_sec=self.blacklist_timeout_sec)
        self.state.add_to_blacklist(now, selected.cluster.centroid, timeout_sec=self.blacklist_timeout_sec)

    def navigation_feedback_callback(self, feedback_msg) -> None:
        _ = feedback_msg

    def update_navigation_state(self, robot_xy: Point2D) -> None:
        if self.active_goal_xy is None:
            return
        if euclidean_distance(robot_xy, self.active_goal_xy) <= self.goal_reached_distance:
            self.navigation_status = "NAVIGATING"
        if self.active_goal_sent_time is None:
            return

        elapsed = (self.get_clock().now() - self.active_goal_sent_time).nanoseconds / 1.0e9
        if elapsed > self.goal_timeout_sec:
            self.navigation_status = "timeout"
            self.get_logger().warn("NavigateToPose goal timed out locally; canceling and clearing active goal.")
            if self.active_goal_handle is not None:
                self.active_goal_handle.cancel_goal_async()
            self.state.add_to_blacklist(self.get_clock().now(), self.active_goal_xy, timeout_sec=self.blacklist_timeout_sec)
            self.active_goal_handle = None
            self.active_goal_xy = None
            self.active_goal_sent_time = None

    def is_navigating(self) -> bool:
        return self.active_goal_xy is not None or self.active_goal_handle is not None

    def reject_candidate(self, reason: str) -> None:
        self.last_rejection_counts[reason] = self.last_rejection_counts.get(reason, 0) + 1

    def log_no_valid_frontiers(self, raw_count: int, accepted_count: int) -> None:
        rejected = ", ".join(f"{key}={value}" for key, value in sorted(self.last_rejection_counts.items()))
        if not rejected:
            rejected = "none"
        self.log_throttled(
            "no_frontiers",
            f"No valid frontier candidates found: raw={raw_count}, accepted={accepted_count}, rejected={{{rejected}}}",
            level="info",
        )

    def log_selected_frontier(self, selected: ScoredFrontier) -> None:
        now = self.get_clock().now()
        elapsed = (now - self.last_selected_log_time).nanoseconds / 1.0e9
        if selected.centroid_cell != self.last_selected_cell or elapsed >= 10.0:
            self.get_logger().info(
                "Selected frontier at "
                f"({selected.cluster.centroid[0]:.2f}, {selected.cluster.centroid[1]:.2f}), "
                f"utility={selected.utility:.3f}, spann={selected.spann:.3f}, "
                f"inv_entropy={selected.inv_entropy:.3f}, eta={selected.eta:.3f}, "
                f"gamma={selected.gamma:.3f}, distance={selected.distance:.2f} m, "
                f"info_gain={selected.information_gain:.2f}, cost={selected.costmap_cost}, "
                f"planner_path={selected.planner_path_length:.2f} m"
            )
            self.last_selected_cell = selected.centroid_cell
            self.last_selected_log_time = now

    def log_throttled(self, key: str, message: str, *, level: str = "info", period_sec: float = 5.0) -> None:
        now = self.get_clock().now()
        last = self._last_throttle_log.get(key)
        if last is not None and (now - last).nanoseconds / 1.0e9 < period_sec:
            return
        self._last_throttle_log[key] = now
        logger = self.get_logger()
        if level == "warn":
            logger.warn(message)
        else:
            logger.info(message)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def offset_frontier_toward_robot(frontier_xy: Point2D, robot_xy: Point2D, offset: float) -> Point2D:
    if offset <= 0.0:
        return frontier_xy
    dx = robot_xy[0] - frontier_xy[0]
    dy = robot_xy[1] - frontier_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= 1.0e-9:
        return frontier_xy
    scale = min(offset, distance) / distance
    return frontier_xy[0] + (dx * scale), frontier_xy[1] + (dy * scale)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ActiveSlamExplorer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
