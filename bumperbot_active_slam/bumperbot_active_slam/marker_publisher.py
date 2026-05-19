"""RViz marker publishing for Active SLAM debug visualization."""

from __future__ import annotations

from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

from bumperbot_active_slam.active_slam_types import ScoredFrontier, VisitedGoal
from bumperbot_active_slam.entropy_utils import Cell, GridMeta, Point2D, map_to_world
from bumperbot_active_slam.frontier_detector import FrontierCluster


class ActiveSlamMarkerPublisher:
    def __init__(self, node, *, global_frame: str, marker_topic: str, debug_visited_goals: bool) -> None:
        self.node = node
        self.global_frame = global_frame
        self.debug_visited_goals = debug_visited_goals
        self.publisher = node.create_publisher(MarkerArray, marker_topic, 10)

    def publish_markers(
        self,
        raw_clusters: list[FrontierCluster],
        scored_frontiers: list[ScoredFrontier],
        selected: ScoredFrontier | None,
        meta: GridMeta,
        *,
        visited_goals: list[VisitedGoal],
        active_goal_xy: Point2D | None,
        navigation_status: str,
    ) -> None:
        markers = MarkerArray()
        markers.markers.append(self.make_delete_all_marker())
        now = self.node.get_clock().now().to_msg()
        marker_id = 1

        for cluster in raw_clusters:
            marker = self.make_cube_marker(
                marker_id,
                "raw_frontiers",
                cluster.centroid,
                scale=0.08,
                color=(1.0, 0.85, 0.0, 0.55),
            )
            marker.header.stamp = now
            markers.markers.append(marker)
            marker_id += 1

        for scored in scored_frontiers:
            marker = self.make_cube_marker(
                marker_id,
                "filtered_frontiers",
                scored.cluster.centroid,
                scale=0.14,
                color=(0.1, 0.35, 1.0, 0.85),
            )
            marker.header.stamp = now
            markers.markers.append(marker)
            marker_id += 1

        if self.debug_visited_goals:
            for visited in visited_goals:
                marker = self.make_cube_marker(
                    marker_id,
                    "visited_goals",
                    visited.xy,
                    scale=0.18,
                    color=(0.7, 0.7, 0.7, 0.7),
                )
                marker.header.stamp = now
                markers.markers.append(marker)
                marker_id += 1

        if selected is not None:
            selected_marker = self.make_cube_marker(
                marker_id,
                "selected_frontier",
                selected.cluster.centroid,
                scale=0.32,
                color=(1.0, 0.1, 0.05, 1.0),
            )
            selected_marker.header.stamp = now
            markers.markers.append(selected_marker)
            marker_id += 1

            entropy_path_marker = self.make_path_marker(
                marker_id,
                "selected_entropy_ray",
                selected.path_cells,
                meta,
                color=(0.1, 0.9, 0.25, 1.0),
                scale=0.035,
            )
            entropy_path_marker.header.stamp = now
            markers.markers.append(entropy_path_marker)
            marker_id += 1

            planner_path_marker = self.make_planner_path_marker(marker_id, selected.planner_path)
            planner_path_marker.header.stamp = now
            markers.markers.append(planner_path_marker)
            marker_id += 1

            goal_marker = self.make_cube_marker(
                marker_id,
                "navigation_goal",
                selected.nav_goal_xy,
                scale=0.22,
                color=(0.0, 1.0, 0.2, 1.0),
            )
            goal_marker.header.stamp = now
            markers.markers.append(goal_marker)
            marker_id += 1

            text_marker = self.make_text_marker(marker_id, selected)
            text_marker.header.stamp = now
            markers.markers.append(text_marker)
            marker_id += 1

            nav_marker = self.make_nav_status_marker(marker_id, selected.cluster.centroid, navigation_status)
            nav_marker.header.stamp = now
            markers.markers.append(nav_marker)
        else:
            nav_marker = self.make_nav_status_marker(
                marker_id,
                self.status_marker_xy(raw_clusters, active_goal_xy),
                navigation_status,
            )
            nav_marker.header.stamp = now
            markers.markers.append(nav_marker)

        self.publisher.publish(markers)

    def status_marker_xy(self, raw_clusters: list[FrontierCluster], active_goal_xy: Point2D | None) -> Point2D:
        if active_goal_xy is not None:
            return active_goal_xy
        if raw_clusters:
            return raw_clusters[0].centroid
        return 0.0, 0.0

    def make_delete_all_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        return marker

    def make_cube_marker(
        self,
        marker_id: int,
        namespace: str,
        xy: Point2D,
        *,
        scale: float,
        color: tuple[float, float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = xy[0]
        marker.pose.position.y = xy[1]
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    def make_path_marker(
        self,
        marker_id: int,
        namespace: str,
        path_cells: list[Cell],
        meta: GridMeta,
        *,
        color: tuple[float, float, float, float],
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        for cell in path_cells:
            x, y = map_to_world(cell[0], cell[1], meta)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.04
            marker.points.append(point)
        return marker

    def make_planner_path_marker(self, marker_id: int, path: Path) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = "selected_planner_path"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06
        marker.color.r = 0.85
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = 1.0
        for pose in path.poses:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = 0.08
            marker.points.append(point)
        return marker

    def make_text_marker(self, marker_id: int, selected: ScoredFrontier) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = "selected_utility"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = selected.cluster.centroid[0]
        marker.pose.position.y = selected.cluster.centroid[1]
        marker.pose.position.z = 0.45
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.22
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = (
            f"U={selected.utility:.3f} spann={selected.spann:.3f}\n"
            f"invH={selected.inv_entropy:.3f} eta={selected.eta:.3f} gamma={selected.gamma:.3f}\n"
            f"IG={selected.information_gain:.2f} cost={selected.costmap_cost} plan={selected.planner_path_length:.2f}m\n"
            f"goal=({selected.nav_goal_xy[0]:.2f},{selected.nav_goal_xy[1]:.2f}) "
            f"{'offset' if selected.used_offset_goal else 'centroid'}"
        )
        return marker

    def make_nav_status_marker(self, marker_id: int, xy: Point2D, navigation_status: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = "navigation_status"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = xy[0]
        marker.pose.position.y = xy[1]
        marker.pose.position.z = 0.75
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.2
        marker.color.r = 0.2
        marker.color.g = 1.0
        marker.color.b = 0.6
        marker.color.a = 1.0
        marker.text = f"STATE: {navigation_status}"
        return marker
